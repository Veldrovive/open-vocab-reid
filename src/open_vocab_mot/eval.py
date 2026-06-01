import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

from open_vocab_mot.data.video_reid_abc import AbstractVideoReIDDataset, IdentityId, SequenceId
from open_vocab_mot.models.pipeline import AbstractVideoReIDPipeline

@dataclass
class IdentityEmbeddingMap:
    # identity_id -> sequence_id -> list of embeddings
    embeddings: dict[IdentityId, dict[SequenceId, list[torch.Tensor]]] = field(default_factory=dict)
    
    def add(self, identity_id: IdentityId, sequence_id: SequenceId, embedding: torch.Tensor):
        if identity_id not in self.embeddings:
            self.embeddings[identity_id] = {}
        if sequence_id not in self.embeddings[identity_id]:
            self.embeddings[identity_id][sequence_id] = []
        self.embeddings[identity_id][sequence_id].append(embedding)

@dataclass
class ExtractionResult:
    video_embeddings: IdentityEmbeddingMap
    frame_embeddings: Optional[IdentityEmbeddingMap] = None


class _EvalTaskDataset(Dataset):
    def __init__(self, base_dataset: AbstractVideoReIDDataset, tasks: list[tuple[IdentityId, SequenceId, list[int]]]):
        self.base_dataset = base_dataset
        self.tasks = tasks
        
    def __len__(self):
        return len(self.tasks)
        
    def __getitem__(self, idx):
        identity_id, sequence_id, frame_indices = self.tasks[idx]
        items = [self.base_dataset[i] for i in frame_indices]
        
        frames = [item.frame_tensor for item in items]
        segmentations = [item.segmentation_tensor for item in items]
        
        return identity_id, sequence_id, frames, segmentations

def _eval_task_collate_fn(batch):
    identity_ids = [item[0] for item in batch]
    sequence_ids = [item[1] for item in batch]
    frames = [item[2] for item in batch]
    segmentations = [item[3] for item in batch]
    return identity_ids, sequence_ids, frames, segmentations


@torch.no_grad()
def extract_embeddings(
    dataset: AbstractVideoReIDDataset,
    model: AbstractVideoReIDPipeline,
    accelerator: Accelerator,
    frames_per_video_embedding: int,
    num_video_embeddings_per_sequence: int,
    return_frame_embeddings: bool = False,
    max_frame_embeddings_per_sequence: Optional[int] = None,
    seed: int = 42,
    batch_size: int = 8,
    num_workers: int = 4
) -> ExtractionResult:
    """
    Extracts video and optionally frame embeddings from a ReID dataset.
    """
    device = accelerator.device
    model.eval()

    video_tasks: list[tuple[IdentityId, SequenceId, list[int]]] = []
    frame_tasks: list[tuple[IdentityId, SequenceId, list[int]]] = []

    if accelerator.is_main_process:
        print("Generating extraction tasks...")

    unique_identities = dataset.unique_identities
    seq_map = dataset.sequence_map

    for identity_id in unique_identities:
        for sequence_id, available_indices in seq_map[identity_id].items():
            if len(available_indices) == 0:
                continue

            # --- Video Tasks ---
            rng = random.Random(seed + identity_id + sequence_id)
            pool = list(available_indices)
            rng.shuffle(pool)

            for _ in range(num_video_embeddings_per_sequence):
                selected = []
                while len(selected) < frames_per_video_embedding:
                    if not pool:
                        pool = list(available_indices)
                        rng.shuffle(pool)
                    needed = frames_per_video_embedding - len(selected)
                    selected.extend(pool[:needed])
                    pool = pool[needed:]
                video_tasks.append((identity_id, sequence_id, selected))

            # --- Frame Tasks ---
            if return_frame_embeddings:
                rng_frame = random.Random(seed + identity_id + sequence_id + 1000)
                frame_pool = list(available_indices)
                if max_frame_embeddings_per_sequence is not None and len(frame_pool) > max_frame_embeddings_per_sequence:
                    frame_pool = rng_frame.sample(frame_pool, max_frame_embeddings_per_sequence)
                
                for ds_idx in frame_pool:
                    # Create a task for a single frame
                    frame_tasks.append((identity_id, sequence_id, [ds_idx]))

    def process_tasks(tasks, desc, extract_video, extract_frames):
        if len(tasks) == 0:
            return IdentityEmbeddingMap()
            
        task_ds = _EvalTaskDataset(dataset, tasks)
        loader = DataLoader(
            task_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_eval_task_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loader = accelerator.prepare(loader)
        local_results = []

        progress = tqdm(loader, desc=desc, disable=not accelerator.is_main_process)
        for identity_ids, sequence_ids, frames, segmentations in progress:
            # Move frames to device if necessary. In our model, pipeline usually expects cpu/gpu tensors and handles it,
            # but let's pass them as is, the pipeline logic should handle `.to(device)`
            
            output = model(
                frames=frames,
                segmentations=segmentations,
                extract_video=extract_video,
                extract_frames=extract_frames
            )
            
            # Since we extract either video or frame
            if extract_video:
                emb_batch = output["video_contrastive_embeddings"]
            else:
                # If extract_frames, since each task is 1 frame, we get 1 frame embedding per video task
                emb_batch = [f_embs[0] for f_embs in output["frame_contrastive_embeddings"]]
                emb_batch = torch.stack(emb_batch)
                
            emb_batch = emb_batch.cpu()
            
            for i in range(len(identity_ids)):
                local_results.append((identity_ids[i], sequence_ids[i], emb_batch[i]))
                
        # Gather across GPUs
        gathered_results = gather_object([local_results])
        
        final_map = IdentityEmbeddingMap()
        for res_list in gathered_results:
            for id_, seq_, emb in res_list:
                final_map.add(id_, seq_, emb)
                
        return final_map

    video_map = process_tasks(video_tasks, "Extracting Video Embeddings", extract_video=True, extract_frames=False)
    frame_map = None
    if return_frame_embeddings:
        frame_map = process_tasks(frame_tasks, "Extracting Frame Embeddings", extract_video=False, extract_frames=True)

    return ExtractionResult(video_embeddings=video_map, frame_embeddings=frame_map)


@torch.no_grad()
def evaluate_reid(
    query_map: IdentityEmbeddingMap,
    key_map: IdentityEmbeddingMap,
    same_source: bool,
    sim_aggregation: str = "max",
    device: str = "cuda"
) -> dict:
    """
    Evaluates Query vs Key IdentityEmbeddingMaps.
    
    If same_source=True, masks out similarities between the exact same embeddings.
    """
    def flatten_map(emb_map: IdentityEmbeddingMap):
        embs = []
        meta = []
        for pid, seq_dict in emb_map.embeddings.items():
            for sid, emb_list in seq_dict.items():
                for emb_idx, emb in enumerate(emb_list):
                    embs.append(emb)
                    meta.append((pid, sid, emb_idx))
        return torch.stack(embs) if embs else torch.empty(0), meta

    query_embs, query_meta = flatten_map(query_map)
    key_embs, key_meta = flatten_map(key_map)

    if len(query_embs) == 0 or len(key_embs) == 0:
        print("Warning: Empty query or key map. Returning empty metrics.")
        return {}

    query_embs = F.normalize(query_embs.to(device), p=2, dim=-1)
    key_embs = F.normalize(key_embs.to(device), p=2, dim=-1)

    print("Computing embedding-level pairwise similarities...")
    pairwise_sims = torch.einsum('qd,kd->qk', query_embs, key_embs)
    pairwise_sims = pairwise_sims.cpu()

    q_pids = torch.tensor([meta[0] for meta in query_meta])
    q_sids = torch.tensor([meta[1] for meta in query_meta])
    q_eidx = torch.tensor([meta[2] for meta in query_meta])

    k_pids = torch.tensor([meta[0] for meta in key_meta])
    k_sids = torch.tensor([meta[1] for meta in key_meta])
    k_eidx = torch.tensor([meta[2] for meta in key_meta])

    N_Q = len(query_meta)

    print("\n--- Standard Video-to-Video Evaluation (Standard CMC/mAP) ---")
    cmc_video = torch.zeros(N_Q)
    ap_video = torch.zeros(N_Q)
    valid_queries_video = 0

    for q_idx in range(N_Q):
        pid = q_pids[q_idx].item()
        sid = q_sids[q_idx].item()
        eidx = q_eidx[q_idx].item()

        if same_source:
            # Mask out the exact same identity, sequence, and embedding index
            exclude_mask = (k_pids == pid) & (k_sids == sid) & (k_eidx == eidx)
        else:
            # Mask out same sequence ID for the same person (common practice in cross-camera ReID)
            exclude_mask = (k_pids == pid) & (k_sids == sid)

        sims = pairwise_sims[q_idx].clone()
        sims[exclude_mask] = -1e9

        truth_indices = (k_pids == pid) & ~exclude_mask
        num_k_truth = truth_indices.sum().item()

        if num_k_truth == 0:
            continue

        valid_queries_video += 1

        sorted_indices = torch.argsort(sims, descending=True)
        sorted_truth = truth_indices[sorted_indices]

        first_match_rank = torch.where(sorted_truth)[0][0].item()
        cmc_video[first_match_rank:] += 1

        correct_ranks = torch.where(sorted_truth)[0]
        precision_at_ranks = (torch.arange(1, len(correct_ranks) + 1, dtype=torch.float32) / 
                               (correct_ranks.float() + 1))
        ap_video[q_idx] = precision_at_ranks.mean()

    cmc_video = cmc_video / valid_queries_video if valid_queries_video > 0 else torch.zeros(N_Q)
    map_video = ap_video.sum() / valid_queries_video if valid_queries_video > 0 else torch.tensor(0.0)

    print(f"Rank-1 Accuracy:  {cmc_video[0].item() * 100:.2f}%" if len(cmc_video) > 0 else "Rank-1 Accuracy:  N/A")
    print(f"Rank-5 Accuracy:  {cmc_video[4].item() * 100:.2f}%" if len(cmc_video) > 4 else "Rank-5 Accuracy:  N/A")
    print(f"Rank-10 Accuracy: {cmc_video[9].item() * 100:.2f}%" if len(cmc_video) > 9 else "Rank-10 Accuracy: N/A")
    print(f"mAP:              {map_video.item() * 100:.2f}%")

    print("\n--- Person-to-Identity Evaluation (Max over gallery videos) ---")
    gallery_people = sorted(list(set(k_pids.tolist())))
    cmc_identity = torch.zeros(N_Q)
    valid_queries_identity = 0

    for q_idx in range(N_Q):
        pid = q_pids[q_idx].item()
        sid = q_sids[q_idx].item()
        eidx = q_eidx[q_idx].item()

        id_sims = []
        id_list = []

        for g_pid in gallery_people:
            person_mask = (k_pids == g_pid)
            if g_pid == pid:
                if same_source:
                    # Exclude the exact same video embedding from the identity max
                    person_mask = person_mask & ~((k_sids == sid) & (k_eidx == eidx))
                else:
                    # Exclude the same sequence from the identity max
                    person_mask = person_mask & (k_sids != sid)

            if person_mask.sum() == 0:
                continue

            person_sim = pairwise_sims[q_idx, person_mask].max().item()
            id_sims.append(person_sim)
            id_list.append(g_pid)

        if pid not in id_list:
            continue

        valid_queries_identity += 1

        id_sims_t = torch.tensor(id_sims)
        id_list_t = torch.tensor(id_list)

        sorted_indices = torch.argsort(id_sims_t, descending=True)
        sorted_ids = id_list_t[sorted_indices]

        first_match_rank = torch.where(sorted_ids == pid)[0][0].item()
        cmc_identity[first_match_rank:] += 1

    cmc_identity = cmc_identity / valid_queries_identity if valid_queries_identity > 0 else torch.zeros(N_Q)

    print(f"Rank-1 Accuracy:  {cmc_identity[0].item() * 100:.2f}%" if len(cmc_identity) > 0 else "Rank-1 Accuracy:  N/A")
    print(f"Rank-5 Accuracy:  {cmc_identity[4].item() * 100:.2f}%" if len(cmc_identity) > 4 else "Rank-5 Accuracy:  N/A")
    print(f"Rank-10 Accuracy: {cmc_identity[9].item() * 100:.2f}%" if len(cmc_identity) > 9 else "Rank-10 Accuracy: N/A")

    return {
        "video": {
            "cmc": cmc_video,
            "mAP": map_video,
            "rank_1": cmc_video[0].item() if len(cmc_video) > 0 else 0,
            "rank_5": cmc_video[4].item() if len(cmc_video) > 4 else 0,
            "rank_10": cmc_video[9].item() if len(cmc_video) > 9 else 0
        },
        "identity": {
            "cmc": cmc_identity,
            "rank_1": cmc_identity[0].item() if len(cmc_identity) > 0 else 0,
            "rank_5": cmc_identity[4].item() if len(cmc_identity) > 4 else 0,
            "rank_10": cmc_identity[9].item() if len(cmc_identity) > 9 else 0
        }
    }
