#!/usr/bin/env python3
import os
import sys
import random
import argparse
from pathlib import Path
from typing import TypedDict
from jaxtyping import Float

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

# Import project utilities and dataset classes
from open_vocab_mot.data.duke_mtmc_video_ds import (
    DukeMTMCVideoDataset,
    DukeSplit,
    DukeCameraId,
    DukePersonId
)
from open_vocab_mot.data.video_reid_abc import (
    VideoReIDBatch,
    collate_video_reid_ds,
    VideoReIDKPFBatchSampler
)
from open_vocab_mot.definitions import DUKEMTMC_VIDEO_REID_PATH, DUKEMTMC_VIDEO_REID_SIDECAR_PATH
from aidan_lib.models.dino_lib_compiled import DINOv3CompiledHarness


from open_vocab_mot.models import HierarchicalVideoReIDTransformer
from open_vocab_mot.losses import CircleLossWithUnknowns


@torch.no_grad()
def process_duke_ds(
    ds: DukeMTMCVideoDataset,
    model: HierarchicalVideoReIDTransformer,
    dino_harness: DINOv3CompiledHarness,
    device: str,
    frames_per_video: int,
    batch_size: int = 64,
    video_batch_size: int = 64,
    num_video_embeddings_per_video: int = 1,
    dino_batch_split: int = 1
):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_video_reid_ds,
        pin_memory=True,
        num_workers=8
    )

    num_frames = len(ds)
    num_videos = sum(len(ds.frame_map[person_id]) for person_id in ds.frame_map)
    frame_embedding_size = model.frame_transformer_dim
    frame_contrastive_embedding_size = model.frame_contrastive_dim
    video_contrastive_embedding_size = model.video_contrastive_dim

    print("Creating buffers...")
    frame_data = []
    frame_embeddings = torch.zeros((num_frames, frame_embedding_size), device="cpu", dtype=torch.float32)
    frame_contrastive_embeddings = torch.zeros((num_frames, frame_contrastive_embedding_size), device="cpu", dtype=torch.float32)
    video_contrastive_embeddings = torch.zeros((num_videos * num_video_embeddings_per_video, video_contrastive_embedding_size), device="cpu", dtype=torch.float32)
    print("Created buffers successfully.")

    progress = tqdm(loader, desc="Processing Frames")
    frame_count = 0
    batch: VideoReIDBatch
    for batch in progress:
        progress.set_description_str("Running DINO")
        valid_imgs = []
        valid_segs = []
        valid_indices = []
        
        for idx, (img, seg) in enumerate(zip(batch.frame_tensors, batch.segmentations)):
            if seg is not None:
                valid_imgs.append(img)
                valid_segs.append(seg)
                valid_indices.append(idx)

        dino_embeddings = [[] for _ in range(len(batch.frame_tensors))]
        
        if len(valid_imgs) > 0:
            if dino_batch_split > 1:
                valid_embs = []
                num_valid = len(valid_imgs)
                for i in range(dino_batch_split):
                    start_idx = i * num_valid // dino_batch_split
                    end_idx = (i + 1) * num_valid // dino_batch_split
                    if start_idx < end_idx:
                        split_imgs = [e.to(device) for e in valid_imgs[start_idx:end_idx]]
                        split_segs = valid_segs[start_idx:end_idx]
                        valid_embs.extend(dino_harness.match_bool_segmentations_to_dino(split_imgs, split_segs))
                        del split_imgs
            else:
                imgs_torch = [e.to(device) for e in valid_imgs]
                valid_embs = dino_harness.match_bool_segmentations_to_dino(imgs_torch, valid_segs)
                del imgs_torch
            
            for valid_idx, emb in zip(valid_indices, valid_embs):
                dino_embeddings[valid_idx] = emb
        
        progress.set_description_str("Extracting DINO embeddings")
        person_ids, camera_ids = batch.identity_ids, batch.sequence_ids
        batch_frame_data = []
        dino_embedding_video: list[torch.Tensor] = []
        for person_id, camera_id, frame_dino_embedding in zip(person_ids, camera_ids, dino_embeddings):
            person_id = int(person_id)
            camera_id = int(camera_id)

            if len(frame_dino_embedding) == 0:
                print(f"Warning: No segmentation found for a frame in person {person_id} camera {camera_id}")
                continue

            batch_frame_data.append((person_id, camera_id))
            dino_embedding_video.append(frame_dino_embedding[0].dino_embeddings)

        if len(dino_embedding_video) == 0:
            continue

        progress.set_description_str("Embedding frames")
        batch_frame_embedding_data = model.embed_frames([dino_embedding_video], device=device)

        batch_frame_contrastive_embeddings = batch_frame_embedding_data["video_frame_contrastive_embeddings"][0].to(device="cpu")
        batch_frame_cls_tokens = batch_frame_embedding_data["video_frame_cls_tokens"][0].to(device="cpu")

        assert len(batch_frame_contrastive_embeddings) == len(batch_frame_cls_tokens) == len(batch_frame_data)
        for frame_contrastive_embedding, frame_cls_token, this_frame_data in zip(batch_frame_contrastive_embeddings, batch_frame_cls_tokens, batch_frame_data):
            frame_embeddings[frame_count] = frame_cls_token
            frame_contrastive_embeddings[frame_count] = frame_contrastive_embedding
            frame_data.append(this_frame_data)
            frame_count += 1

    # Slice down frame buffers to their true filled size
    frame_embeddings = frame_embeddings[:frame_count]
    frame_contrastive_embeddings = frame_contrastive_embeddings[:frame_count]

    # Map from person to camera to list of frame indices
    frame_index_map: dict[DukePersonId, dict[DukeCameraId, list[int]]] = {}
    for frame_idx, (person_id, camera_id) in enumerate(frame_data):
        if person_id not in frame_index_map:
            frame_index_map[person_id] = {}
        camera_index_map = frame_index_map[person_id]

        if camera_id not in camera_index_map:
            camera_index_map[camera_id] = []

        frame_list = camera_index_map[camera_id]
        frame_list.append(frame_idx)

    # Convert into the format that we can pass into the video embedding layer
    video_embedding_index_map: dict[DukePersonId, dict[DukeCameraId, list[int]]] = {}

    rng = random.Random(42)
    video_count = 0
    progress = tqdm(total=num_videos * num_video_embeddings_per_video, desc="Embedding Videos")
    batch_video_data_buffer: list = []
    batch_frame_buffer: list[torch.Tensor] = []
    
    for person_id, camera_index_map in frame_index_map.items():
        for camera_id, frame_indices in camera_index_map.items():
            num_frames = min(len(frame_indices), frames_per_video)
            for _ in range(num_video_embeddings_per_video):
                sampled_frame_indices = rng.sample(frame_indices, k=num_frames)
                video_frame_tensor = torch.stack([frame_embeddings[i] for i in sampled_frame_indices]).to(device=device)

                batch_video_data_buffer.append((person_id, camera_id, video_count))
                batch_frame_buffer.append(video_frame_tensor)

                if person_id not in video_embedding_index_map:
                    video_embedding_index_map[person_id] = {}
                
                if camera_id not in video_embedding_index_map[person_id]:
                    video_embedding_index_map[person_id][camera_id] = []

                if len(batch_video_data_buffer) == video_batch_size:
                    video_embedding_data = model.embed_video_frames(batch_frame_buffer, device=device)
                    batch_video_contrastive_embeddings = video_embedding_data["video_contrastive_embeddings"]

                    for (p_id, c_id, v_idx), video_contrastive_embedding in zip(batch_video_data_buffer, batch_video_contrastive_embeddings):
                        video_embedding_index_map[p_id][c_id].append(v_idx)
                        video_contrastive_embeddings[v_idx] = video_contrastive_embedding.cpu()

                    batch_video_data_buffer.clear()
                    batch_frame_buffer.clear()

                video_count += 1
                progress.update(1)
    
    if len(batch_video_data_buffer) > 0:
        video_embedding_data = model.embed_video_frames(batch_frame_buffer, device=device)
        batch_video_contrastive_embeddings = video_embedding_data["video_contrastive_embeddings"]

        for (p_id, c_id, v_idx), video_contrastive_embedding in zip(batch_video_data_buffer, batch_video_contrastive_embeddings):
            video_embedding_index_map[p_id][c_id].append(v_idx)
            video_contrastive_embeddings[v_idx] = video_contrastive_embedding.cpu()

        batch_video_data_buffer.clear()
        batch_frame_buffer.clear()

    # Slice down video buffer to its true filled size
    video_contrastive_embeddings = video_contrastive_embeddings[:video_count]
    
    return {
        "frame_index_map": frame_index_map,
        "video_embedding_index_map": video_embedding_index_map,
        "frame_data": frame_data,
        "frame_embeddings": frame_embeddings,
        "frame_contrastive_embeddings": frame_contrastive_embeddings,
        "video_contrastive_embeddings": video_contrastive_embeddings
    }


def group_embeddings(processed_ds, num_embeddings_per_video):
    video_embeddings = processed_ds["video_contrastive_embeddings"]
    video_map = processed_ds["video_embedding_index_map"]
    
    grouped_list = []
    metadata = []
    
    for person_id, camera_map in video_map.items():
        for camera_id, indices in camera_map.items():
            assert len(indices) == num_embeddings_per_video, \
                f"Expected {num_embeddings_per_video} embeddings, but got {len(indices)}"
            video_embs = torch.stack([video_embeddings[idx] for idx in indices])
            grouped_list.append(video_embs)
            metadata.append((person_id, camera_id))
            
    return torch.stack(grouped_list), metadata


@torch.no_grad()
def evaluate_duke_reid(
    processed_query: dict,
    processed_gallery: dict,
    num_embeddings_per_video: int = 1,
    sim_aggregation: str = "max",
    device: str = "cuda"
):
    query_embs, query_meta = group_embeddings(processed_query, num_embeddings_per_video)
    gallery_embs, gallery_meta = group_embeddings(processed_gallery, num_embeddings_per_video)
    
    query_embs = F.normalize(query_embs.to(device), p=2, dim=-1)
    gallery_embs = F.normalize(gallery_embs.to(device), p=2, dim=-1)
    
    print("Computing embedding-level pairwise similarities...")
    pairwise_sims = torch.einsum('qmd,gnd->qgmn', query_embs, gallery_embs)
    
    print(f"Aggregating video-to-video similarities using '{sim_aggregation}'...")
    if sim_aggregation == "max":
        video_sims = pairwise_sims.max(dim=-1)[0].max(dim=-1)[0]
    elif sim_aggregation == "mean":
        video_sims = pairwise_sims.mean(dim=(-2, -1))
    else:
        raise ValueError(f"Unknown sim_aggregation: {sim_aggregation}")
    
    video_sims = video_sims.cpu()
    
    q_pids = torch.tensor([meta[0] for meta in query_meta])
    q_cids = torch.tensor([meta[1] for meta in query_meta])
    g_pids = torch.tensor([meta[0] for meta in gallery_meta])
    g_cids = torch.tensor([meta[1] for meta in gallery_meta])
    
    N_Q = len(query_meta)
    
    print("\n--- Standard Video-to-Video Evaluation (Standard CMC/mAP) ---")
    cmc_video = torch.zeros(N_Q)
    ap_video = torch.zeros(N_Q)
    valid_queries_video = 0
    
    for q_idx in range(N_Q):
        pid = q_pids[q_idx]
        cid = q_cids[q_idx]
        
        exclude_mask = (g_pids == pid) & (g_cids == cid)
        
        sims = video_sims[q_idx].clone()
        sims[exclude_mask] = -1e9
        
        truth_indices = (g_pids == pid) & ~exclude_mask
        num_g_truth = truth_indices.sum().item()
        
        if num_g_truth == 0:
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
        
    cmc_video = cmc_video / valid_queries_video
    map_video = ap_video.sum() / valid_queries_video
    
    print(f"Rank-1 Accuracy:  {cmc_video[0].item() * 100:.2f}%")
    print(f"Rank-5 Accuracy:  {cmc_video[4].item() * 100:.2f}%")
    print(f"Rank-10 Accuracy: {cmc_video[9].item() * 100:.2f}%")
    print(f"mAP:              {map_video.item() * 100:.2f}%")
    
    print("\n--- Person-to-Identity Evaluation (Max over gallery videos) ---")
    gallery_people = sorted(list(set(g_pids.tolist())))
    cmc_identity = torch.zeros(N_Q)
    valid_queries_identity = 0
    
    for q_idx in range(N_Q):
        pid = q_pids[q_idx].item()
        cid = q_cids[q_idx].item()
        
        id_sims = []
        id_list = []
        
        for g_pid in gallery_people:
            person_mask = (g_pids == g_pid)
            if g_pid == pid:
                person_mask = person_mask & (g_cids != cid)
                
            if person_mask.sum() == 0:
                continue
                
            person_sim = video_sims[q_idx, person_mask].max().item()
            id_sims.append(person_sim)
            id_list.append(g_pid)
            
        if pid not in id_list:
            continue
            
        valid_queries_identity += 1
        
        id_sims = torch.tensor(id_sims)
        id_list = torch.tensor(id_list)
        
        sorted_indices = torch.argsort(id_sims, descending=True)
        sorted_ids = id_list[sorted_indices]
        
        first_match_rank = torch.where(sorted_ids == pid)[0][0].item()
        cmc_identity[first_match_rank:] += 1
        
    cmc_identity = cmc_identity / valid_queries_identity
    
    print(f"Rank-1 Accuracy:  {cmc_identity[0].item() * 100:.2f}%")
    print(f"Rank-5 Accuracy:  {cmc_identity[4].item() * 100:.2f}%")
    print(f"Rank-10 Accuracy: {cmc_identity[9].item() * 100:.2f}%")
    
    return {
        "video": {
            "cmc": cmc_video,
            "mAP": map_video,
            "rank_1": cmc_video[0].item(),
            "rank_5": cmc_video[4].item(),
            "rank_10": cmc_video[9].item()
        },
        "identity": {
            "cmc": cmc_identity,
            "rank_1": cmc_identity[0].item(),
            "rank_5": cmc_identity[4].item(),
            "rank_10": cmc_identity[9].item()
        }
    }


def main():
    from aidan_lib.models.dino_lib_compiled import DINOv3Checkpoint
    parser = argparse.ArgumentParser(description="Train and evaluate Hierarchical Video ReID Transformer")
    
    # Training configurations
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batches-per-epoch", type=int, default=300, help="Number of batches per epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--frame-loss-weight", type=float, default=0.85, help="Frame loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dino-checkpoint", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m", help="DINO checkpoint")
    parser.add_argument("--dino-batch-split", type=int, default=2, help="Number of splits for the DINO embedding batch to save GPU memory")
    
    # Batch sampler configurations
    parser.add_argument("--frames-per-video", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--people-per-batch", type=int, default=16, help="Number of people per batch")
    parser.add_argument("--views-per-person", type=int, default=3, help="Number of views per person")
    
    # Hardware/Env settings
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use")
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="Force CUDA_VISIBLE_DEVICES env variable")
    
    # Output paths
    parser.add_argument("--checkpoint-path", type=str, default="weights/reid_transformer_latest.pt", help="Path to save trained weights")
    parser.add_argument("--plot-path", type=str, default="weights/loss_curve.png", help="Path to save loss plot")
    
    # Eval configurations
    parser.add_argument("--num-embeddings", type=int, default=3, help="Number of random video samples per video for evaluation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation phase after training")
    
    # Wandb configurations
    parser.add_argument("--wandb-project", type=str, default="open-vocab-mot", help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Wandb entity (username or team)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Set CUDA_VISIBLE_DEVICES if provided
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Forced CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
        
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    print(f"Using device: {device}")
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Ensure outputs directory exists
    checkpoint_file = Path(args.checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    plot_file = Path(args.plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    print("Loading DukeMTMC dataset for training...")
    train_ds = DukeMTMCVideoDataset(
        ds_root=DUKEMTMC_VIDEO_REID_PATH,
        main_split=DukeSplit.TRAIN,
        sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
        load_image_pil=False,
        load_image_tensor=True,
        load_segmentations=True,
        verbose=True
    )
    
    print("Setting up Batch Sampler...")
    train_sampler = VideoReIDKPFBatchSampler(
        train_ds,
        batches_per_epoch=args.batches_per_epoch,
        num_identities_per_batch=args.people_per_batch,
        num_sequences_per_identity=args.views_per_person,
        num_frames_per_sequence=args.frames_per_video,
        allow_same_identity_same_sequence=True,
        allow_reduced_sequences_per_identity=False,
        allow_resampling_sample_indices=True,
        epoch_deterministic=False,
        seed=args.seed,
        verbose=True
    )
    
    train_loader = DataLoader(
        dataset=train_ds,
        collate_fn=collate_video_reid_ds,
        batch_sampler=train_sampler,
        num_workers=8,
        pin_memory=True
    )
    
    print("Loading Compiled DINO Harness...")
    dino_harness = DINOv3CompiledHarness(
        checkpoint=args.dino_checkpoint,
        device=device,
        dtype=torch.bfloat16,
        max_side_len=1024,
        warmup=False
    )
    
    print("Initializing Hierarchical Video ReID Transformer model...")
    model = HierarchicalVideoReIDTransformer(
        input_dim=dino_harness.embedding_dim,
        frame_transformer_dim=256,
        frame_contrastive_dim=256,
        frame_num_heads=4,
        frame_num_layers=3,
        video_transformer_dim=256,
        video_contrastive_dim=256,
        video_num_heads=4,
        video_num_layers=3,
        dropout=0.1
    ).to(device)
    
    criterion = CircleLossWithUnknowns()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    history = {
        "video_loss": [],
        "frame_loss": [],
        "total_loss": []
    }
    
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress):
            # 1. Extract DINO embeddings (Gradients disabled for DINO)
            num_frames = len(batch.frame_tensors)
            with torch.no_grad():
                valid_imgs = []
                valid_segs = []
                valid_indices = []
                
                for idx, (img, seg) in enumerate(zip(batch.frame_tensors, batch.segmentations)):
                    if seg is not None:
                        valid_imgs.append(img)
                        valid_segs.append(seg)
                        valid_indices.append(idx)

                dino_embeddings = [[] for _ in range(len(batch.frame_tensors))]
                
                if len(valid_imgs) > 0:
                    if args.dino_batch_split > 1:
                        valid_embs = []
                        num_valid = len(valid_imgs)
                        for i in range(args.dino_batch_split):
                            start_idx = i * num_valid // args.dino_batch_split
                            end_idx = (i + 1) * num_valid // args.dino_batch_split
                            if start_idx < end_idx:
                                split_imgs = [e.to(device) for e in valid_imgs[start_idx:end_idx]]
                                split_segs = valid_segs[start_idx:end_idx]
                                valid_embs.extend(dino_harness.match_bool_segmentations_to_dino(split_imgs, split_segs))
                                del split_imgs
                    else:
                        imgs_torch = [e.to(device) for e in valid_imgs]
                        valid_embs = dino_harness.match_bool_segmentations_to_dino(imgs_torch, valid_segs)
                        del imgs_torch
                    
                    for valid_idx, emb in zip(valid_indices, valid_embs):
                        dino_embeddings[valid_idx] = emb
                
            # 2. Group embeddings by video (person_id, camera_id) and track person_ids
            video_indices: dict[tuple[int, int], int] = {}
            video_embeddings: list[list[torch.Tensor]] = []
            video_person_ids: list[int] = []
            
            for sample_idx in range(len(batch.identity_ids)):
                # Skip frame if no segmentations/embeddings were found
                if len(dino_embeddings[sample_idx]) == 0:
                    continue
                    
                video_key = (int(batch.identity_ids[sample_idx]), int(batch.sequence_ids[sample_idx]))
                
                if video_key not in video_indices:
                    video_index = len(video_embeddings)
                    video_embeddings.append([])
                    video_indices[video_key] = video_index
                    video_person_ids.append(int(batch.identity_ids[sample_idx]))
                    
                video_index = video_indices[video_key]
                video_embeddings[video_index].append(
                    dino_embeddings[sample_idx][0].dino_embeddings
                )
                
            # Skip batch if we don't have enough valid videos to form pairs
            if len(video_embeddings) < 2:
                continue
                
            # 3. Create Positive and Negative Masks
            # --- Video level ---
            video_pids = torch.tensor(video_person_ids, device=device)
            video_pos_mask = (video_pids.unsqueeze(0) == video_pids.unsqueeze(1))
            video_neg_mask = ~video_pos_mask
            video_pos_mask.fill_diagonal_(False)
            video_neg_mask.fill_diagonal_(False)
            
            # --- Frame level ---
            flat_frame_person_ids = []
            for v_idx, frames in enumerate(video_embeddings):
                flat_frame_person_ids.extend([video_person_ids[v_idx]] * len(frames))
                
            frame_pids = torch.tensor(flat_frame_person_ids, device=device)
            frame_pos_mask = (frame_pids.unsqueeze(0) == frame_pids.unsqueeze(1))
            frame_neg_mask = ~frame_pos_mask
            frame_pos_mask.fill_diagonal_(False)
            frame_neg_mask.fill_diagonal_(False)
            
            # 4. Forward Pass through REID Transformer
            optimizer.zero_grad()
            reid_output = model(video_embeddings)
            
            # 5. Compute Losses
            # Video contrastive loss
            video_embeddings_out = reid_output["video_contrastive_embeddings"]
            video_loss = criterion(video_embeddings_out, video_pos_mask, video_neg_mask)
            
            # Frame contrastive loss
            frame_embeddings_out = torch.cat(reid_output["video_frame_contrastive_embeddings"], dim=0)
            frame_loss = criterion(frame_embeddings_out, frame_pos_mask, frame_neg_mask)
            
            # Combine losses
            total_loss = (1.0 - args.frame_loss_weight) * video_loss + args.frame_loss_weight * frame_loss
            
            # 6. Backward Pass and Optimize
            total_loss.backward()
            optimizer.step()
            
            # Track history
            history["video_loss"].append(video_loss.item())
            history["frame_loss"].append(frame_loss.item())
            history["total_loss"].append(total_loss.item())
            
            if not args.no_wandb:
                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/video_loss": video_loss.item(),
                    "train/frame_loss": frame_loss.item(),
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                })
            
            progress.set_postfix({"Loss": f"{total_loss.item():.4f}"})
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Total Loss = {total_loss.item():.4f} "
                      f"(Video: {video_loss.item():.4f}, Frame: {frame_loss.item():.4f})")

    print("\nTraining completed.")
    
    # Save model weights
    print(f"Saving model checkpoint to {args.checkpoint_path}...")
    torch.save(model.state_dict(), args.checkpoint_path)
    print("Checkpoint saved.")
    
    # Save loss plot
    print(f"Generating loss plot to {args.plot_path}...")
    plt.figure(figsize=(10, 6))
    plt.plot(history["video_loss"], label="Video Loss", alpha=0.7)
    plt.plot(history["frame_loss"], label="Frame Loss", alpha=0.7)
    plt.plot(history["total_loss"], label="Total Loss", linewidth=2)
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.title(f"Training Loss over Time (Frame Weight = {args.frame_loss_weight})")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.plot_path)
    plt.close()
    print("Loss plot saved.")
    
    if args.skip_eval:
        print("Evaluation phase skipped as requested.")
        return
        
    print("\n--- Starting Evaluation ---")
    model.eval()
    
    print("Loading DukeMTMC dataset splits for evaluation...")
    gallery_ds = DukeMTMCVideoDataset(
        ds_root=DUKEMTMC_VIDEO_REID_PATH,
        main_split=DukeSplit.GALLERY,
        sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
        load_image_pil=False,
        load_image_tensor=True,
        load_segmentations=True,
        verbose=True
    )
    
    query_ds = DukeMTMCVideoDataset(
        ds_root=DUKEMTMC_VIDEO_REID_PATH,
        main_split=DukeSplit.QUERY,
        sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
        load_image_pil=False,
        load_image_tensor=True,
        load_segmentations=True,
        verbose=True
    )
    
    print("\nProcessing Query Dataset...")
    processed_query_ds = process_duke_ds(
        query_ds,
        model,
        dino_harness,
        device=device,
        frames_per_video=args.frames_per_video,
        batch_size=512,
        num_video_embeddings_per_video=args.num_embeddings,
        dino_batch_split=args.dino_batch_split
    )
    
    print("\nProcessing Gallery Dataset...")
    processed_gallery_ds = process_duke_ds(
        gallery_ds,
        model,
        dino_harness,
        device=device,
        frames_per_video=args.frames_per_video,
        batch_size=512,
        num_video_embeddings_per_video=args.num_embeddings,
        dino_batch_split=args.dino_batch_split
    )
    
    print("\nRunning metrics computation...")
    eval_metrics = evaluate_duke_reid(
        processed_query_ds,
        processed_gallery_ds,
        num_embeddings_per_video=args.num_embeddings,
        sim_aggregation="max",
        device=device
    )

    if not args.no_wandb:
        wandb.log({
            "eval/video_rank_1": eval_metrics["video"]["rank_1"],
            "eval/video_rank_5": eval_metrics["video"]["rank_5"],
            "eval/video_rank_10": eval_metrics["video"]["rank_10"],
            "eval/video_mAP": eval_metrics["video"]["mAP"],
            "eval/identity_rank_1": eval_metrics["identity"]["rank_1"],
            "eval/identity_rank_5": eval_metrics["identity"]["rank_5"],
            "eval/identity_rank_10": eval_metrics["identity"]["rank_10"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
