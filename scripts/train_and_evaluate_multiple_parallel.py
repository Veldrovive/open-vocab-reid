#!/usr/bin/env python3
import os
import sys
import random
import dataclasses
from pathlib import Path
from typing import TypedDict, Optional
from jaxtyping import Float

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, BaseModel

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import gather_object
from torch.utils.checkpoint import checkpoint

# Import project utilities and dataset classes
from open_vocab_mot.data.whale_ds import (
    WhaleDataset,
    WhaleSplit
)
from open_vocab_mot.data.duke_mtmc_video_ds import (
    DukeMTMCVideoDataset,
    DukeSplit,
    DukeCameraId,
    DukePersonId
)
from open_vocab_mot.data import (
    Wildlife10KSubsetDataset,
    Wildlife10KSplit,
    Wildlife10KDatasets
)
from open_vocab_mot.data.video_reid_abc import (
    VideoReIDBatch,
    collate_video_reid_ds,
    VideoReIDKPFBatchIterableDataset
)
from open_vocab_mot.definitions import (
    DUKEMTMC_VIDEO_REID_PATH, 
    DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
    WHALE_DATASET_PATH,
    WHALE_DATASET_SIDECAR_PATH,
    WILDLIFE_10K_PATH,
    WILDLIFE_10K_SIDECAR_PATH
)
from aidan_lib.models.dino_lib_compiled import DINOv3CompiledHarness

from open_vocab_mot.models import HierarchicalVideoReIDTransformer
from open_vocab_mot.losses import CircleLossWithUnknowns


@torch.no_grad()
def process_val_ds(
    ds: Dataset,
    model: HierarchicalVideoReIDTransformer,
    dino_harness: DINOv3CompiledHarness,
    accelerator: Accelerator,
    frames_per_video: int,
    batch_size: int = 64,
    video_batch_size: int = 64,
    num_video_embeddings_per_video: int = 1,
    dino_batch_split: int = 1,
    split_single_sequences: bool = False
):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_video_reid_ds,
        pin_memory=True,
        num_workers=8
    )
    
    loader = accelerator.prepare(loader)
    device = accelerator.device

    # Handle potentially wrapped model
    unwrapped_model = accelerator.unwrap_model(model)
    frame_embedding_size = unwrapped_model.frame_transformer_dim
    frame_contrastive_embedding_size = unwrapped_model.frame_contrastive_dim

    if accelerator.is_main_process:
        print("Creating buffers...")
    frame_data = []
    frame_embeddings_list = []
    frame_contrastive_embeddings_list = []
    if accelerator.is_main_process:
        print("Created buffers successfully.")

    progress = tqdm(loader, desc="Processing Frames", disable=not accelerator.is_main_process)
    batch: VideoReIDBatch
    for batch in progress:
        if accelerator.is_main_process:
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
        
        if accelerator.is_main_process:
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

        if accelerator.is_main_process:
            progress.set_description_str("Embedding frames")
            
        batch_frame_embedding_data = unwrapped_model.embed_frames([dino_embedding_video], device=device)

        batch_frame_contrastive_embeddings = batch_frame_embedding_data["video_frame_contrastive_embeddings"][0].to(device="cpu")
        batch_frame_cls_tokens = batch_frame_embedding_data["video_frame_cls_tokens"][0].to(device="cpu")

        assert len(batch_frame_contrastive_embeddings) == len(batch_frame_cls_tokens) == len(batch_frame_data)
        for frame_contrastive_embedding, frame_cls_token, this_frame_data in zip(batch_frame_contrastive_embeddings, batch_frame_cls_tokens, batch_frame_data):
            frame_embeddings_list.append(frame_cls_token)
            frame_contrastive_embeddings_list.append(frame_contrastive_embedding)
            frame_data.append(this_frame_data)

        del dino_embeddings, dino_embedding_video
        del batch_frame_embedding_data, batch_frame_contrastive_embeddings, batch_frame_cls_tokens
        del batch

    # Gather frame data across GPUs
    local_frame_embeddings = torch.stack(frame_embeddings_list) if len(frame_embeddings_list) > 0 else torch.empty((0, frame_embedding_size), dtype=torch.float32)
    local_frame_contrastive_embeddings = torch.stack(frame_contrastive_embeddings_list) if len(frame_contrastive_embeddings_list) > 0 else torch.empty((0, frame_contrastive_embedding_size), dtype=torch.float32)
    
    local_data = {
        "frame_embeddings": local_frame_embeddings,
        "frame_contrastive_embeddings": local_frame_contrastive_embeddings,
        "frame_data": frame_data
    }
    
    gathered_data = gather_object([local_data])
    
    all_frame_embeddings_list = []
    all_frame_contrastive_embeddings_list = []
    all_frame_data = []
    
    for d in gathered_data:
        if d["frame_embeddings"].numel() > 0:
            all_frame_embeddings_list.append(d["frame_embeddings"])
            all_frame_contrastive_embeddings_list.append(d["frame_contrastive_embeddings"])
        all_frame_data.extend(d["frame_data"])
        
    if len(all_frame_embeddings_list) > 0:
        frame_embeddings = torch.cat(all_frame_embeddings_list, dim=0)
        frame_contrastive_embeddings = torch.cat(all_frame_contrastive_embeddings_list, dim=0)
    else:
        frame_embeddings = torch.empty((0, frame_embedding_size), dtype=torch.float32)
        frame_contrastive_embeddings = torch.empty((0, frame_contrastive_embedding_size), dtype=torch.float32)
    
    frame_data = all_frame_data

    # Map from person to camera to list of frame indices
    frame_index_map: dict[int, dict[int, list[int]]] = {}
    for frame_idx, (person_id, camera_id) in enumerate(frame_data):
        if person_id not in frame_index_map:
            frame_index_map[person_id] = {}
        camera_index_map = frame_index_map[person_id]

        if camera_id not in camera_index_map:
            camera_index_map[camera_id] = []

        frame_list = camera_index_map[camera_id]
        frame_list.append(frame_idx)

    # Split single sequences
    if split_single_sequences:
        for person_id, camera_index_map in list(frame_index_map.items()):
            if len(camera_index_map) == 1:
                camera_id = list(camera_index_map.keys())[0]
                frame_indices = camera_index_map[camera_id]
                
                if len(frame_indices) >= 2:
                    mid = len(frame_indices) // 2
                    part1 = frame_indices[:mid]
                    part2 = frame_indices[mid:]
                    
                    camera_index_map[camera_id] = part1
                    camera_index_map[camera_id + 100000] = part2

    # Prepare tasks for video embeddings
    video_tasks = []
    for person_id, camera_index_map in frame_index_map.items():
        for camera_id, frame_indices in camera_index_map.items():
            video_tasks.append((person_id, camera_id, frame_indices))
            
    # Split tasks for GPUs
    tasks_per_gpu = len(video_tasks) // accelerator.num_processes
    start_idx = accelerator.process_index * tasks_per_gpu
    end_idx = start_idx + tasks_per_gpu if accelerator.process_index < accelerator.num_processes - 1 else len(video_tasks)
    
    # Ensure remaining tasks are handled by the last process
    if accelerator.process_index == accelerator.num_processes - 1:
        local_tasks = video_tasks[start_idx:]
    else:
        local_tasks = video_tasks[start_idx:end_idx]
    
    video_embedding_index_map: dict[int, dict[int, list[int]]] = {}
    local_video_contrastive_embeddings = []

    rng = random.Random(42)
    video_count = 0
    progress = tqdm(total=len(local_tasks) * num_video_embeddings_per_video, desc="Embedding Videos", disable=not accelerator.is_main_process)
    
    batch_video_data_buffer: list = []
    batch_frame_buffer: list[torch.Tensor] = []
    
    for person_id, camera_id, frame_indices in local_tasks:
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
                video_embedding_data = unwrapped_model.embed_video_frames(batch_frame_buffer, device=device)
                batch_video_contrastive_embeddings = video_embedding_data["video_contrastive_embeddings"]

                for (p_id, c_id, v_idx), video_contrastive_embedding in zip(batch_video_data_buffer, batch_video_contrastive_embeddings):
                    video_embedding_index_map[p_id][c_id].append(v_idx)
                    local_video_contrastive_embeddings.append(video_contrastive_embedding.cpu())

                batch_video_data_buffer.clear()
                batch_frame_buffer.clear()

            video_count += 1
            if accelerator.is_main_process:
                progress.update(1)
    
    if len(batch_video_data_buffer) > 0:
        video_embedding_data = unwrapped_model.embed_video_frames(batch_frame_buffer, device=device)
        batch_video_contrastive_embeddings = video_embedding_data["video_contrastive_embeddings"]

        for (p_id, c_id, v_idx), video_contrastive_embedding in zip(batch_video_data_buffer, batch_video_contrastive_embeddings):
            video_embedding_index_map[p_id][c_id].append(v_idx)
            local_video_contrastive_embeddings.append(video_contrastive_embedding.cpu())

        batch_video_data_buffer.clear()
        batch_frame_buffer.clear()

    local_video_data = {
        "video_contrastive_embeddings": torch.stack(local_video_contrastive_embeddings) if len(local_video_contrastive_embeddings) > 0 else torch.empty((0, unwrapped_model.video_contrastive_dim), dtype=torch.float32),
        "video_embedding_index_map": video_embedding_index_map
    }
    
    gathered_video_data = gather_object([local_video_data])
    
    final_video_contrastive_embeddings_list = []
    final_video_embedding_index_map = {}
    current_video_idx = 0
    
    for d in gathered_video_data:
        embs = d["video_contrastive_embeddings"]
        idx_map = d["video_embedding_index_map"]
        
        if embs.numel() > 0:
            final_video_contrastive_embeddings_list.append(embs)
            
        for person_id, camera_map in idx_map.items():
            if person_id not in final_video_embedding_index_map:
                final_video_embedding_index_map[person_id] = {}
            for camera_id, v_indices in camera_map.items():
                if camera_id not in final_video_embedding_index_map[person_id]:
                    final_video_embedding_index_map[person_id][camera_id] = []
                # Shift indices
                shifted_indices = [idx + current_video_idx for idx in v_indices]
                final_video_embedding_index_map[person_id][camera_id].extend(shifted_indices)
                
        current_video_idx += embs.size(0) if embs.numel() > 0 else 0
        
    if len(final_video_contrastive_embeddings_list) > 0:
        video_contrastive_embeddings = torch.cat(final_video_contrastive_embeddings_list, dim=0)
    else:
        video_contrastive_embeddings = torch.empty((0, unwrapped_model.video_contrastive_dim), dtype=torch.float32)
    
    return {
        "frame_index_map": frame_index_map,
        "video_embedding_index_map": final_video_embedding_index_map,
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
        if sims[sorted_indices[0]] > 0.999:
            print(f"WARNING: q_idx={q_idx} has max sim {sims[sorted_indices[0]].item()} with idx={sorted_indices[0].item()}! Exact same embedding?")
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


@torch.no_grad()
def evaluate_reid(
    processed_val: dict,
    num_embeddings_per_video: int = 1,
    sim_aggregation: str = "max",
    device: str = "cuda"
):
    val_embs, val_meta = group_embeddings(processed_val, num_embeddings_per_video)
    
    val_embs = F.normalize(val_embs.to(device), p=2, dim=-1)
    
    print("Computing embedding-level pairwise similarities...")
    pairwise_sims = torch.einsum('qmd,gnd->qgmn', val_embs, val_embs)
    
    print(f"Aggregating video-to-video similarities using '{sim_aggregation}'...")
    if sim_aggregation == "max":
        video_sims = pairwise_sims.max(dim=-1)[0].max(dim=-1)[0]
    elif sim_aggregation == "mean":
        video_sims = pairwise_sims.mean(dim=(-2, -1))
    else:
        raise ValueError(f"Unknown sim_aggregation: {sim_aggregation}")
    
    video_sims = video_sims.cpu()
    
    pids = torch.tensor([meta[0] for meta in val_meta])
    cids = torch.tensor([meta[1] for meta in val_meta])
    
    N = len(val_meta)
    
    print("\n--- Standard Video-to-Video Evaluation (Standard CMC/mAP) ---")
    cmc_video = torch.zeros(N)
    ap_video = torch.zeros(N)
    valid_queries_video = 0
    
    for q_idx in range(N):
        pid = pids[q_idx]
        cid = cids[q_idx]
        
        exclude_mask = (pids == pid) & (cids == cid)
        
        sims = video_sims[q_idx].clone()
        sims[exclude_mask] = -1e9
        
        truth_indices = (pids == pid) & ~exclude_mask
        num_g_truth = truth_indices.sum().item()
        
        if num_g_truth == 0:
            continue
            
        valid_queries_video += 1
        
        sorted_indices = torch.argsort(sims, descending=True)
        if sims[sorted_indices[0]] > 0.999:
            print(f"WARNING: q_idx={q_idx} has max sim {sims[sorted_indices[0]].item()} with idx={sorted_indices[0].item()}! Exact same embedding?")
        sorted_truth = truth_indices[sorted_indices]
        
        first_match_rank = torch.where(sorted_truth)[0][0].item()
        cmc_video[first_match_rank:] += 1
        
        correct_ranks = torch.where(sorted_truth)[0]
        precision_at_ranks = (torch.arange(1, len(correct_ranks) + 1, dtype=torch.float32) / 
                               (correct_ranks.float() + 1))
        ap_video[q_idx] = precision_at_ranks.mean()
        
    cmc_video = cmc_video / valid_queries_video if valid_queries_video > 0 else torch.zeros(N)
    map_video = ap_video.sum() / valid_queries_video if valid_queries_video > 0 else torch.tensor(0.0)
    
    print(f"Rank-1 Accuracy:  {cmc_video[0].item() * 100:.2f}%" if len(cmc_video) > 0 else "Rank-1 Accuracy:  N/A")
    print(f"Rank-5 Accuracy:  {cmc_video[4].item() * 100:.2f}%" if len(cmc_video) > 4 else "Rank-5 Accuracy:  N/A")
    print(f"Rank-10 Accuracy: {cmc_video[9].item() * 100:.2f}%" if len(cmc_video) > 9 else "Rank-10 Accuracy: N/A")
    print(f"mAP:              {map_video.item() * 100:.2f}%")
    
    print("\n--- Person-to-Identity Evaluation (Max over gallery videos) ---")
    gallery_people = sorted(list(set(pids.tolist())))
    cmc_identity = torch.zeros(N)
    valid_queries_identity = 0
    
    for q_idx in range(N):
        pid = pids[q_idx].item()
        cid = cids[q_idx].item()
        
        id_sims = []
        id_list = []
        
        for g_pid in gallery_people:
            person_mask = (pids == g_pid)
            if g_pid == pid:
                person_mask = person_mask & (cids != cid)
                
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
        
    cmc_identity = cmc_identity / valid_queries_identity if valid_queries_identity > 0 else torch.zeros(N)
    
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



class DatasetConfig(BaseModel):
    use_for_training: bool = True
    weight: float = 1.0
    frames_per_video: int = 4
    people_per_batch: int = 16
    views_per_person: int = 3
    use_for_eval: bool = False

class DukeDatasetConfig(DatasetConfig):
    pass

class WhaleDatasetConfig(DatasetConfig):
    min_num_images: int = 0

class Wildlife10kSubsetDatasetConfig(DatasetConfig):
    subset_dataset: Wildlife10KDatasets
    min_num_images: int = 0


class TrainConfig(BaseModel):
    epochs: int = 3
    batches_per_epoch: int = 300
    lr: float = 1e-4
    frame_loss_weight: float = 0.85
    seed: int = 42
    dino_checkpoint: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    dino_batch_split: int = 1

    device: str = "cuda"
    cuda_visible_devices: Optional[str] = None

    checkpoint_path: str = "weights/reid_transformer_latest.pt"
    plot_path: str = "weights/loss_curve.png"

    num_embeddings: int = 3
    skip_eval: bool = False
    eval_batch_size: int = 128

    wandb_project: str = "open-vocab-mot"
    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    no_wandb: bool = False

    frame_transformer_dim: int = 512
    frame_contrastive_dim: int = 256
    frame_num_heads: int = 8
    frame_num_layers: int = 4
    video_transformer_dim: int = 384
    video_contrastive_dim: int = 256
    video_num_heads: int = 6
    video_num_layers: int = 3
    dropout: float = 0.1

    duke: DukeDatasetConfig = Field(default_factory=DukeDatasetConfig)
    whale: WhaleDatasetConfig = Field(default_factory=WhaleDatasetConfig)
    wildlife_subsets: list[Wildlife10kSubsetDatasetConfig] = Field(default_factory=list)

@hydra.main(version_base=None, config_path="../configs/train_and_evaluate_multiple", config_name="config")
def main(cfg: DictConfig):
    """
    Example run command:
    export CUDA_VISIBLE_DEVICES=1,2
    uv run accelerate launch --multi_gpu --num_processes=2 scripts/train_and_evaluate_multiple_parallel.py

    export CUDA_VISIBLE_DEVICES=0
    uv run accelerate launch scripts/train_and_evaluate_multiple_parallel.py
    """
    try:
        args = TrainConfig(**OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        print("Configuration validation error:", e)
        sys.exit(1)

    print(args.model_dump_json(indent=2))

    # Initialize Accelerate with bf16 and even_batches=False
    dataloader_config = DataLoaderConfiguration(even_batches=False)
    accelerator = Accelerator(mixed_precision="bf16", dataloader_config=dataloader_config)
    
    # Initialize wandb
    if accelerator.is_main_process and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config=args.model_dump()
        )
        
    device = accelerator.device
    
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        if accelerator.is_main_process:
            print(f"Forced CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
            
    if accelerator.is_main_process:
        print(f"Using device: {device}")
    
    # Set seeds
    random.seed(args.seed + accelerator.process_index)
    torch.manual_seed(args.seed + accelerator.process_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + accelerator.process_index)
        
    # Ensure outputs directory exists
    if accelerator.is_main_process:
        checkpoint_file = Path(args.checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        plot_file = Path(args.plot_path)
        plot_file.parent.mkdir(parents=True, exist_ok=True)

    duke_loader = None
    if args.duke.use_for_training:
        if accelerator.is_main_process:
            print("Loading DukeMTMC dataset for training...")
        duke_ds = DukeMTMCVideoDataset(
            ds_root=DUKEMTMC_VIDEO_REID_PATH,
            main_split=DukeSplit.TRAIN,
            sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=accelerator.is_main_process
        )
        duke_iterable = VideoReIDKPFBatchIterableDataset(
            duke_ds,
            batches_per_epoch=None,
            num_identities_per_batch=args.duke.people_per_batch,
            num_sequences_per_identity=args.duke.views_per_person,
            num_frames_per_sequence=args.duke.frames_per_video,
            allow_same_identity_same_sequence=True,
            allow_reduced_sequences_per_identity=False,
            allow_resampling_sample_indices=True,
            epoch_deterministic=False,
            seed=args.seed + accelerator.process_index * 100,
            verbose=accelerator.is_main_process
        )
        duke_loader = DataLoader(
            dataset=duke_iterable,
            batch_size=None,
            collate_fn=collate_video_reid_ds,
            num_workers=4,
            pin_memory=True
        )

    whale_loader = None
    if args.whale.use_for_training:
        if accelerator.is_main_process:
            print("Loading Whale dataset for training...")
        whale_ds = WhaleDataset(
            ds_root=WHALE_DATASET_PATH,
            split=WhaleSplit.TRAIN,
            sidecar_root=WHALE_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            min_num_images=args.whale.min_num_images,
            verbose=accelerator.is_main_process
        )
        whale_iterable = VideoReIDKPFBatchIterableDataset(
            whale_ds,
            batches_per_epoch=None,
            num_identities_per_batch=args.whale.people_per_batch,
            num_sequences_per_identity=args.whale.views_per_person,
            num_frames_per_sequence=args.whale.frames_per_video,
            allow_same_identity_same_sequence=True,
            allow_reduced_sequences_per_identity=False,
            allow_resampling_sample_indices=True,
            epoch_deterministic=False,
            seed=args.seed + 1 + accelerator.process_index * 100,
            verbose=accelerator.is_main_process
        )
        whale_loader = DataLoader(
            dataset=whale_iterable,
            batch_size=None,
            collate_fn=collate_video_reid_ds,
            num_workers=4,
            pin_memory=True
        )

    wildlife_loaders = {}
    for subset_cfg in args.wildlife_subsets:
        if subset_cfg.use_for_training:
            subset_name = subset_cfg.subset_dataset
            if accelerator.is_main_process:
                print(f"Loading Wildlife10K subset {subset_name} for training...")
            subset_ds = Wildlife10KSubsetDataset(
                ds_root=WILDLIFE_10K_PATH,
                dataset_name=subset_name,
                split=Wildlife10KSplit.TRAIN,
                sidecar_root=WILDLIFE_10K_SIDECAR_PATH,
                load_image_pil=False,
                load_image_tensor=True,
                load_segmentations=True,
                min_num_images=subset_cfg.min_num_images,
                verbose=accelerator.is_main_process
            )
            subset_iterable = VideoReIDKPFBatchIterableDataset(
                subset_ds,
                batches_per_epoch=None,
                num_identities_per_batch=subset_cfg.people_per_batch,
                num_sequences_per_identity=subset_cfg.views_per_person,
                num_frames_per_sequence=subset_cfg.frames_per_video,
                allow_same_identity_same_sequence=True,
                allow_reduced_sequences_per_identity=False,
                allow_resampling_sample_indices=True,
                epoch_deterministic=False,
                seed=args.seed + 2 + len(wildlife_loaders) + accelerator.process_index * 100,
                verbose=accelerator.is_main_process
            )
            wildlife_loaders[subset_name] = DataLoader(
                dataset=subset_iterable,
                batch_size=None,
                collate_fn=collate_video_reid_ds,
                num_workers=4,
                pin_memory=True
            )
    
    if accelerator.is_main_process:
        print("Loading Compiled DINO Harness...")
        
    dino_harness = DINOv3CompiledHarness(
        checkpoint=args.dino_checkpoint,
        device=device,
        dtype=torch.bfloat16,
        max_side_len=1024,
        warmup=False
    )
    
    if accelerator.is_main_process:
        print("Initializing Hierarchical Video ReID Transformer model...")
        
    model = HierarchicalVideoReIDTransformer(
        input_dim=dino_harness.embedding_dim,
        frame_transformer_dim=args.frame_transformer_dim,
        frame_contrastive_dim=args.frame_contrastive_dim,
        frame_num_heads=args.frame_num_heads,
        frame_num_layers=args.frame_num_layers,
        video_transformer_dim=args.video_transformer_dim,
        video_contrastive_dim=args.video_contrastive_dim,
        video_num_heads=args.video_num_heads,
        video_num_layers=args.video_num_layers,
        dropout=args.dropout
    )
    
    criterion = CircleLossWithUnknowns().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Prepare with accelerate
    prepared = accelerator.prepare(model, optimizer)
    model = prepared[0]
    optimizer = prepared[1]
    
    loaders = {}
    loader_weights = {}
    
    if duke_loader is not None:
        loaders["duke"] = iter(duke_loader)
        loader_weights["duke"] = args.duke.weight
        
    if whale_loader is not None:
        loaders["whale"] = iter(whale_loader)
        loader_weights["whale"] = args.whale.weight
        
    for subset_cfg in args.wildlife_subsets:
        subset_name = subset_cfg.subset_dataset
        if subset_name in wildlife_loaders:
            loaders[subset_name] = iter(wildlife_loaders[subset_name])
            loader_weights[subset_name] = subset_cfg.weight
        
    if not loaders:
        raise ValueError("No datasets enabled for training. Set at least one dataset's 'use' flag to True.")

    dataset_names = [k for k, v in loader_weights.items() if v > 0]
    weights = [loader_weights[k] for k in dataset_names]
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    dataset_rng = random.Random(args.seed)
    
    history = {
        "total_loss": []
    }
    for name in dataset_names:
        history[f"{name}_video_loss"] = []
        history[f"{name}_frame_loss"] = []
        history[f"{name}_total_loss"] = []
    
    if accelerator.is_main_process:
        print("Starting training...")
        
    model.train()
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            
        progress = tqdm(total=args.batches_per_epoch, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        
        for batch_idx in range(args.batches_per_epoch):
            selected_ds_name = dataset_rng.choices(dataset_names, weights=probs, k=1)[0]
            selected_loader = loaders[selected_ds_name]
            
            try:
                batch = next(selected_loader)
            except StopIteration:
                if selected_ds_name == "duke":
                    loaders["duke"] = iter(duke_loader)
                elif selected_ds_name == "whale":
                    loaders["whale"] = iter(whale_loader)
                elif selected_ds_name in wildlife_loaders:
                    loaders[selected_ds_name] = iter(wildlife_loaders[selected_ds_name])
                selected_loader = loaders[selected_ds_name]
                batch = next(selected_loader)
            # 1. Extract DINO embeddings (Gradients disabled for DINO)
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
            accelerator.backward(total_loss)
            optimizer.step()
            
            # Track history and log (only on main process)
            if accelerator.is_main_process:
                history[f"{selected_ds_name}_video_loss"].append(video_loss.item())
                history[f"{selected_ds_name}_frame_loss"].append(frame_loss.item())
                history[f"{selected_ds_name}_total_loss"].append(total_loss.item())
                history["total_loss"].append(total_loss.item())
                
                if not args.no_wandb:
                    wandb.log({
                        "train/loss": total_loss.item(),
                        f"train/{selected_ds_name}/video_loss": video_loss.item(),
                        f"train/{selected_ds_name}/frame_loss": frame_loss.item(),
                        f"train/{selected_ds_name}/loss": total_loss.item(),
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                    })
                
                progress.set_postfix({"Loss": f"{total_loss.item():.4f}", "DS": selected_ds_name})
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx} [{selected_ds_name}]: Total Loss = {total_loss.item():.4f} "
                          f"(Video: {video_loss.item():.4f}, Frame: {frame_loss.item():.4f})")
                
                # Update progress bar
                progress.update(1)
            
            # Explicitly free memory at the end of the batch
            del dino_embeddings
            del video_embeddings
            del video_pos_mask, video_neg_mask, frame_pos_mask, frame_neg_mask
            del reid_output
            del video_loss, frame_loss, total_loss
            del batch

    if accelerator.is_main_process:
        print("\nTraining completed.")
        
        # Save model weights
        print(f"Saving model checkpoint to {args.checkpoint_path}...")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), args.checkpoint_path)
        print("Checkpoint saved.")
        
        # Save loss plot
        print(f"Generating loss plot to {args.plot_path}...")
        plt.figure(figsize=(10, 6))
        plt.plot(history["total_loss"], label="Total Loss", linewidth=2)
        for name in dataset_names:
            if len(history[f"{name}_total_loss"]) > 0:
                plt.plot(history[f"{name}_total_loss"], label=f"{name} Total Loss", alpha=0.5)
        plt.xlabel("Batch Index")
        plt.ylabel("Loss")
        plt.title(f"Training Loss over Time (Frame Weight = {args.frame_loss_weight})")
        plt.legend()
        plt.grid(True)
        plt.savefig(args.plot_path)
        plt.close()
        print("Loss plot saved.")
        
    if args.skip_eval:
        if accelerator.is_main_process:
            print("Evaluation phase skipped as requested.")
        return
        
    # Wait for main process to finish saving before eval
    accelerator.wait_for_everyone()
        
    if accelerator.is_main_process:
        print("\n--- Starting Evaluation ---")
        
    model.eval()
    
    if args.duke.use_for_eval:
        if accelerator.is_main_process:
            print("Loading DukeMTMC test sets for evaluation...")
        gallery_ds = DukeMTMCVideoDataset(
            ds_root=DUKEMTMC_VIDEO_REID_PATH,
            main_split=DukeSplit.GALLERY,
            sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=accelerator.is_main_process
        )
        query_ds = DukeMTMCVideoDataset(
            ds_root=DUKEMTMC_VIDEO_REID_PATH,
            main_split=DukeSplit.QUERY,
            sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=accelerator.is_main_process
        )
        
        if accelerator.is_main_process:
            print("\nProcessing Duke Query Dataset...")
        processed_query_ds = process_val_ds(
            query_ds,
            model,
            dino_harness,
            accelerator,
            frames_per_video=args.duke.frames_per_video,
            batch_size=args.eval_batch_size,
            num_video_embeddings_per_video=args.num_embeddings,
            dino_batch_split=args.dino_batch_split,
            split_single_sequences=False
        )
        
        if accelerator.is_main_process:
            print("\nProcessing Duke Gallery Dataset...")
        processed_gallery_ds = process_val_ds(
            gallery_ds,
            model,
            dino_harness,
            accelerator,
            frames_per_video=args.duke.frames_per_video,
            batch_size=args.eval_batch_size,
            num_video_embeddings_per_video=args.num_embeddings,
            dino_batch_split=args.dino_batch_split,
            split_single_sequences=False
        )
        
        if accelerator.is_main_process:
            print("\nRunning metrics computation for Duke...")
            eval_metrics = evaluate_duke_reid(
                processed_query_ds,
                processed_gallery_ds,
                num_embeddings_per_video=args.num_embeddings,
                sim_aggregation="max",
                device=device
            )

            if not args.no_wandb:
                wandb.log({
                    "eval/duke/video_rank_1": eval_metrics["video"]["rank_1"],
                    "eval/duke/video_rank_5": eval_metrics["video"]["rank_5"],
                    "eval/duke/video_rank_10": eval_metrics["video"]["rank_10"],
                    "eval/duke/video_mAP": eval_metrics["video"]["mAP"],
                    "eval/duke/identity_rank_1": eval_metrics["identity"]["rank_1"],
                    "eval/duke/identity_rank_5": eval_metrics["identity"]["rank_5"],
                    "eval/duke/identity_rank_10": eval_metrics["identity"]["rank_10"],
                })

    if args.whale.use_for_eval:
        if accelerator.is_main_process:
            print("\nLoading WhaleDataset VAL set for evaluation...")
        whale_val_ds = WhaleDataset(
            ds_root=WHALE_DATASET_PATH,
            split=WhaleSplit.VAL,
            sidecar_root=WHALE_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            min_num_images=args.whale.min_num_images,
            verbose=accelerator.is_main_process
        )
        
        if accelerator.is_main_process:
            print("\nProcessing Whale Validation Dataset...")
        processed_val_ds = process_val_ds(
            whale_val_ds,
            model,
            dino_harness,
            accelerator,
            frames_per_video=args.whale.frames_per_video,
            batch_size=args.eval_batch_size,
            num_video_embeddings_per_video=args.num_embeddings,
            dino_batch_split=args.dino_batch_split,
            split_single_sequences=True
        )
        
        if accelerator.is_main_process:
            print("\nRunning metrics computation for Whale...")
            eval_metrics = evaluate_reid(
                processed_val_ds,
                num_embeddings_per_video=args.num_embeddings,
                sim_aggregation="max",
                device=device
            )

            if not args.no_wandb:
                wandb.log({
                    "eval/whale/video_rank_1": eval_metrics["video"]["rank_1"],
                    "eval/whale/video_rank_5": eval_metrics["video"]["rank_5"],
                    "eval/whale/video_rank_10": eval_metrics["video"]["rank_10"],
                    "eval/whale/video_mAP": eval_metrics["video"]["mAP"],
                    "eval/whale/identity_rank_1": eval_metrics["identity"]["rank_1"],
                    "eval/whale/identity_rank_5": eval_metrics["identity"]["rank_5"],
                    "eval/whale/identity_rank_10": eval_metrics["identity"]["rank_10"],
                })

    for subset_cfg in args.wildlife_subsets:
        if subset_cfg.use_for_eval:
            subset_name = subset_cfg.subset_dataset
            if accelerator.is_main_process:
                print(f"\nLoading Wildlife10K subset {subset_name} VAL set for evaluation...")
            subset_val_ds = Wildlife10KSubsetDataset(
                ds_root=WILDLIFE_10K_PATH,
                dataset_name=subset_name,
                split=Wildlife10KSplit.VAL,
                sidecar_root=WILDLIFE_10K_SIDECAR_PATH,
                load_image_pil=False,
                load_image_tensor=True,
                load_segmentations=True,
                min_num_images=subset_cfg.min_num_images,
                verbose=accelerator.is_main_process
            )
            
            if accelerator.is_main_process:
                print(f"\nProcessing {subset_name} Validation Dataset...")
            processed_val_ds = process_val_ds(
                subset_val_ds,
                model,
                dino_harness,
                accelerator,
                frames_per_video=subset_cfg.frames_per_video,
                batch_size=args.eval_batch_size,
                num_video_embeddings_per_video=args.num_embeddings,
                dino_batch_split=args.dino_batch_split,
                split_single_sequences=True
            )
            
            if accelerator.is_main_process:
                print(f"\nRunning metrics computation for {subset_name}...")
                eval_metrics = evaluate_reid(
                    processed_val_ds,
                    num_embeddings_per_video=args.num_embeddings,
                    sim_aggregation="max",
                    device=device
                )

                if not args.no_wandb:
                    wandb.log({
                        f"eval/{subset_name}/video_rank_1": eval_metrics["video"]["rank_1"],
                        f"eval/{subset_name}/video_rank_5": eval_metrics["video"]["rank_5"],
                        f"eval/{subset_name}/video_rank_10": eval_metrics["video"]["rank_10"],
                        f"eval/{subset_name}/video_mAP": eval_metrics["video"]["mAP"],
                        f"eval/{subset_name}/identity_rank_1": eval_metrics["identity"]["rank_1"],
                        f"eval/{subset_name}/identity_rank_5": eval_metrics["identity"]["rank_5"],
                        f"eval/{subset_name}/identity_rank_10": eval_metrics["identity"]["rank_10"],
                    })

    if accelerator.is_main_process and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
