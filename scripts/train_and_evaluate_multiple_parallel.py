#!/usr/bin/env python3

"""
Example run command:
export CUDA_VISIBLE_DEVICES=2,3
uv run accelerate launch --multi_gpu --num_processes=2 scripts/train_and_evaluate_multiple_parallel.py

export CUDA_VISIBLE_DEVICES=0
uv run accelerate launch scripts/train_and_evaluate_multiple_parallel.py
"""

import os
import sys
import time
import random
import dataclasses
from pathlib import Path
from typing import TypedDict, Optional
from jaxtyping import Float

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, BaseModel

import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import functional as tv_F

class RandomRatioCrop(v2.Transform):
    def __init__(self, ratio_range=(0.9, 1.0)):
        super().__init__()
        self.ratio_range = ratio_range

    def forward(self, *inputs):
        h, w = None, None
        flat_inputs = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], tuple) else inputs
        for img in flat_inputs:
            if hasattr(img, "shape") and len(img.shape) >= 2:
                h, w = img.shape[-2], img.shape[-1]
                break
        
        if h is None or w is None:
            return inputs if len(inputs) > 1 else inputs[0]
        
        ratio = random.uniform(*self.ratio_range)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        outputs = []
        for inpt in flat_inputs:
            if hasattr(inpt, "shape") and len(inpt.shape) >= 2:
                outputs.append(tv_F.crop(inpt, top, left, new_h, new_w))
            else:
                outputs.append(inpt)
                
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import gather_object
from torch.utils.checkpoint import checkpoint

# Import project utilities and dataset classes
from open_vocab_mot.data import (
    Wildlife10KDatasets,
    DatasetConfig,
    DukeDatasetConfig,
    WhaleDatasetConfig,
    Wildlife10kSubsetDatasetConfig,
    VeRiDatasetConfig,
    VRAIDatasetConfig,
    load_dataset_for_training,
    load_dataset_for_eval
)
from open_vocab_mot.data.video_reid_abc import (
    VideoReIDBatch,
    collate_video_reid_ds,
    VideoReIDKPFBatchIterableDataset
)
from aidan_lib.models.dino_lib_compiled import DINOv3CompiledHarness
from aidan_lib.utils import set_seed

from open_vocab_mot.models import HierarchicalVideoReIDTransformer, NestedHierarchicalVideoReIDTransformer, HierarchicalReIDPipeline
from open_vocab_mot.losses import CircleLossWithUnknowns
from open_vocab_mot.eval import extract_embeddings, evaluate_reid, get_eval_identities_by_max_embeddings


class StepProfiler:
    def __init__(self, device):
        self.device = device
        self.starts = {}
        self.ends = {}
    
    def start(self, name):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.starts[name] = event
        else:
            self.starts[name] = time.perf_counter()
            
    def stop(self, name):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.ends[name] = event
        else:
            self.ends[name] = time.perf_counter()
            
    def elapsed(self, name):
        if name not in self.starts or name not in self.ends:
            return 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize() # Ensure events are recorded and executed
            return self.starts[name].elapsed_time(self.ends[name]) / 1000.0
        else:
            return self.ends[name] - self.starts[name]

def compute_batch_loss(batch, dino_harness, model, criterion, device, frame_loss_weight, cross_loss_weight, dino_batch_split, profiler=None):
    if profiler is not None:
        profiler.start("dino")
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
        
    if profiler is not None:
        profiler.stop("dino")
        profiler.start("forward")
        
    # 2. Group embeddings by video (person_id, camera_id) and track person_ids
    video_indices: dict[tuple[int, int], int] = {}
    video_embeddings: list[list[torch.Tensor]] = []
    video_person_ids: list[int] = []
    
    for sample_idx in range(len(batch.identity_ids)):
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
        
    if len(video_embeddings) < 2:
        if profiler is not None:
            profiler.stop("forward")
        return None, None, None, None
        
    # 3. Create Positive and Negative Masks
    video_pids = torch.tensor(video_person_ids, device=device)
    video_pos_mask = (video_pids.unsqueeze(0) == video_pids.unsqueeze(1))
    video_neg_mask = ~video_pos_mask
    video_pos_mask.fill_diagonal_(False)
    video_neg_mask.fill_diagonal_(False)
    
    flat_frame_person_ids = []
    for v_idx, frames in enumerate(video_embeddings):
        flat_frame_person_ids.extend([video_person_ids[v_idx]] * len(frames))
        
    frame_pids = torch.tensor(flat_frame_person_ids, device=device)
    frame_pos_mask = (frame_pids.unsqueeze(0) == frame_pids.unsqueeze(1))
    frame_neg_mask = ~frame_pos_mask
    frame_pos_mask.fill_diagonal_(False)
    frame_neg_mask.fill_diagonal_(False)
    
    # 4. Forward Pass
    reid_output = model(video_embeddings)
    
    # 5. Compute Losses
    video_embeddings_out = reid_output["video_contrastive_embeddings"]
    video_loss = criterion(video_embeddings_out, video_pos_mask, video_neg_mask)
    
    frame_embeddings_out = torch.cat(reid_output["video_frame_contrastive_embeddings"], dim=0)
    frame_loss = criterion(frame_embeddings_out, frame_pos_mask, frame_neg_mask)
    
    mixed_embeddings = torch.cat([video_embeddings_out, frame_embeddings_out], dim=0)
    mixed_pids = torch.cat([video_pids, frame_pids], dim=0)
    
    mixed_pos_mask = (mixed_pids.unsqueeze(0) == mixed_pids.unsqueeze(1))
    mixed_neg_mask = ~mixed_pos_mask
    mixed_pos_mask.fill_diagonal_(False)
    mixed_neg_mask.fill_diagonal_(False)
    
    num_v = len(video_pids)
    num_f = len(frame_pids)
    
    cross_mask = torch.ones((num_v + num_f, num_v + num_f), dtype=torch.bool, device=device)
    cross_mask[:num_v, :num_v] = False
    cross_mask[num_v:, num_v:] = False
    
    mixed_pos_mask = mixed_pos_mask & cross_mask
    mixed_neg_mask = mixed_neg_mask & cross_mask
    
    cross_loss = criterion(mixed_embeddings, mixed_pos_mask, mixed_neg_mask)
    
    video_weight = 1.0 - frame_loss_weight - cross_loss_weight
    total_loss = video_weight * video_loss + frame_loss_weight * frame_loss + cross_loss_weight * cross_loss
    
    if profiler is not None:
        profiler.stop("forward")
        
    return video_loss, frame_loss, cross_loss, total_loss

class ColorJitterConfig(BaseModel):
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1

class AugmentationConfig(BaseModel):
    enable: bool = True
    crop_ratio_range: tuple[float, float] = (0.9, 1.0)
    random_flip_prop: float = 0.5

    color_jitter_config: ColorJitterConfig = Field(default_factory=ColorJitterConfig)

class TrainConfig(BaseModel):
    epochs: int = 3
    batches_per_epoch: int = 300
    lr: float = 1e-4
    frame_loss_weight: float = 0.85
    cross_loss_weight: float = 0.05
    seed: int = 42
    dino_checkpoint: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    dino_batch_split: int = 1

    device: str = "cuda"
    cuda_visible_devices: Optional[str] = None

    checkpoint_path: str = "weights/reid_transformer_latest.pt"
    plot_path: str = "weights/loss_curve.png"

    num_embeddings: int = 3
    compute_eval_loss: bool = True
    compute_eval_accuracy: bool = True
    eval_loss_batches: int = 20
    eval_max_embeddings: Optional[int] = 2000
    mini_eval_max_frame_embeddings: Optional[int] = 1
    eval_batch_size: int = 128

    wandb_project: str = "open-vocab-mot"
    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    no_wandb: bool = False
    skip_eval: bool = False
    run_initial_mini_eval: bool = False

    augmentation_config: AugmentationConfig = Field(default_factory=AugmentationConfig)

    use_nested_tensors: bool = True
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
    veri: VeRiDatasetConfig = Field(default_factory=VeRiDatasetConfig)
    vrai: VRAIDatasetConfig = Field(default_factory=VRAIDatasetConfig)


class CombinedIterableDataset(IterableDataset):
    def __init__(
        self,
        datasets: list[IterableDataset],
        dataset_names: list[str],
        weights: list[float],
        batches_per_epoch: int,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.weights = weights
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed

    def __iter__(self):
        worker_info = get_worker_info()
        seed = self.seed
        if worker_info is not None:
            seed += worker_info.id
            
        rng = random.Random(seed)
        
        iters = [iter(ds) for ds in self.datasets]
        
        if self.batches_per_epoch is not None:
            if worker_info is not None:
                per_worker = self.batches_per_epoch // worker_info.num_workers
                worker_id = worker_info.id
                if worker_id < self.batches_per_epoch % worker_info.num_workers:
                    per_worker += 1
                batches_to_yield = per_worker
            else:
                batches_to_yield = self.batches_per_epoch
        else:
            batches_to_yield = float('inf')

        batch_count = 0
        while batch_count < batches_to_yield:
            chosen_idx = rng.choices(range(len(iters)), weights=self.weights, k=1)[0]
            try:
                batch_items = next(iters[chosen_idx])
            except StopIteration:
                iters[chosen_idx] = iter(self.datasets[chosen_idx])
                batch_items = next(iters[chosen_idx])
            
            yield batch_items, self.dataset_names[chosen_idx]
            batch_count += 1


def combined_collate_fn(data):
    # data is a single yielded element from the combined iterable dataset
    batch_items, dataset_name = data
    return collate_video_reid_ds(batch_items), dataset_name


def _run_evaluation_tasks(
    name, cfg, tasks, ds_dict, pipeline_model, accelerator, device, args, epoch, is_mini, eval_prefix, max_frame_embeddings
):
    if len(tasks) == 0:
        return
        
    # Figure out which sources and modalities need extracting
    sources_to_extract = {"val": {"video": False, "frame": False}, "query": {"video": False, "frame": False}, "gallery": {"video": False, "frame": False}}
    
    for task in tasks:
        if task.query_source in sources_to_extract:
            sources_to_extract[task.query_source][task.query_modality] = True
        if task.gallery_source in sources_to_extract:
            sources_to_extract[task.gallery_source][task.gallery_modality] = True

    datasets_for_target_ids = [ds for ds_name, ds in ds_dict.items() if ds_name in ["val", "query", "gallery"]]
    target_ids = get_eval_identities_by_max_embeddings(
        datasets=datasets_for_target_ids,
        num_video_embeddings_per_sequence=args.num_embeddings,
        max_embeddings=args.eval_max_embeddings,
        seed=args.seed
    )

    extractions = {}
    for source in ["val", "query", "gallery"]:
        if source in ds_dict:
            need_video = sources_to_extract[source]["video"]
            need_frame = sources_to_extract[source]["frame"]
            if need_video or need_frame:
                if accelerator.is_main_process:
                    print(f"\nProcessing {name} {source.capitalize()} Dataset...")
                
                source_target_ids = target_ids if is_mini else None
                
                extractions[source] = extract_embeddings(
                    dataset=ds_dict[source],
                    model=pipeline_model,
                    accelerator=accelerator,
                    frames_per_video_embedding=cfg.frames_per_video,
                    num_video_embeddings_per_sequence=args.num_embeddings,
                    return_video_embeddings=need_video,
                    return_frame_embeddings=need_frame,
                    max_frame_embeddings_per_sequence=max_frame_embeddings,
                    target_identities=source_target_ids,
                    seed=args.seed,
                    batch_size=args.eval_batch_size
                )
                
    if accelerator.is_main_process:
        print(f"\nRunning metrics computation for {name}...")
        
    for task in tasks:
        if task.query_source not in extractions or task.gallery_source not in extractions:
            continue
            
        q_ext = extractions[task.query_source]
        g_ext = extractions[task.gallery_source]
        
        q_map = q_ext.video_embeddings if task.query_modality == "video" else q_ext.frame_embeddings
        g_map = g_ext.video_embeddings if task.gallery_modality == "video" else g_ext.frame_embeddings
        
        if q_map is None or g_map is None:
            continue
            
        same_source = (task.query_source == task.gallery_source)
        mode = f"{task.query_modality}_to_{task.gallery_modality}"
        
        if accelerator.is_main_process:
            metrics = evaluate_reid(
                query_map=q_map,
                key_map=g_map,
                same_source=same_source,
                sim_aggregation="max",
                device=device,
                mode=mode
            )
            
            if not args.no_wandb and metrics:
                prefix = f"eval/{name}/{eval_prefix}{mode}"
                logs = {
                    f"{prefix}_rank_1": metrics["video"]["rank_1"],
                    f"{prefix}_mAP": metrics["video"]["mAP"],
                }
                if not is_mini:
                    logs.update({
                        f"{prefix}_rank_5": metrics["video"]["rank_5"],
                        f"{prefix}_rank_10": metrics["video"]["rank_10"],
                        f"{prefix}_identity_rank_1": metrics["identity"]["rank_1"],
                        f"{prefix}_identity_rank_5": metrics["identity"]["rank_5"],
                        f"{prefix}_identity_rank_10": metrics["identity"]["rank_10"],
                    })
                if is_mini:
                    logs["epoch"] = epoch + 1
                    
                wandb.log(logs)

def run_mini_evaluation(epoch, args, model, dino_harness, criterion, device, accelerator, eval_datasets_cache):
    if not args.compute_eval_loss and not args.compute_eval_accuracy:
        return
        
    if accelerator.is_main_process:
        print(f"\n--- Running Mini-Evaluation for Epoch {epoch+1} ---")
        
    model.eval()
    pipeline_model = HierarchicalReIDPipeline(
        dino_harness=dino_harness,
        reid_model=accelerator.unwrap_model(model),
        dino_batch_split=args.dino_batch_split
    )
    
    def get_ds(name, dtype, cfg):
        if name not in eval_datasets_cache:
            if accelerator.is_main_process:
                print(f"Loading {name} test sets for evaluation...")
            eval_datasets_cache[name] = load_dataset_for_eval(
                dataset_type=dtype,
                config=cfg,
                verbose=accelerator.is_main_process
            )
        return eval_datasets_cache[name]
        
    eval_configs = []
    eval_configs.append(("duke", args.duke, "duke"))
    eval_configs.append(("whale", args.whale, "whale"))
    for subset_cfg in args.wildlife_subsets:
        eval_configs.append((subset_cfg.subset_dataset, subset_cfg, "wildlife10k_subset"))
    eval_configs.append(("veri", args.veri, "veri"))
    eval_configs.append(("vrai", args.vrai, "vrai"))
    
    for name, cfg, dtype in eval_configs:
        tasks = cfg.mini_eval_tasks
        if len(tasks) == 0:
            continue
            
        ds_dict = get_ds(name, dtype, cfg)
        
        if args.compute_eval_loss:
            if accelerator.is_main_process:
                print(f"Computing mini validation loss for {name}...")
            # Just grab the first available dataset for computing loss (usually val or gallery)
            target_ds = ds_dict.get("val", ds_dict.get("gallery", list(ds_dict.values())[0]))
            
            iterable_ds = VideoReIDKPFBatchIterableDataset(
                target_ds,
                batches_per_epoch=args.eval_loss_batches,
                num_identities_per_batch=cfg.people_per_batch,
                num_sequences_per_identity=cfg.views_per_person,
                num_frames_per_sequence=cfg.frames_per_video,
                allow_same_identity_same_sequence=True,
                allow_reduced_sequences_per_identity=False,
                allow_resampling_sample_indices=True,
                epoch_deterministic=False,
                seed=args.seed + accelerator.process_index * 100,
                verbose=False
            )
            loader = DataLoader(iterable_ds, batch_size=None, collate_fn=collate_video_reid_ds, num_workers=2)
            
            total_loss_sum = 0.0
            total_cross_loss_sum = 0.0
            total_video_loss_sum = 0.0
            total_frame_loss_sum = 0.0
            count = 0
            for batch in loader:
                with torch.no_grad():
                    v_loss, f_loss, c_loss, t_loss = compute_batch_loss(
                        batch, dino_harness, model, criterion, device, 
                        args.frame_loss_weight, args.cross_loss_weight, args.dino_batch_split
                    )
                if t_loss is not None:
                    t_loss_gathered = accelerator.gather(t_loss.unsqueeze(0))
                    c_loss_gathered = accelerator.gather(c_loss.unsqueeze(0))
                    v_loss_gathered = accelerator.gather(v_loss.unsqueeze(0))
                    f_loss_gathered = accelerator.gather(f_loss.unsqueeze(0))
                    total_loss_sum += t_loss_gathered.mean().item()
                    total_cross_loss_sum += c_loss_gathered.mean().item()
                    total_video_loss_sum += v_loss_gathered.mean().item()
                    total_frame_loss_sum += f_loss_gathered.mean().item()
                    count += 1
                    
            if count > 0 and accelerator.is_main_process and not args.no_wandb:
                wandb.log({
                    f"eval/{name}/loss": total_loss_sum / count,
                    f"eval/{name}/cross_loss": total_cross_loss_sum / count,
                    f"eval/{name}/video_loss": total_video_loss_sum / count,
                    f"eval/{name}/frame_loss": total_frame_loss_sum / count,
                    "epoch": epoch + 1
                })
                
        if args.compute_eval_accuracy:
            if accelerator.is_main_process:
                print(f"Computing mini validation accuracy for {name}...")
                
            _run_evaluation_tasks(
                name=name, cfg=cfg, tasks=tasks, ds_dict=ds_dict,
                pipeline_model=pipeline_model, accelerator=accelerator, device=device,
                args=args, epoch=epoch, is_mini=True, eval_prefix="mini_", 
                max_frame_embeddings=args.mini_eval_max_frame_embeddings
            )


@hydra.main(version_base=None, config_path="../configs/train_and_evaluate_multiple", config_name="config")
def main(cfg: DictConfig):
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
    
    set_seed(args.seed + accelerator.process_index)
        
    # Ensure outputs directory exists
    if accelerator.is_main_process:
        checkpoint_file = Path(args.checkpoint_path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        plot_file = Path(args.plot_path)
        plot_file.parent.mkdir(parents=True, exist_ok=True)

    transform = None
    # if args.enable_augmentations:
    if args.augmentation_config.enable:
        if accelerator.is_main_process:
            print(f"Enabling training augmentations with config {args.augmentation_config}")
        
        crop_ratio_range = args.augmentation_config.crop_ratio_range
        random_flip_prop = args.augmentation_config.random_flip_prop
        color_jitter_config = args.augmentation_config.color_jitter_config
        transform = v2.Compose([
            RandomRatioCrop(ratio_range=crop_ratio_range),
            v2.RandomHorizontalFlip(p=random_flip_prop),
            v2.ColorJitter(brightness=color_jitter_config.brightness,
                           contrast=color_jitter_config.contrast,
                           saturation=color_jitter_config.saturation,
                           hue=color_jitter_config.hue)
        ])

    duke_iterable = None
    if args.duke.use_for_training:
        if accelerator.is_main_process:
            print("Loading DukeMTMC dataset for training...")
        duke_iterable = load_dataset_for_training(
            dataset_type="duke",
            config=args.duke,
            seed=args.seed + accelerator.process_index * 100,
            verbose=accelerator.is_main_process,
            transform=transform
        )

    whale_iterable = None
    if args.whale.use_for_training:
        if accelerator.is_main_process:
            print("Loading Whale dataset for training...")
        whale_iterable = load_dataset_for_training(
            dataset_type="whale",
            config=args.whale,
            seed=args.seed + 1 + accelerator.process_index * 100,
            verbose=accelerator.is_main_process,
            transform=transform
        )

    wildlife_iterables = {}
    for subset_cfg in args.wildlife_subsets:
        if subset_cfg.use_for_training:
            subset_name = subset_cfg.subset_dataset
            if accelerator.is_main_process:
                print(f"Loading Wildlife10K subset {subset_name} for training...")
            subset_iterable = load_dataset_for_training(
                dataset_type="wildlife10k_subset",
                config=subset_cfg,
                seed=args.seed + 2 + len(wildlife_iterables) + accelerator.process_index * 100,
                verbose=accelerator.is_main_process,
                transform=transform
            )
            wildlife_iterables[subset_name] = subset_iterable
            
    veri_iterable = None
    if args.veri.use_for_training:
        if accelerator.is_main_process:
            print("Loading VeRi dataset for training...")
        veri_iterable = load_dataset_for_training(
            dataset_type="veri",
            config=args.veri,
            seed=args.seed + 3 + len(wildlife_iterables) + accelerator.process_index * 100,
            verbose=accelerator.is_main_process,
            transform=transform
        )
        
    vrai_iterable = None
    if args.vrai.use_for_training:
        if accelerator.is_main_process:
            print("Loading VRAI dataset for training...")
        vrai_iterable = load_dataset_for_training(
            dataset_type="vrai",
            config=args.vrai,
            seed=args.seed + 4 + len(wildlife_iterables) + accelerator.process_index * 100,
            verbose=accelerator.is_main_process,
            transform=transform
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
        
    if args.use_nested_tensors:
        print(f"Using a nested NestedHierarchicalVideoReIDTransformer")
        model_class = NestedHierarchicalVideoReIDTransformer
    else:
        print(f"Using a HierarchicalVideoReIDTransformer")
        model_class = HierarchicalVideoReIDTransformer
    model = model_class(
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
    
    # Combine datasets
    datasets = []
    dataset_names = []
    weights = []

    if duke_iterable is not None:
        datasets.append(duke_iterable)
        dataset_names.append("duke")
        weights.append(args.duke.weight)
        
    if whale_iterable is not None:
        datasets.append(whale_iterable)
        dataset_names.append("whale")
        weights.append(args.whale.weight)
        
    for subset_cfg in args.wildlife_subsets:
        subset_name = subset_cfg.subset_dataset
        if subset_name in wildlife_iterables:
            datasets.append(wildlife_iterables[subset_name])
            dataset_names.append(subset_name)
            weights.append(subset_cfg.weight)
            
    if veri_iterable is not None:
        datasets.append(veri_iterable)
        dataset_names.append("veri")
        weights.append(args.veri.weight)
        
    if vrai_iterable is not None:
        datasets.append(vrai_iterable)
        dataset_names.append("vrai")
        weights.append(args.vrai.weight)
            
    if not datasets:
        raise ValueError("No datasets enabled for training. Set at least one dataset's 'use' flag to True.")

    combined_dataset = CombinedIterableDataset(
        datasets=datasets,
        dataset_names=dataset_names,
        weights=weights,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed + accelerator.process_index * 100,
    )

    combined_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=None,
        collate_fn=combined_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    
    history = {
        "total_loss": []
    }
    for name in dataset_names:
        history[f"{name}_video_loss"] = []
        history[f"{name}_frame_loss"] = []
        history[f"{name}_cross_loss"] = []
        history[f"{name}_total_loss"] = []
    
    eval_datasets_cache = {}

    if accelerator.is_main_process:
        print("Starting training...")
        
    model.train()
    
    if args.run_initial_mini_eval:
        if accelerator.is_main_process:
            print("\n--- Running Initial Mini-Evaluation (Baseline) ---")
        run_mini_evaluation(-1, args, model, dino_harness, criterion, device, accelerator, eval_datasets_cache)
        model.train()
        
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
            
        progress = tqdm(total=args.batches_per_epoch, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        profiler = StepProfiler(device)
        
        batch_start_time = time.perf_counter()
        for batch_idx, (batch, selected_ds_name) in enumerate(combined_loader):
            data_load_time = time.perf_counter() - batch_start_time
            if batch_idx >= args.batches_per_epoch:
                break
                
            # 1-5. Compute Losses via helper
            video_loss, frame_loss, cross_loss, total_loss = compute_batch_loss(
                batch, dino_harness, model, criterion, device, 
                args.frame_loss_weight, args.cross_loss_weight, args.dino_batch_split,
                profiler=profiler
            )
            
            if total_loss is None:
                batch_start_time = time.perf_counter()
                continue
            
            # 6. Backward Pass and Optimize
            profiler.start("backward")
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()
            profiler.stop("backward")
            
            t_total_step = time.perf_counter() - batch_start_time
            steps_per_sec = 1.0 / t_total_step if t_total_step > 0 else 0
            
            # Get timing metrics
            t_dino = profiler.elapsed("dino")
            t_forward = profiler.elapsed("forward")
            t_backward = profiler.elapsed("backward")
            
            # Track history and log (only on main process)
            if accelerator.is_main_process:
                history[f"{selected_ds_name}_video_loss"].append(video_loss.item())
                history[f"{selected_ds_name}_frame_loss"].append(frame_loss.item())
                history[f"{selected_ds_name}_cross_loss"].append(cross_loss.item())
                history[f"{selected_ds_name}_total_loss"].append(total_loss.item())
                history["total_loss"].append(total_loss.item())
                
                if not args.no_wandb:
                    wandb.log({
                        "train/loss": total_loss.item(),
                        f"train/{selected_ds_name}/video_loss": video_loss.item(),
                        f"train/{selected_ds_name}/frame_loss": frame_loss.item(),
                        f"train/{selected_ds_name}/cross_loss": cross_loss.item(),
                        f"train/{selected_ds_name}/loss": total_loss.item(),
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "profiling/data_load_time": data_load_time,
                        "profiling/dino_time": t_dino,
                        "profiling/forward_time": t_forward,
                        "profiling/backward_time": t_backward,
                        "profiling/total_step_time": t_total_step,
                        "profiling/steps_per_sec": steps_per_sec,
                    })
                
                progress.set_postfix({
                    "Loss": f"{total_loss.item():.4f}", 
                    "DS": selected_ds_name,
                    "s/it": f"{t_total_step:.2f}"
                })
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx} [{selected_ds_name}]: Total Loss = {total_loss.item():.4f} "
                          f"(Video: {video_loss.item():.4f}, Frame: {frame_loss.item():.4f}, Cross: {cross_loss.item():.4f}) | "
                          f"Data: {data_load_time:.3f}s, DINO: {t_dino:.3f}s, Fwd: {t_forward:.3f}s, Bwd: {t_backward:.3f}s, Total: {t_total_step:.3f}s ({steps_per_sec:.2f} it/s)")
                
                # Update progress bar
                progress.update(1)
            
            # Explicitly free memory at the end of the batch
            del batch
            batch_start_time = time.perf_counter()

        run_mini_evaluation(epoch, args, model, dino_harness, criterion, device, accelerator, eval_datasets_cache)
        model.train()

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
    
    pipeline_model = HierarchicalReIDPipeline(
        dino_harness=dino_harness,
        reid_model=accelerator.unwrap_model(model),
        dino_batch_split=args.dino_batch_split
    )
    
    eval_configs = []
    eval_configs.append(("duke", args.duke, "duke"))
    eval_configs.append(("whale", args.whale, "whale"))
    for subset_cfg in args.wildlife_subsets:
        eval_configs.append((subset_cfg.subset_dataset, subset_cfg, "wildlife10k_subset"))
    eval_configs.append(("veri", args.veri, "veri"))
    eval_configs.append(("vrai", args.vrai, "vrai"))

    for name, cfg, dtype in eval_configs:
        tasks = cfg.eval_tasks
        if len(tasks) == 0:
            continue
            
        if name not in eval_datasets_cache:
            if accelerator.is_main_process:
                print(f"\nLoading {name} test sets for evaluation...")
            eval_datasets_cache[name] = load_dataset_for_eval(
                dataset_type=dtype,
                config=cfg,
                verbose=accelerator.is_main_process
            )
        else:
            if accelerator.is_main_process:
                print(f"\nUsing cached {name} test sets for evaluation...")
        ds_dict = eval_datasets_cache[name]
        
        _run_evaluation_tasks(
            name=name, cfg=cfg, tasks=tasks, ds_dict=ds_dict,
            pipeline_model=pipeline_model, accelerator=accelerator, device=device,
            args=args, epoch=-1, is_mini=False, eval_prefix="", 
            max_frame_embeddings=None
        )
        
        accelerator.wait_for_everyone()

    if accelerator.is_main_process and not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

