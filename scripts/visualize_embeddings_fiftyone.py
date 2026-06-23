#!/usr/bin/env python3

"""
Visualize ReID embeddings using FiftyOne and UMAP.
Example usage:
export CUDA_VISIBLE_DEVICES=4
uv run scripts/visualize_embeddings_fiftyone.py --config-name=config_no_unsup checkpoint_path="weights/reid_transformer_latest.pt" eval_max_embeddings=500
"""

import os
import sys
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Import project utilities and dataset classes
from open_vocab_mot.data import load_dataset_for_eval
from open_vocab_mot.data.video_reid_abc import VideoReIDItem
from aidan_lib.models.dino_lib_compiled import DINOv3CompiledHarness
from aidan_lib.utils import set_seed

from open_vocab_mot.models import HierarchicalVideoReIDTransformer, NestedHierarchicalVideoReIDTransformer, HierarchicalReIDPipeline

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class VideoGroupingMode(str, Enum):
    FRAME_ONLY = "frame_only"         # Every frame is its own video
    SPLIT_SEQUENCE = "split_sequence" # Split a sequence into multiple chunks of N frames
    EXISTING = "existing"             # Use existing sequence ID

class VisDatasetConfig(BaseModel):
    enable: bool = False
    subset_dataset: Optional[str] = None # for wildlife/unsup
    input_set: str = "val" # "query", "gallery", "val", "train"
    grouping_mode: VideoGroupingMode = VideoGroupingMode.EXISTING
    frames_per_video: int = 8
    max_identities: int = 50

class VisualizeConfig(BaseModel):
    checkpoint_path: str = "weights/reid_transformer_latest.pt"
    dino_checkpoint: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    dino_batch_split: int = 1
    seed: int = 42
    
    cuda_visible_devices: Optional[str] = None
    
    recompute: bool = False
    
    extract_video_embeddings: bool = True
    extract_frame_embeddings: bool = True
    include_masks: bool = True
    
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

    duke: VisDatasetConfig = Field(default_factory=VisDatasetConfig)
    whale: VisDatasetConfig = Field(default_factory=VisDatasetConfig)
    wildlife_subsets: list[VisDatasetConfig] = Field(default_factory=list)
    veri: VisDatasetConfig = Field(default_factory=VisDatasetConfig)
    vrai: VisDatasetConfig = Field(default_factory=VisDatasetConfig)
    hypersim: VisDatasetConfig = Field(default_factory=VisDatasetConfig)

try:
    import fiftyone as fo
    import fiftyone.brain as fob
except ImportError:
    print("Error: fiftyone or fiftyone-brain is not installed.")
    print("Please run: uv add fiftyone fiftyone-brain umap-learn")
    sys.exit(1)

def extract_and_visualize(args: VisualizeConfig, model, dino_harness, accelerator):
    device = accelerator.device
    model.eval()
    
    pipeline_model = HierarchicalReIDPipeline(
        dino_harness=dino_harness,
        reid_model=accelerator.unwrap_model(model),
        dino_batch_split=args.dino_batch_split
    )
    
    eval_configs = []
    if args.duke.enable:
        eval_configs.append(("duke", args.duke, "duke"))
    if args.whale.enable:
        eval_configs.append(("whale", args.whale, "whale"))
    for subset_cfg in args.wildlife_subsets:
        if subset_cfg.enable:
            eval_configs.append((subset_cfg.subset_dataset, subset_cfg, "wildlife10k_subset"))
    if args.veri.enable:
        eval_configs.append(("veri", args.veri, "veri"))
    if args.vrai.enable:
        eval_configs.append(("vrai", args.vrai, "vrai"))
    if args.hypersim.enable:
        eval_configs.append(("hypersim", args.hypersim, "hypersim"))
        
    dataset_name = "ReID_Embeddings"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    fo_dataset = fo.Dataset(dataset_name)

    all_embeddings = []
    samples = []
    
    with torch.no_grad():
        for name, cfg, dtype in eval_configs:
            if accelerator.is_main_process:
                print(f"\nProcessing {name} dataset for visualization...")
            
            from open_vocab_mot.data.dataset_factory import DatasetConfig, DukeDatasetConfig, WhaleDatasetConfig, Wildlife10kSubsetDatasetConfig, VeRiDatasetConfig, VRAIDatasetConfig, HypersimDatasetConfig
            
            # Map dtype to config class
            config_mapping = {
                "duke": DukeDatasetConfig,
                "whale": WhaleDatasetConfig,
                "wildlife10k_subset": Wildlife10kSubsetDatasetConfig,
                "veri": VeRiDatasetConfig,
                "vrai": VRAIDatasetConfig,
                "hypersim": HypersimDatasetConfig,
            }
            
            ds_cfg_class = config_mapping.get(dtype, DatasetConfig)
            ds_kwargs = {}
            if hasattr(cfg, "subset_dataset") and cfg.subset_dataset is not None:
                ds_kwargs["subset_dataset"] = cfg.subset_dataset
                
            dummy_cfg = ds_cfg_class(**ds_kwargs)

            ds_dict = load_dataset_for_eval(
                dataset_type=dtype,
                config=dummy_cfg,
                verbose=accelerator.is_main_process
            )
            
            # Select target split based on cfg.input_set
            target_ds = None
            if cfg.input_set in ds_dict:
                target_ds = ds_dict[cfg.input_set]
            else:
                # Fallback to whatever is available
                target_ds = ds_dict.get("query", ds_dict.get("val", list(ds_dict.values())[0]))
                print(f"Warning: requested input_set '{cfg.input_set}' not found for {name}. Using fallback split.")
            
            unique_identities = target_ds.unique_identities
            seq_map = target_ds.sequence_map
            
            # Subsample identities to keep visualization fast and uncluttered
            max_ids = cfg.max_identities
            if len(unique_identities) > max_ids:
                identities_to_process = random.sample(unique_identities, max_ids)
            else:
                identities_to_process = unique_identities
                
            for identity_id in tqdm(identities_to_process, desc=f"Extracting {name}"):
                for original_sequence_id, available_indices in seq_map[identity_id].items():
                    if len(available_indices) == 0:
                        continue
                        
                    pool = list(available_indices)
                    
                    # Prepare sequence chunks based on grouping_mode
                    sequence_chunks = []
                    
                    if cfg.grouping_mode == VideoGroupingMode.FRAME_ONLY:
                        for idx, frame_idx in enumerate(pool):
                            sequence_chunks.append({
                                "sequence_id": f"{original_sequence_id}_f{idx}",
                                "indices": [frame_idx]
                            })
                    elif cfg.grouping_mode == VideoGroupingMode.SPLIT_SEQUENCE:
                        chunk_size = cfg.frames_per_video
                        for i in range(0, len(pool), chunk_size):
                            chunk = pool[i:i+chunk_size]
                            sequence_chunks.append({
                                "sequence_id": f"{original_sequence_id}_c{i//chunk_size}",
                                "indices": chunk
                            })
                    else: # EXISTING
                        if len(pool) > cfg.frames_per_video:
                            chunk = random.sample(pool, cfg.frames_per_video)
                        else:
                            chunk = pool
                        sequence_chunks.append({
                            "sequence_id": str(original_sequence_id),
                            "indices": chunk
                        })
                    
                    for chunk_info in sequence_chunks:
                        selected_indices = chunk_info["indices"]
                        sequence_id = chunk_info["sequence_id"]
                        
                        items = [target_ds[i] for i in selected_indices]
                        frames = [item.frame_tensor for item in items]
                        segmentations = [item.segmentation_tensor for item in items]
                        frame_paths = [str(item.frame_path) for item in items]
                        
                        # Save the exact tensors (which may be dynamically cropped/masked) to a cache 
                        # so FiftyOne displays exactly what the model sees.
                        import torchvision
                        cache_dir = Path(f".fiftyone_cache/{name}")
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        
                        if len(frames) == 0:
                            continue
                            
                        emb_cache_path = cache_dir / f"{identity_id}_{sequence_id}_embeddings.pt"
                        expected_img_paths = [cache_dir / f"{identity_id}_{sequence_id}_{idx}.jpg" for idx in range(len(frames))]
                        
                        is_cached = emb_cache_path.exists() and all(p.exists() for p in expected_img_paths)
                        
                        if is_cached and not args.recompute:
                            output = torch.load(emb_cache_path, map_location="cpu", weights_only=False)
                            for idx, p in enumerate(expected_img_paths):
                                frame_paths[idx] = str(p.absolute())
                        else:
                            for idx, tensor in enumerate(frames):
                                if tensor is not None:
                                    out_path = expected_img_paths[idx]
                                    torchvision.utils.save_image(tensor, out_path)
                                    frame_paths[idx] = str(out_path.absolute())
    
                            output = pipeline_model(
                                frames=[frames],
                                segmentations=[segmentations],
                                extract_video=args.extract_video_embeddings,
                                extract_frames=args.extract_frame_embeddings
                            )
                            
                            output_to_save = {}
                            if output.get("video_contrastive_embeddings") is not None:
                                output_to_save["video_contrastive_embeddings"] = output["video_contrastive_embeddings"].cpu()
                            if output.get("frame_contrastive_embeddings") is not None:
                                out_frame = []
                                for item in output["frame_contrastive_embeddings"]:
                                    if isinstance(item, torch.Tensor):
                                        out_frame.append(item.cpu())
                                    elif isinstance(item, list):
                                        out_frame.append([x.cpu() if isinstance(x, torch.Tensor) else x for x in item])
                                    else:
                                        out_frame.append(item)
                                output_to_save["frame_contrastive_embeddings"] = out_frame
                            torch.save(output_to_save, emb_cache_path)
                        
                        # Video Embedding Sample
                        if args.extract_video_embeddings:
                            vid_emb = output.get("video_contrastive_embeddings")
                            if vid_emb is not None and vid_emb.shape[0] > 0:
                                # Use first frame as representative for video
                                sample = fo.Sample(filepath=frame_paths[0])
                                sample["dataset_name"] = name
                                sample["identity_id"] = str(identity_id)
                                sample["sequence_id"] = str(sequence_id)
                                sample["embedding_type"] = "video"
                                
                                if args.include_masks and segmentations[0] is not None:
                                    mask_np = segmentations[0].squeeze().cpu().numpy().astype(bool)
                                    sample["segmentation"] = fo.Segmentation(mask=mask_np)
                                    
                                samples.append(sample)
                                all_embeddings.append(vid_emb[0].cpu().numpy())
                                
                        # Frame Embedding Samples
                        if args.extract_frame_embeddings:
                            frame_embs = output.get("frame_contrastive_embeddings")
                            if frame_embs is not None and len(frame_embs) > 0 and len(frame_embs[0]) > 0:
                                f_embs = frame_embs[0]
                                for idx, f_emb in enumerate(f_embs):
                                    if idx < len(frame_paths):
                                        sample = fo.Sample(filepath=frame_paths[idx])
                                        sample["dataset_name"] = name
                                        sample["identity_id"] = str(identity_id)
                                        sample["sequence_id"] = str(sequence_id)
                                        sample["embedding_type"] = "frame"
                                        
                                        if args.include_masks and segmentations[idx] is not None:
                                            mask_np = segmentations[idx].squeeze().cpu().numpy().astype(bool)
                                            sample["segmentation"] = fo.Segmentation(mask=mask_np)
                                            
                                        samples.append(sample)
                                        all_embeddings.append(f_emb.cpu().numpy())
                                
    if len(samples) > 0:
        if accelerator.is_main_process:
            print(f"Adding {len(samples)} samples to FiftyOne dataset...")
        fo_dataset.add_samples(samples)
        
        if accelerator.is_main_process:
            print("Computing UMAP visualization...")
            fob.compute_visualization(
                fo_dataset,
                embeddings=all_embeddings,
                brain_key="reid_embeddings_umap",
                method="umap",
                num_dims=3
            )
            print("Launching FiftyOne App...")
            session = fo.launch_app(fo_dataset)
            session.wait()
    else:
        if accelerator.is_main_process:
            print("No embeddings extracted.")


@hydra.main(version_base=None, config_path="../configs/visualize", config_name="config")
def main(cfg: DictConfig):
    try:
        args = VisualizeConfig(**OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        print("Configuration validation error:", e)
        sys.exit(1)

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
            
    if accelerator.is_main_process:
        print(f"Using device: {device}")
    
    set_seed(args.seed + accelerator.process_index)
    
    dino_harness = DINOv3CompiledHarness(
        checkpoint=args.dino_checkpoint,
        device=device,
        dtype=torch.bfloat16,
        max_side_len=1024,
        warmup=False
    )
    
    if args.use_nested_tensors:
        model_class = NestedHierarchicalVideoReIDTransformer
    else:
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
    ).to(device)
    
    if accelerator.is_main_process:
        print(f"Loading checkpoint from {args.checkpoint_path}...")
    
    if os.path.exists(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        if accelerator.is_main_process:
            print("Checkpoint loaded.")
    else:
        if accelerator.is_main_process:
            print(f"Warning: Checkpoint {args.checkpoint_path} not found. Using untrained weights.")
            
    model = accelerator.prepare(model)
    
    extract_and_visualize(args, model, dino_harness, accelerator)

if __name__ == "__main__":
    main()
