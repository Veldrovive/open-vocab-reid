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

# Ensure train_and_evaluate_multiple_parallel TrainConfig is importable
# We add the scripts directory to sys.path so we can import the config schema
sys.path.append(str(Path(__file__).parent))
from train_and_evaluate_multiple_parallel import TrainConfig

try:
    import fiftyone as fo
    import fiftyone.brain as fob
except ImportError:
    print("Error: fiftyone or fiftyone-brain is not installed.")
    print("Please run: uv add fiftyone fiftyone-brain umap-learn")
    sys.exit(1)

def extract_and_visualize(args: TrainConfig, model, dino_harness, accelerator):
    device = accelerator.device
    model.eval()
    
    pipeline_model = HierarchicalReIDPipeline(
        dino_harness=dino_harness,
        reid_model=accelerator.unwrap_model(model),
        dino_batch_split=args.dino_batch_split
    )
    
    eval_configs = []
    if args.duke.use_for_training or args.duke.eval_tasks:
        eval_configs.append(("duke", args.duke, "duke"))
    if args.whale.use_for_training or args.whale.eval_tasks:
        eval_configs.append(("whale", args.whale, "whale"))
    for subset_cfg in args.wildlife_subsets:
        if subset_cfg.use_for_training or subset_cfg.eval_tasks:
            eval_configs.append((subset_cfg.subset_dataset, subset_cfg, "wildlife10k_subset"))
    if args.veri.use_for_training or args.veri.eval_tasks:
        eval_configs.append(("veri", args.veri, "veri"))
    if args.vrai.use_for_training or args.vrai.eval_tasks:
        eval_configs.append(("vrai", args.vrai, "vrai"))
        
    dataset_name = "ReID_Embeddings"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    fo_dataset = fo.Dataset(dataset_name)

    all_embeddings = []
    samples = []
    
    frames_per_video_embedding = args.duke.frames_per_video if hasattr(args.duke, 'frames_per_video') else 8

    with torch.no_grad():
        for name, cfg, dtype in eval_configs:
            if accelerator.is_main_process:
                print(f"\nProcessing {name} dataset for visualization...")
            
            ds_dict = load_dataset_for_eval(
                dataset_type=dtype,
                config=cfg,
                verbose=accelerator.is_main_process
            )
            
            # Use query or val split
            target_ds = ds_dict.get("query", ds_dict.get("val", list(ds_dict.values())[0]))
            
            unique_identities = target_ds.unique_identities
            seq_map = target_ds.sequence_map
            
            # Subsample identities to keep visualization fast and uncluttered
            max_ids = 50
            if len(unique_identities) > max_ids:
                identities_to_process = random.sample(unique_identities, max_ids)
            else:
                identities_to_process = unique_identities
                
            for identity_id in tqdm(identities_to_process, desc=f"Extracting {name}"):
                for sequence_id, available_indices in seq_map[identity_id].items():
                    if len(available_indices) == 0:
                        continue
                    
                    # We'll visualize one video embedding and all its frames
                    pool = list(available_indices)
                    if len(pool) > frames_per_video_embedding:
                        selected_indices = random.sample(pool, frames_per_video_embedding)
                    else:
                        selected_indices = pool
                        
                    items = [target_ds[i] for i in selected_indices]
                    frames = [item.frame_tensor for item in items]
                    segmentations = [item.segmentation_tensor for item in items]
                    frame_paths = [str(item.frame_path) for item in items]
                    
                    output = pipeline_model(
                        frames=[frames],
                        segmentations=[segmentations],
                        extract_video=True,
                        extract_frames=True
                    )
                    
                    # Video Embedding Sample
                    vid_emb = output["video_contrastive_embeddings"]
                    if vid_emb is not None and vid_emb.shape[0] > 0:
                        # Use first frame as representative for video
                        sample = fo.Sample(filepath=frame_paths[0])
                        sample["dataset_name"] = name
                        sample["identity_id"] = str(identity_id)
                        sample["sequence_id"] = str(sequence_id)
                        sample["embedding_type"] = "video"
                        samples.append(sample)
                        all_embeddings.append(vid_emb[0].cpu().numpy())
                        
                    # Frame Embedding Samples
                    frame_embs = output["frame_contrastive_embeddings"]
                    if frame_embs is not None and len(frame_embs) > 0 and len(frame_embs[0]) > 0:
                        f_embs = frame_embs[0]
                        for idx, f_emb in enumerate(f_embs):
                            if idx < len(frame_paths):
                                sample = fo.Sample(filepath=frame_paths[idx])
                                sample["dataset_name"] = name
                                sample["identity_id"] = str(identity_id)
                                sample["sequence_id"] = str(sequence_id)
                                sample["embedding_type"] = "frame"
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


@hydra.main(version_base=None, config_path="../configs/train_and_evaluate_multiple", config_name="config")
def main(cfg: DictConfig):
    try:
        args = TrainConfig(**OmegaConf.to_container(cfg, resolve=True))
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
