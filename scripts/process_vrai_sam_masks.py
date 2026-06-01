import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from open_vocab_mot.data.vrai_ds import VRAIDataset, VRAISplit, VRAITestSet
from open_vocab_mot.data import custom_collate, process_sam_masks_for_dataset
from aidan_lib.models.sam3_batched_img import SAM3BatchedImageHarness


def main():
    parser = argparse.ArgumentParser(description="Process VRAI dataset and save SAM3 masks")
    parser.add_argument("--ds-root", type=str, required=True, help="Path to VRAI dataset root directory")
    parser.add_argument("--sidecar-root", type=str, required=True, help="Path to save the generated sidecar masks")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SAM3 on (e.g. 'cuda', 'cuda:0')")
    parser.add_argument("--split", type=str, default="all", choices=["train", "query", "gallery", "all"], help="Dataset split to process")
    parser.add_argument("--test-set", type=str, default="full", choices=["dev", "full"], help="Test set to process if split is query or gallery")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--sam-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="SAM3 dtype precision")
    args = parser.parse_args()

    ds_root = Path(args.ds_root)
    sidecar_root = Path(args.sidecar_root)

    split_map = {
        "train": VRAISplit.TRAIN,
        "query": VRAISplit.QUERY,
        "gallery": VRAISplit.GALLERY,
    }
    
    test_set_map = {
        "dev": VRAITestSet.DEV,
        "full": VRAITestSet.FULL,
    }
    test_set = test_set_map[args.test_set]

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.sam_dtype]

    print("Initializing SAM3BatchedImageHarness...")
    sam_harness = SAM3BatchedImageHarness(device=args.device, dtype=dtype)

    splits_to_process = ["train", "query", "gallery"] if args.split == "all" else [args.split]

    for split_str in splits_to_process:
        split = split_map[split_str]
        print(f"\n--- Loading VRAIDataset split {split.name} ---")
        vrai_ds = VRAIDataset(
            ds_root=ds_root, 
            main_split=split,
            test_set=test_set,
            sidecar_root=sidecar_root, 
            load_image_tensor=True, 
            verbose=True
        )

        loader = DataLoader(
            vrai_ds, 
            batch_size=args.batch_size, 
            collate_fn=custom_collate, 
            shuffle=False, 
            num_workers=args.num_workers
        )

        def get_items_and_prompts_fn(batch):
            return [(batch, "Person")]

        def get_image_tensor_fn(item):
            return item.frame_tensor

        def get_save_path_fn(item):
            frame_path = item.frame_path
            relative_parent_path = Path("images_train") if split == VRAISplit.TRAIN else Path("images_dev")
            segmentation_path = sidecar_root / relative_parent_path / f"{frame_path.stem}_major_mask.png"
            return segmentation_path

        print(f"Starting processing for {split.name}...")
        process_sam_masks_for_dataset(
            loader=loader,
            sam_harness=sam_harness,
            device=args.device,
            get_items_and_prompts_fn=get_items_and_prompts_fn,
            get_image_tensor_fn=get_image_tensor_fn,
            get_save_path_fn=get_save_path_fn,
            mask_selection_strategy="max_area",
            desc=f"Processing VRAI {split.name}",
            skip_existing=True,
        )

if __name__ == "__main__":
    main()
