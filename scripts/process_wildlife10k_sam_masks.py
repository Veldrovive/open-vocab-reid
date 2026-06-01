import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image, ImageReadMode

from open_vocab_mot.data import Wildlife10kDataset, custom_collate, process_sam_masks_for_dataset
from aidan_lib.models.sam3_batched_img import SAM3BatchedImageHarness


def main():
    parser = argparse.ArgumentParser(description="Process Wildlife10k dataset and save SAM3 masks")
    parser.add_argument("--ds-root", type=str, required=True, help="Path to Wildlife10k dataset root directory")
    parser.add_argument("--sidecar-root", type=str, required=True, help="Path to save the generated sidecar masks")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SAM3 on (e.g. 'cuda', 'cuda:0')")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--sam-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="SAM3 dtype precision")
    args = parser.parse_args()

    ds_root = Path(args.ds_root)
    sidecar_root = Path(args.sidecar_root)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.sam_dtype]

    print("Loading Wildlife10kDataset...")
    wildlife_ds = Wildlife10kDataset(
        ds_root=ds_root, 
        sidecar_root=sidecar_root, 
        load_image_tensor=False, 
        verbose=True,
        download_if_missing=True,
        split=None  # Process the entire dataset
    )

    print("Initializing SAM3BatchedImageHarness...")
    sam_harness = SAM3BatchedImageHarness(device=args.device, dtype=dtype, score_threshold=0.0)

    loader = DataLoader(
        wildlife_ds, 
        batch_size=args.batch_size, 
        collate_fn=custom_collate, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    def get_items_and_prompts_fn(batch):
        species_to_items = {}
        for item in batch:
            species = wildlife_ds._frame_list[item.sample_index]['species']
            if species not in species_to_items:
                species_to_items[species] = []
            species_to_items[species].append(item)
            
        sub_batches = []
        species_prompt_map = {
            "whale": "Whale or Beluga in water"
        }
        for species, items in species_to_items.items():
            prompt = species_prompt_map.get(species.lower(), species)
            prompt = f"{prompt} or Animal"
            sub_batches.append((items, prompt))
        return sub_batches

    def get_image_tensor_fn(item):
        image_tensor = read_image(str(item.frame_path), ImageReadMode.RGB)
        return image_tensor.float() / 255.0

    def get_save_path_fn(item):
        rel_path = item.frame_path.relative_to(ds_root)
        return sidecar_root / rel_path.with_suffix('.png')

    print("Starting processing...")
    process_sam_masks_for_dataset(
        loader=loader,
        sam_harness=sam_harness,
        device=args.device,
        get_items_and_prompts_fn=get_items_and_prompts_fn,
        get_image_tensor_fn=get_image_tensor_fn,
        get_save_path_fn=get_save_path_fn,
        mask_selection_strategy="max_score",
        desc="Processing Wildlife10k",
        skip_existing=True,
    )


if __name__ == "__main__":
    main()
