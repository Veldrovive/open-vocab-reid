import argparse
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Assuming your imports map like this based on the abstract structure
from open_vocab_mot.data import WhaleDataset
from open_vocab_mot.data.video_reid_abc import VideoReIDItem
from aidan_lib.models.sam3_batched_img import SAM3BatchedImageHarness


def custom_collate(batch: list[VideoReIDItem]) -> list[VideoReIDItem]:
    """Simple collate to keep the list of items intact without a specific Batch dataclass."""
    return batch


def main():
    parser = argparse.ArgumentParser(description="Process Whale dataset and save SAM3 masks")
    parser.add_argument("--ds-root", type=str, required=True, help="Path to Whale dataset root directory")
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

    print("Loading WhaleDataset...")
    whale_ds = WhaleDataset(
        ds_root=ds_root, 
        sidecar_root=sidecar_root, 
        load_image_tensor=True, 
        verbose=True,
        download_if_missing=True
    )

    print("Initializing SAM3BatchedImageHarness...")
    sam_harness = SAM3BatchedImageHarness(device=args.device, dtype=dtype)

    loader = DataLoader(
        whale_ds, 
        batch_size=args.batch_size, 
        collate_fn=custom_collate, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    time_converting_to_tensor = 0
    time_in_sam = 0
    time_finding_best_mask = 0
    time_saving = 0

    print("Starting processing...")
    for batch in tqdm(loader, desc="Processing Whales"):
        tensor_image_batch = []
        
        start_time = time.perf_counter()
        for item in batch:
            # WhaleDataset already gives us a normalized float tensor
            tensor_image_batch.append(item.frame_tensor.to(args.device))
        time_converting_to_tensor += time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        # Use "Whale" as the target prompt for this dataset
        sam_output = sam_harness(tensor_image_batch, "Whale", move_to_cpu=False)
        time_in_sam += time.perf_counter() - start_time

        start_time = time.perf_counter()
        major_masks = []
        for i, frame_out in enumerate(sam_output):
            if len(frame_out.masks) == 0:
                _, H, W = tensor_image_batch[i].shape
                major_masks.append(torch.zeros((H, W), dtype=torch.bool, device=args.device))
            else:
                major_mask_idx = torch.argmax(frame_out.masks.sum((1, 2)))
                major_masks.append(frame_out.masks[major_mask_idx])
        time_finding_best_mask += time.perf_counter() - start_time

        start_time = time.perf_counter()
        for idx, item in enumerate(batch):
            major_mask = major_masks[idx]

            # Reconstruct the correct relative path to mirror the dataset structure
            rel_path = item.frame_path.relative_to(ds_root)
            
            # The WhaleDataset implementation natively checks for .png fallbacks, 
            # so we just replace the extension rather than appending a suffix
            sidecar_mask_path = sidecar_root / rel_path.with_suffix('.png')
            
            # Ensure our parent directory exists
            sidecar_mask_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the mask
            float_mask = major_mask.float()
            save_image(float_mask, sidecar_mask_path)
        time_saving += time.perf_counter() - start_time

    print("Finished processing!")
    print(f"Time moving to device: {time_converting_to_tensor:.2f}s")
    print(f"Time in SAM: {time_in_sam:.2f}s")
    print(f"Time finding best mask: {time_finding_best_mask:.2f}s")
    print(f"Time saving: {time_saving:.2f}s")


if __name__ == "__main__":
    main()