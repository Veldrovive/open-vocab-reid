import argparse
import time
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from open_vocab_mot import DUKEMTMC_VIDEO_REID_PATH, DUKEMTMC_VIDEO_REID_SIDECAR_PATH
from open_vocab_mot.data import DukeMTMCVideoDataset, DukeSplit, collate_duke_mtmc_video_ds, DukeMTMCItemBatch
from aidan_lib.models.sam3_batched_img import SAM3BatchedImageHarness


def main():
    parser = argparse.ArgumentParser(description="Process DukeMTMC-VideoReID dataset and save SAM3 masks")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run SAM3 on (e.g. 'cuda', 'cuda:0')")
    parser.add_argument("--split", type=str, default="train", choices=["train", "query", "gallery"], help="Dataset split to process")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--sam-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="SAM3 dtype precision")
    args = parser.parse_args()

    # if args.device.startswith("cuda"):
    #     torch.cuda.set_device(args.device)

    split_map = {
        "train": DukeSplit.TRAIN,
        "query": DukeSplit.QUERY,
        "gallery": DukeSplit.GALLERY,
    }
    split = split_map[args.split]

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.sam_dtype]

    print(f"Loading DukeMTMCVideoDataset split {split.name}...")
    duke_ds = DukeMTMCVideoDataset(DUKEMTMC_VIDEO_REID_PATH, split, load_image=True, verbose=True)

    print("Initializing SAM3BatchedImageHarness...")
    sam_harness = SAM3BatchedImageHarness(device=args.device, dtype=dtype)

    loader = DataLoader(
        duke_ds, 
        batch_size=args.batch_size, 
        collate_fn=collate_duke_mtmc_video_ds, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    time_converting_to_tensor = 0
    time_in_sam = 0
    time_finding_best_mask = 0
    time_saving = 0

    print("Starting processing...")
    for batch in tqdm(loader, desc=f"Processing {split.name}"):
        assert isinstance(batch, DukeMTMCItemBatch)
        tensor_image_batch = []
        
        start_time = time.perf_counter()
        for img in batch.frames:
            tensor_image_batch.append(
                TF.to_tensor(img).to(args.device)
            )
        time_converting_to_tensor += time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        sam_output = sam_harness(tensor_image_batch, "Person", move_to_cpu=False)
        time_in_sam += time.perf_counter() - start_time

        start_time = time.perf_counter()
        major_masks = []
        for i, frame_out in enumerate(sam_output):
            if len(frame_out.masks) == 0:
                # If no masks are found, append an empty mask
                _, H, W = tensor_image_batch[i].shape
                major_masks.append(torch.zeros((H, W), dtype=torch.bool, device=args.device))
            else:
                # Assume the mask with the largest area is the "major mask"
                major_mask_idx = torch.argmax(frame_out.masks.sum((1, 2)))
                major_masks.append(frame_out.masks[major_mask_idx])
        time_finding_best_mask += time.perf_counter() - start_time

        start_time = time.perf_counter()
        for idx in range(len(batch.frame_paths)):
            frame_path = batch.frame_paths[idx]
            parent_path = frame_path.parent
            frame_stem = frame_path.stem
            major_mask = major_masks[idx]

            # Get the path in the sidecar dir
            parent_rel_path = parent_path.relative_to(DUKEMTMC_VIDEO_REID_PATH)
            sidecar_parent_path = DUKEMTMC_VIDEO_REID_SIDECAR_PATH / parent_rel_path
            # Ensure our parent directory exists
            sidecar_parent_path.mkdir(parents=True, exist_ok=True)

            # Save the mask
            sidecar_mask_path = sidecar_parent_path / f"{frame_stem}_major_mask.png"
            float_mask = major_mask.float()
            save_image(float_mask, sidecar_mask_path)
        time_saving += time.perf_counter() - start_time

    print("Finished processing!")
    print(f"Time converting to tensor: {time_converting_to_tensor:.2f}s")
    print(f"Time in SAM: {time_in_sam:.2f}s")
    print(f"Time finding best mask: {time_finding_best_mask:.2f}s")
    print(f"Time saving: {time_saving:.2f}s")


if __name__ == "__main__":
    main()
