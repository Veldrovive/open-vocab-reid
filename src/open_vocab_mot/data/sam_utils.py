import time
from pathlib import Path
from typing import Callable, Iterable, Any, List
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from aidan_lib.models.sam3_batched_img import SAM3BatchedImageHarness
from open_vocab_mot.data.video_reid_abc import VideoReIDItem

def custom_collate(batch: List[VideoReIDItem]) -> List[VideoReIDItem]:
    """Simple collate to keep the list of items intact without a specific Batch dataclass."""
    return batch


def process_sam_masks_for_dataset(
    loader: Iterable,
    sam_harness: SAM3BatchedImageHarness,
    device: str,
    get_items_and_prompts_fn: Callable[[Any], list[tuple[list, str]]],
    get_image_tensor_fn: Callable[[Any], torch.Tensor],
    get_save_path_fn: Callable[[Any], Path],
    mask_selection_strategy: str = "max_area", # "max_area" or "max_score"
    desc: str = "Processing SAM Masks",
    skip_existing: bool = True,
):
    """
    Unified loop for processing dataset items with SAM and saving the resulting masks.
    
    Args:
        loader: DataLoader for the dataset.
        sam_harness: Initialized SAM3BatchedImageHarness.
        device: Target device for inference.
        get_items_and_prompts_fn: A function that takes a batch from the loader and returns 
                                  a list of (sub_batch_items, prompt) tuples.
        get_image_tensor_fn: A function that takes a single item and returns its image as a torch.Tensor (C, H, W).
                             The tensor can be on CPU and will be moved to device internally.
        get_save_path_fn: A function that takes a single item and returns the absolute Path where 
                          its sidecar mask should be saved.
        mask_selection_strategy: Either "max_area" (largest mask by area) or "max_score" (highest confidence score).
        desc: tqdm description.
        skip_existing: If True, skips processing items whose sidecar mask already exists.
    """
    time_converting_to_tensor = 0
    time_in_sam = 0
    time_finding_best_mask = 0
    time_saving = 0

    for batch in tqdm(loader, desc=desc):
        sub_batches = get_items_and_prompts_fn(batch)
        
        for items, prompt in sub_batches:
            if skip_existing:
                items_to_process = []
                for item in items:
                    save_path = get_save_path_fn(item)
                    if not save_path.exists():
                        items_to_process.append(item)
                items = items_to_process
                
            if not items:
                continue
                
            tensor_image_batch = []
            start_time = time.perf_counter()
            for item in items:
                tensor = get_image_tensor_fn(item)
                tensor_image_batch.append(tensor.to(device))
            time_converting_to_tensor += time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            sam_output = sam_harness(tensor_image_batch, prompt, move_to_cpu=False)
            time_in_sam += time.perf_counter() - start_time

            start_time = time.perf_counter()
            major_masks = []
            empty_count = 0
            for i, frame_out in enumerate(sam_output):
                if len(frame_out.masks) == 0:
                    empty_count += 1
                    _, H, W = tensor_image_batch[i].shape
                    major_masks.append(torch.zeros((H, W), dtype=torch.bool, device=device))
                else:
                    if mask_selection_strategy == "max_area":
                        major_mask_idx = torch.argmax(frame_out.masks.sum((1, 2)))
                        major_masks.append(frame_out.masks[major_mask_idx])
                    elif mask_selection_strategy == "max_score":
                        max_score_idx = torch.argmax(frame_out.scores)
                        major_masks.append(frame_out.masks[max_score_idx])
                    else:
                        raise ValueError(f"Unknown mask_selection_strategy: {mask_selection_strategy}")
            if empty_count > 0:
                print(f"Found {empty_count}/{len(items)} empty masks for prompt '{prompt}'")
            time_finding_best_mask += time.perf_counter() - start_time

            start_time = time.perf_counter()
            for idx, item in enumerate(items):
                sidecar_mask_path = get_save_path_fn(item)
                # Ensure our parent directory exists
                sidecar_mask_path.parent.mkdir(parents=True, exist_ok=True)
                # Save the mask
                major_mask = major_masks[idx]
                float_mask = major_mask.float()
                save_image(float_mask, sidecar_mask_path)
            time_saving += time.perf_counter() - start_time

    print("Finished processing!")
    print(f"Time moving to device/converting to tensor: {time_converting_to_tensor:.2f}s")
    print(f"Time in SAM: {time_in_sam:.2f}s")
    print(f"Time finding best mask: {time_finding_best_mask:.2f}s")
    print(f"Time saving: {time_saving:.2f}s")
