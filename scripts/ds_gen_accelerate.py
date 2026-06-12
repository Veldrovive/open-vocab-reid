#!/usr/bin/env python3

"""
uvx yt-dlp -f "bestvideo[vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best" --merge-output-format mp4 \
    -o [NAME].mp4 \
    https://www.youtube.com/watch?v=[VIDEO_ID]

uv run scripts/unsupervised_ds_gen_parallel.py \
    --device cuda \
    --batch-size 30 \
    --skip-frames 4 \
    --overlap 5 \
    --edge-width 15 \
    --min-cc-ratio 0.05 \
    --verbose

uv run accelerate launch scripts/ds_gen_accelerate.py \
    --batch-size 30 \
    --skip-frames 4 \
    --overlap 5 \
    --edge-width 15 \
    --min-cc-ratio 0.05 \
    --verbose
"""

import os
import argparse
from pathlib import Path
from pydantic import BaseModel, ValidationError
import yaml
import math
import cv2
import numpy as np
from PIL import Image
from typing import Iterator, List, Any
from tqdm import tqdm
from shutil import rmtree
import imageio
import threading
import queue
import traceback
from accelerate import Accelerator

from open_vocab_mot import UNSUPERVISED_DATASET_INPUT_PATH, UNSUPERVISED_DATASET_OUTPUT_PATH

from aidan_lib.video_utils.load_batched_frames import load_constrained_batched_frames as orig_load_batched_frames
from aidan_lib.video_utils.scene_split import ConstrainedScene, get_constrained_scenes as orig_get_constrained_scenes, get_transnet_model
from aidan_lib.video_utils.video_data import GenericVideoData, get_video_data as orig_get_video_data
from aidan_lib.models.sam3_video import generate_video_segmentation, SAM3Harness
from aidan_lib.visualization.segmentations import visualize_segmentations, int_mask_to_binary_masks
from aidan_lib.visualization.colors import get_colors

# --- Wrappers for directory support ---

DEFAULT_FPS = 15.0

def get_video_data(vid_path: Path) -> GenericVideoData:
    if vid_path.is_dir():
        images = sorted([p for p in vid_path.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        frame_count = len(images)
        if frame_count == 0:
            return GenericVideoData(fps=DEFAULT_FPS, width=0, height=0, frame_count=0)
        first_img = cv2.imread(str(images[0]))
        height, width = first_img.shape[:2]
        return GenericVideoData(fps=DEFAULT_FPS, width=width, height=height, frame_count=frame_count)
    else:
        return orig_get_video_data(vid_path)

def get_constrained_scenes(vid_path: Path, model, threshold=0.75) -> List[Any]:
    if vid_path.is_dir():
        images = sorted([p for p in vid_path.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        frame_count = len(images)
        return [ConstrainedScene(
            shot_id=0,
            start_frame=0,
            end_frame=frame_count - 1,
            start_time=0.0,
            end_time=float((frame_count - 1) / DEFAULT_FPS) if frame_count > 0 else 0.0,
            probability=1.0
        )]
    else:
        return orig_get_constrained_scenes(vid_path, model, threshold)

def load_constrained_batched_frames(
    vid_path: Path, 
    constrained_scenes,
    batch_size: int = 120, 
    skip_frames: int | None = None, 
    convert_pil: bool = True,
    overlap: int = 0
):
    if vid_path.is_dir():
        images = sorted([p for p in vid_path.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        global_frame = -1
        batch = []
        batch_frames = []
        has_unyielded_frames = False
        
        if not constrained_scenes:
            return
            
        scene_index = 0
        current_scene = constrained_scenes[scene_index]

        def get_start_end(scene):
            if isinstance(scene, dict):
                return scene['start_frame'], scene['end_frame']
            return scene.start_frame, scene.end_frame

        def get_total_frames(start: int, end: int, skip: int | None) -> int:
            if skip is None:
                return max(0, end - start + 1)
            first_frame = start + (skip - (start % skip)) % skip
            last_frame = end - (end % skip)
            if first_frame > last_frame:
                return 0
            return (last_frame - first_frame) // skip + 1

        def get_target_batch_size(total: int, read: int, max_size: int, overlap_val: int, current_len: int) -> int:
            remaining = total - read
            if remaining <= 0:
                return max_size
                
            first_batch_capacity = max_size - current_len
            if remaining <= first_batch_capacity:
                return current_len + remaining
                
            max_new = max(1, max_size - overlap_val)
            num_batches = 1 + math.ceil((remaining - first_batch_capacity) / max_new)
            
            target_new = math.ceil(remaining / num_batches)
            
            return min(max_size, current_len + target_new)

        start_f, end_f = get_start_end(current_scene)
        scene_frames_total = get_total_frames(start_f, end_f, skip_frames)
        scene_frames_read = 0
        current_target_batch_size = get_target_batch_size(scene_frames_total, scene_frames_read, batch_size, overlap, len(batch))

        for global_frame, img_path in enumerate(images):
            while global_frame > end_f:
                if has_unyielded_frames:
                    yield batch, batch_frames, True
                    has_unyielded_frames = False
                    
                    if overlap > 0:
                        batch = batch[-overlap:]
                        batch_frames = batch_frames[-overlap:]
                    else:
                        batch, batch_frames = [], []
                    
                scene_index += 1
                if scene_index < len(constrained_scenes):
                    current_scene = constrained_scenes[scene_index]
                    start_f, end_f = get_start_end(current_scene)
                    scene_frames_total = get_total_frames(start_f, end_f, skip_frames)
                    scene_frames_read = 0
                    current_target_batch_size = get_target_batch_size(scene_frames_total, scene_frames_read, batch_size, overlap, len(batch))
                else:
                    break
                    
            if scene_index >= len(constrained_scenes):
                break
                
            if global_frame < start_f:
                continue

            if skip_frames is not None and global_frame % skip_frames != 0:
                continue

            if convert_pil:
                frame = Image.open(str(img_path)).convert("RGB")
            else:
                frame = cv2.imread(str(img_path))

            batch.append(frame)
            batch_frames.append(global_frame)
            scene_frames_read += 1
            has_unyielded_frames = True

            if len(batch) >= current_target_batch_size:
                is_end_of_scene = scene_frames_read >= scene_frames_total
                yield batch, batch_frames, is_end_of_scene
                has_unyielded_frames = False
                
                if overlap > 0:
                    batch = batch[-overlap:]
                    batch_frames = batch_frames[-overlap:]
                else:
                    batch, batch_frames = [], []
                    
                current_target_batch_size = get_target_batch_size(scene_frames_total, scene_frames_read, batch_size, overlap, len(batch))
        
        if has_unyielded_frames:
            yield batch, batch_frames, True
    else:
        yield from orig_load_batched_frames(vid_path, constrained_scenes, batch_size, skip_frames, convert_pil, overlap)

# --- End Wrappers ---

class VideoDirConfig(BaseModel):
    prompts: list[str]

class VideoConfig(BaseModel):
    added_prompts: list[str]
    removed_prompts: list[str]

def combine_prompts(base_prompts: list[str], added_prompts: list[str], removed_prompts: list[str]):
    prompts = set(base_prompts)
    prompts.update(added_prompts)
    prompts.difference_update(removed_prompts)
    return list(prompts)

class VideoData(BaseModel):
    path: Path
    config: VideoConfig | None

class VideoDirData(BaseModel):
    path: Path
    config: VideoDirConfig
    videos: list[VideoData]

class VideoDirs(BaseModel):
    dirs_data: list[VideoDirData]

    def iter_videos(self) -> Iterator[tuple[VideoDirData, VideoData]]:
        for dir_data in self.dirs_data:
            for video_data in dir_data.videos:
                yield dir_data, video_data

ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".webm"]

def ingest_dataset_paths(ds_root: Path, allowed_video_extensions: list[str]) -> VideoDirs:
    for i, video_ext in enumerate(allowed_video_extensions):
        if video_ext[0] != ".":
            print(f"WARNING: Allowed video extension {video_ext} does not include a . as the first character. This is invalid. Adding one.")
            allowed_video_extensions[i] = f".{video_ext}"

    video_dirs: list[VideoDirData] = []
    for video_dir in ds_root.iterdir():
        print(f"Crawling video dir: {video_dir}")
        if not video_dir.is_dir():
            print(f"Skipping file {video_dir} as it is not a directory")
            continue

        config_file_path = video_dir / "config.yaml"
        if not config_file_path.exists():
            print(f"Skipping directory {video_dir} because it does not contain a config.yaml")
            continue

        with open(config_file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        try:
            video_dir_config = VideoDirConfig.model_validate(config_dict)
        except ValidationError as e:
            print(f"Failed to validate config for video dir {video_dir}")
            raise e

        video_datas: list[VideoData] = []
        for video_file in video_dir.iterdir():
            print(f"Checking file: {video_file}")
            if video_file.name == "config.yaml" or video_file.suffix == ".yaml":
                continue

            is_valid_video = False
            if video_file.is_file() and video_file.suffix in allowed_video_extensions:
                is_valid_video = True
            elif video_file.is_dir():
                is_valid_video = True

            if not is_valid_video:
                print(f"Skipping file {video_file} as it is not an allowed video format or directory")
                continue

            config_file = video_file.parent / f"{video_file.stem}.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)

                try:
                    video_config = VideoConfig.model_validate(config_dict)
                except ValidationError as e:
                    print(f"Failed to validate config for video {video_file}")
                    raise e
            else:
                video_config = None

            video_data = VideoData(path=video_file, config=video_config)
            video_datas.append(video_data)
        
        video_dir_data = VideoDirData(
            path=video_dir,
            config=video_dir_config,
            videos=video_datas
        )
        video_dirs.append(video_dir_data)
    
    return VideoDirs(dirs_data=video_dirs)


def io_processing_worker(work_queue, video_info, skip_frames, min_cc_ratio, edge_width,
                         needs_demo_video, tmp_demo_video_path,
                         has_full_segmentations, tmp_full_segmentations_dir,
                         has_cropped_segmentations, tmp_cropped_segmentations_dir):
    
    visible_seg_map: dict[int, list[int]] = {}  
    known_negatives: set[tuple[int, int]] = set()
    global_id_to_prompt: dict[int, str] = {}
    global_id_to_color: dict[int, tuple] = {}

    if needs_demo_video:
        demo_video_frame_rate = video_info.fps / skip_frames if skip_frames is not None else video_info.fps
        demo_video_writer = imageio.get_writer(tmp_demo_video_path, fps=demo_video_frame_rate, format="mp4", codec="libx264")
    else:
        demo_video_writer = None

    try:
        while True:
            item = work_queue.get()
            if item is None:  # Sentinel value indicating we are done
                break
                
            frame_num, frame, segmentations, background_index = item

            all_masks = []
            all_obj_ids = []
            all_obj_prompts = []

            for prompt, sam_seg in segmentations.items():
                masks, obj_ids = int_mask_to_binary_masks(sam_seg, background_index=background_index)
                
                # Filter out small connected components
                new_masks = []
                for mask in masks:
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                    if num_labels > 1:
                        areas = stats[1:, cv2.CC_STAT_AREA]
                        max_area = np.max(areas)
                        valid_labels = np.where(areas >= max_area * min_cc_ratio)[0] + 1
                        
                        filtered_mask = np.isin(labels, valid_labels)
                        
                        # Remove filtered pixels from sam_seg
                        removed_pixels = mask & ~filtered_mask
                        sam_seg[removed_pixels] = background_index
                        
                        new_masks.append(filtered_mask)
                    else:
                        new_masks.append(mask)
                
                all_masks.extend(new_masks)
                all_obj_ids.extend(obj_ids)
                all_obj_prompts.extend([prompt] * len(obj_ids))
            
            masks = all_masks
            obj_ids = all_obj_ids
            obj_prompts = all_obj_prompts

            # Assign consistent colors to new obj_ids
            for obj_id in obj_ids:
                if obj_id not in global_id_to_color:
                    global_id_to_color[obj_id] = get_colors(len(global_id_to_color) + 1)[-1]
            
            # We can use the unique ids to populate the visible seg map for this frame
            visible_seg_map[frame_num] = obj_ids

            # Update the global map from id to prompt
            for obj_id, prompt in zip(obj_ids, obj_prompts):
                global_id_to_prompt[obj_id] = prompt

            # Known negatives constraint: appear at the same time and at least one point have no overlap
            for i in range(len(obj_ids)):
                for j in range(i+1, len(obj_ids)):
                    lower = min(obj_ids[i], obj_ids[j])
                    higher = max(obj_ids[i], obj_ids[j])
                    if (lower, higher) not in known_negatives:
                        if not np.any(masks[i] & masks[j]):
                            known_negatives.add((lower, higher))

            if needs_demo_video:
                labels = [f"{obj_prompt} {obj_id}" for obj_prompt, obj_id in zip(obj_prompts, obj_ids)]
                frame_colors = [global_id_to_color[obj_id] for obj_id in obj_ids]
                segmented_frame = visualize_segmentations(frame, masks, colors=frame_colors, labels=labels)
                segmented_frame_array = np.expand_dims(np.array(segmented_frame), axis=0)
                demo_video_writer.append_data(segmented_frame_array)

            if not has_full_segmentations:
                frame_path = tmp_full_segmentations_dir / f"frame_{frame_num}.jpg"
                frame.save(frame_path)

                for prompt, sam_seg in segmentations.items():
                    safe_prompt = prompt.replace(" ", "_").replace("/", "_")
                    segmentation_path = tmp_full_segmentations_dir / f"frame_{frame_num}_{safe_prompt}_seg.png"
                    seg_img = Image.fromarray(sam_seg)
                    seg_img.save(segmentation_path)

            if not has_cropped_segmentations:
                frame_w, frame_h = frame.size

                for mask, obj_id in zip(masks, obj_ids):
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    
                    if not np.any(rows) or not np.any(cols):
                        continue
                        
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    
                    rmin = max(0, rmin - edge_width)
                    rmax = min(frame_h, rmax + edge_width + 1)
                    cmin = max(0, cmin - edge_width)
                    cmax = min(frame_w, cmax + edge_width + 1)
                    
                    cropped_frame = frame.crop((cmin, rmin, cmax, rmax))
                    
                    cropped_mask_array = mask[rmin:rmax, cmin:cmax]
                    cropped_mask_img = Image.fromarray((cropped_mask_array * 255).astype(np.uint8))
                    
                    obj_dir = tmp_cropped_segmentations_dir / f"id_{obj_id}"
                    obj_dir.mkdir(parents=True, exist_ok=True)
                    
                    cropped_frame.save(obj_dir / f"frame_{frame_num}.jpg")
                    cropped_mask_img.save(obj_dir / f"frame_{frame_num}_mask.png")

            work_queue.task_done()

        # --- GENERATE METADATA ---
        negatives_map: dict[int, list[int]] = {obj_id: [] for obj_id in global_id_to_prompt.keys()}
        for id1, id2 in known_negatives:
            negatives_map[id1].append(id2)
            negatives_map[id2].append(id1)
            
        if not has_cropped_segmentations:
            for obj_id, prompt in global_id_to_prompt.items():
                obj_dir = tmp_cropped_segmentations_dir / f"id_{obj_id}"
                if obj_dir.exists():
                    meta = {
                        "prompt": prompt,
                        "negatives": negatives_map.get(obj_id, [])
                    }
                    with open(obj_dir / "metadata.yaml", "w") as f:
                        yaml.dump(meta, f, default_flow_style=False)

        if not has_full_segmentations:
            full_meta = {
                "visible_seg_map": visible_seg_map,
                "known_negatives": [list(pair) for pair in known_negatives], 
                "id_to_prompt": global_id_to_prompt
            }
            with open(tmp_full_segmentations_dir / "metadata.yaml", "w") as f:
                yaml.dump(full_meta, f, default_flow_style=False)

    finally:
        if demo_video_writer is not None:
            demo_video_writer.close()


def main():
    accelerator = Accelerator()

    # Hide distributed environment variables from SAM3 so it processes videos independently 
    # per GPU instead of trying to distribute a single video's frames across all GPUs.
    if "WORLD_SIZE" in os.environ:
        del os.environ["WORLD_SIZE"]
    if "RANK" in os.environ:
        del os.environ["RANK"]
    if "LOCAL_RANK" in os.environ:
        del os.environ["LOCAL_RANK"]

    parser = argparse.ArgumentParser(description="Unsupervised dataset generation")
    parser.add_argument("--batch-size", type=int, default=120, help="Batch size for SAM3")
    parser.add_argument("--skip-frames", type=int, default=5, help="Number of frames to skip")
    parser.add_argument("--overlap", type=int, default=1, help="Overlap between batches")
    parser.add_argument("--edge-width", type=int, default=15, help="Edge width for cropped segmentations")
    parser.add_argument("--min-cc-ratio", type=float, default=0.05, help="Minimum connected component area ratio to the largest component")
    parser.add_argument("--no-demo-videos", action="store_true", help="Disable demo video creation")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--device", type=str, default=None, help="Device to use for models (e.g. cuda, cuda:0)")
    parser.add_argument("--num-prompt-applications", type=int, default=1, help="Number of prompt applications")
    parser.add_argument("--prompt-frame-spacing", type=str, default="space_between", choices=["space_around", "space_between"], help="Prompt frame spacing strategy")
    parser.add_argument("--iou-thresh", type=float, default=0.9, help="IOU threshold for segmentation")
    args = parser.parse_args()

    batch_size = args.batch_size
    skip_frames = args.skip_frames
    overlap = args.overlap
    create_demo_videos = not args.no_demo_videos
    edge_width = args.edge_width
    verbose = args.verbose
    device = args.device if args.device is not None else str(accelerator.device)
    min_cc_ratio = args.min_cc_ratio
    num_prompt_applications = args.num_prompt_applications
    prompt_frame_spacing = args.prompt_frame_spacing
    iou_thresh = args.iou_thresh

    video_dirs = ingest_dataset_paths(UNSUPERVISED_DATASET_INPUT_PATH, ALLOWED_VIDEO_EXTENSIONS)
    
    if accelerator.is_main_process:
        print(f"Found {len(video_dirs.dirs_data)} video directories:\n{video_dirs.dirs_data}")
        for dir_data, video_data in video_dirs.iter_videos():
            video_path = video_data.path
            video_parent_relpath = str(video_path.parent.relative_to(UNSUPERVISED_DATASET_INPUT_PATH))
            base_out_dir = UNSUPERVISED_DATASET_OUTPUT_PATH / video_parent_relpath / video_path.stem
            base_out_dir.mkdir(exist_ok=True, parents=True)
            
    accelerator.wait_for_everyone()

    if len(video_dirs.dirs_data) == 0:
        if accelerator.is_main_process:
            print("No videos found to process.")
        return

    sam = SAM3Harness(max_num_objects=64, device=device)
    transnet = get_transnet_model(device)

    for dir_data, video_data in video_dirs.iter_videos():
        video_path = video_data.path
        video_parent_relpath = str(video_path.parent.relative_to(UNSUPERVISED_DATASET_INPUT_PATH))
        base_out_dir = UNSUPERVISED_DATASET_OUTPUT_PATH / video_parent_relpath / video_path.stem

        video_info = get_video_data(video_path)

        prompts = combine_prompts(
            dir_data.config.prompts,
            video_data.config.added_prompts if video_data.config else [],
            video_data.config.removed_prompts if video_data.config else []
        )

        demo_video_path = base_out_dir / f"{video_path.stem}_segs.mp4"
        tmp_demo_video_path = base_out_dir / f"{video_path.stem}_segs_tmp.mp4"
        has_demo_video = demo_video_path.exists()
        needs_demo_video = not has_demo_video and create_demo_videos

        full_segmentations_dir = base_out_dir / "full_frame_segmentations"
        tmp_full_segmentations_dir = base_out_dir / "full_frame_segmentations_tmp"
        has_full_segmentations = full_segmentations_dir.exists()

        cropped_segmentations_dir = base_out_dir / "cropped_segmentations"
        tmp_cropped_segmentations_dir = base_out_dir / "cropped_segmentations_tmp"
        has_cropped_segmentations = cropped_segmentations_dir.exists()

        needs_processing = not (has_demo_video and has_full_segmentations and has_cropped_segmentations)

        if not needs_processing:
            continue

        lock_file = base_out_dir / ".processing.lock"
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
        except FileExistsError:
            continue

        print(f"[Process {accelerator.process_index}] Processing video {video_path} to {base_out_dir} with prompts {prompts}")

        if tmp_demo_video_path.exists():
            print(f"Removing old temp demo video")
            tmp_demo_video_path.unlink(missing_ok=True)

        if tmp_full_segmentations_dir.exists():
            print(f"Removing old temp full frame segmentations dir")
            rmtree(tmp_full_segmentations_dir, ignore_errors=True)
        tmp_full_segmentations_dir.mkdir(exist_ok=True)

        if tmp_cropped_segmentations_dir.exists():
            print(f"Removing old temp cropped segmentations dir")
            rmtree(tmp_cropped_segmentations_dir, ignore_errors=True)
        tmp_cropped_segmentations_dir.mkdir(exist_ok=True)

        constrained_scenes = get_constrained_scenes(video_path, transnet, threshold=0.75)

        batch_frame_loader = load_constrained_batched_frames(
            video_path,
            constrained_scenes,
            batch_size=batch_size,
            skip_frames=skip_frames,
            convert_pil=True,
            overlap=overlap
        )

        frame_seg_generator = generate_video_segmentation(
            harness=sam,
            prompts=prompts,
            batch_frame_loader=batch_frame_loader,
            num_prompt_applications=num_prompt_applications,
            prompt_frame_spacing=prompt_frame_spacing,
            iou_thresh=iou_thresh
        )

        try:
            frames_to_process = video_info.frame_count // skip_frames if skip_frames else video_info.frame_count
            progress = tqdm(total=frames_to_process, disable=not verbose, position=accelerator.process_index)
            
            # Setup a queue with a strict limit to prevent RAM bloat
            work_queue = queue.Queue(maxsize=50) 
            
            # Start the I/O processing worker in a background thread
            io_thread = threading.Thread(
                target=io_processing_worker,
                args=(
                    work_queue, video_info, skip_frames, min_cc_ratio, edge_width,
                    needs_demo_video, tmp_demo_video_path,
                    has_full_segmentations, tmp_full_segmentations_dir,
                    has_cropped_segmentations, tmp_cropped_segmentations_dir
                )
            )
            io_thread.start()

            # The Main Thread only cares about pushing GPU inference to the queue
            for frame_info in frame_seg_generator:
                frame_num, frame, segmentations, background_index = frame_info
                progress.update(1)
                progress.set_description(f"Frame {frame_num}/{video_info.frame_count}")
                
                # Push the data to the queue. 
                # This will block if the queue hits maxsize (50), keeping RAM in check.
                work_queue.put((frame_num, frame, segmentations, background_index))

            # Send the shutdown signal to the worker
            work_queue.put(None)
            
            # Wait for all I/O processing and metadata generation to finish
            io_thread.join()

            # Now that the thread is fully done and closed all files, rename them
            if needs_demo_video: 
                tmp_demo_video_path.rename(demo_video_path)

            if not has_full_segmentations:
                tmp_full_segmentations_dir.rename(full_segmentations_dir)
                
            if not has_cropped_segmentations:
                tmp_cropped_segmentations_dir.rename(cropped_segmentations_dir)

            lock_file.unlink(missing_ok=True)
            print(f"[Process {accelerator.process_index}] Finished processing video {video_path} to {base_out_dir}")

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            traceback.print_exc()
            # Ensure we don't leave a hanging thread if an exception occurs in the main thread
            if 'work_queue' in locals():
                work_queue.put(None)
                io_thread.join()
            lock_file.unlink(missing_ok=True)

    # Release GPU memory proactively when the process finishes all available videos
    print(f"[Process {accelerator.process_index}] Finished all available videos. Releasing GPU memory and exiting.")
    del sam
    del transnet
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    
    # # Force exit to ensure process terminates and frees any remaining resources
    # import sys
    # sys.exit(0)

    # Instead, we wait till everything gets here. Since we already relesased resources it's fine
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()