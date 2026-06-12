from pathlib import Path
from typing import Iterator, NamedTuple
import yaml
import torch
import random
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from torch.utils.data import IterableDataset, get_worker_info

from open_vocab_mot.data.video_reid_abc import VideoReIDItem

class TrackletInfo(NamedTuple):
    global_id: int
    video_name: str
    original_obj_id: int
    prompt: str
    negative_global_ids: list[int]
    frame_paths: list[Path]
    mask_paths: list[Path]

class UnsupervisedVideoReIDDataset:
    def __init__(
        self,
        ds_root: Path | str,
        num_sequences_per_tracklet: int,
        num_frames_per_sequence: int,
        warn_skipped_identities: bool = True,
        load_image_pil: bool = False,
        load_image_tensor: bool = False,
        image_tensor_dtype: torch.dtype | None = None,
        load_segmentations: bool = False,
        transform=None,
    ):
        self.ds_root = Path(ds_root)
        self.num_sequences_per_tracklet = num_sequences_per_tracklet
        self.num_frames_per_sequence = num_frames_per_sequence
        self.warn_skipped_identities = warn_skipped_identities
        self.load_image_pil = load_image_pil
        self.load_image_tensor = load_image_tensor
        self.image_tensor_dtype = image_tensor_dtype
        self.load_segmentations = load_segmentations
        self.transform = transform
        
        self.min_frames_required = self.num_sequences_per_tracklet * self.num_frames_per_sequence

        self.tracklets: dict[int, TrackletInfo] = {}
        self.video_to_tracklets: dict[str, list[int]] = {}
        
        # Flat list for sample_index resolution if needed
        self._flat_frames: list[tuple[int, int]] = [] 
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        global_id_counter = 0
        skipped_count = 0
        passed_count = 0
        
        temp_id_map: dict[tuple[str, int], int] = {}
        temp_negatives: dict[int, list[int]] = {}
        
        video_dirs = [d for d in self.ds_root.iterdir() if d.is_dir()]
        for video_dir in video_dirs:
            video_name = video_dir.name
            cropped_dir = video_dir / "cropped_segmentations"
            if not cropped_dir.exists():
                continue
                
            id_dirs = [d for d in cropped_dir.iterdir() if d.is_dir() and d.name.startswith("id_")]
            for id_dir in id_dirs:
                try:
                    original_obj_id = int(id_dir.name.split("_")[1])
                except ValueError:
                    continue
                    
                metadata_path = id_dir / "metadata.yaml"
                if not metadata_path.exists():
                    continue
                    
                with open(metadata_path, 'r') as f:
                    meta = yaml.safe_load(f)
                    
                prompt = meta.get("prompt", "UNKNOWN")
                negatives = meta.get("negatives", [])
                
                frame_paths = sorted(list(id_dir.glob("frame_*.jpg")))
                frame_paths = sorted(frame_paths, key=lambda x: int(x.stem.split("_")[1]))
                
                valid_frame_paths = []
                valid_mask_paths = []
                
                for fp in frame_paths:
                    mp = fp.parent / f"{fp.stem}_mask.png"
                    if mp.exists():
                        valid_frame_paths.append(fp)
                        valid_mask_paths.append(mp)
                        
                if len(valid_frame_paths) < self.min_frames_required:
                    if self.warn_skipped_identities:
                        print(f"Warning: Skipping tracklet {original_obj_id} in {video_name} because it has {len(valid_frame_paths)} frames, but {self.min_frames_required} are required.")
                    skipped_count += 1
                    continue
                    
                global_id = global_id_counter
                global_id_counter += 1
                
                temp_id_map[(video_name, original_obj_id)] = global_id
                temp_negatives[global_id] = negatives
                
                tracklet_info = TrackletInfo(
                    global_id=global_id,
                    video_name=video_name,
                    original_obj_id=original_obj_id,
                    prompt=prompt,
                    negative_global_ids=[], 
                    frame_paths=valid_frame_paths,
                    mask_paths=valid_mask_paths
                )
                self.tracklets[global_id] = tracklet_info
                
                if video_name not in self.video_to_tracklets:
                    self.video_to_tracklets[video_name] = []
                self.video_to_tracklets[video_name].append(global_id)
                
                for f_idx in range(len(valid_frame_paths)):
                    self._flat_frames.append((global_id, f_idx))
                
                passed_count += 1
                
        # Link negatives
        for global_id, tracklet in self.tracklets.items():
            neg_orig_ids = temp_negatives[global_id]
            valid_neg_global_ids = []
            for n_id in neg_orig_ids:
                n_global = temp_id_map.get((tracklet.video_name, n_id))
                if n_global is not None:
                    valid_neg_global_ids.append(n_global)
            self.tracklets[global_id] = tracklet._replace(negative_global_ids=valid_neg_global_ids)
            
        print(f"Filtered dataset: {passed_count} identities passed, {skipped_count} skipped. This equates to {passed_count * self.num_sequences_per_tracklet} sequences generated.")

    def get_item(self, global_id: int, frame_idx: int, sequence_id: int, sample_index: int) -> VideoReIDItem:
        tracklet = self.tracklets[global_id]
        frame_path = tracklet.frame_paths[frame_idx]
        segmentation_path = tracklet.mask_paths[frame_idx]
        
        image_pil = None
        if self.load_image_pil:
            image_pil = Image.open(frame_path).convert("RGB")
            
        image_tensor = None
        if self.load_image_tensor:
            image_tensor = read_image(str(frame_path), ImageReadMode.RGB)
            if self.image_tensor_dtype is not None:
                image_tensor = image_tensor.to(self.image_tensor_dtype) / 255.0
            else:
                image_tensor = image_tensor.float() / 255.0
                
        segmentation_tensor = None
        if self.load_segmentations:
            segmentation_tensor = read_image(str(segmentation_path), ImageReadMode.GRAY)
            segmentation_tensor = segmentation_tensor.float() / 255.0

        if self.transform is not None:
            if image_tensor is not None:
                image_tensor = tv_tensors.Image(image_tensor)
            if segmentation_tensor is not None:
                segmentation_tensor = tv_tensors.Mask(segmentation_tensor)
            
            if image_tensor is not None and segmentation_tensor is not None:
                image_tensor, segmentation_tensor = self.transform(image_tensor, segmentation_tensor)
            elif image_tensor is not None:
                image_tensor = self.transform(image_tensor)
            elif segmentation_tensor is not None:
                segmentation_tensor = self.transform(segmentation_tensor)

        return VideoReIDItem(
            sample_index=sample_index,
            identity_id=global_id,
            sequence_id=sequence_id,
            frame_id=frame_path.stem,
            frame_path=frame_path,
            frame=image_pil,
            frame_tensor=image_tensor,
            segmentation_path=segmentation_path,
            segmentation_tensor=segmentation_tensor,
        )

class UnsupervisedBatchIterableDataset(IterableDataset[list[VideoReIDItem]]):
    def __init__(
        self,
        dataset: UnsupervisedVideoReIDDataset,
        batches_per_epoch: int | None,
        num_identities_per_batch: int,
        num_hard_negatives_per_positive: int,
        num_sequences_per_tracklet: int,
        num_frames_per_sequence: int,
        epoch_deterministic: bool = False,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.dataset = dataset
        self.batches_per_epoch = batches_per_epoch
        self.num_identities_per_batch = num_identities_per_batch
        self.num_hard_negatives_per_positive = num_hard_negatives_per_positive
        self.num_sequences_per_tracklet = num_sequences_per_tracklet
        self.num_frames_per_sequence = num_frames_per_sequence
        self.epoch_deterministic = epoch_deterministic
        self.seed = seed
        self.verbose = verbose

    def _split_tracklet(self, rng: random.Random, tracklet_id: int) -> list[list[int]]:
        tracklet = self.dataset.tracklets[tracklet_id]
        num_frames = len(tracklet.frame_paths)
        chunk_size = num_frames // self.num_sequences_per_tracklet
        
        sequences = []
        for i in range(self.num_sequences_per_tracklet):
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size
            available_indices = list(range(chunk_start, chunk_end))
            sampled_indices = sorted(rng.sample(available_indices, self.num_frames_per_sequence))
            sequences.append(sampled_indices)
            
        return sequences

    def __iter__(self) -> Iterator[list[VideoReIDItem]]:
        worker_info = get_worker_info()
        
        seed = self.seed
        if seed is not None:
            if worker_info is not None:
                seed += worker_info.id
            if self.epoch_deterministic and self.verbose:
                print(f"Resetting RNG for worker {worker_info.id if worker_info else 0} with seed {seed}")
                
        rng = random.Random(seed) if seed is not None else random.Random()

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

        class TrackletPool:
            def __init__(self, video_to_tracklets, r):
                self.video_to_tracklets = {k: list(v) for k, v in video_to_tracklets.items() if len(v) > 0}
                self.rng = r
                self.reset()
                
            def reset(self):
                self.available_videos = list(self.video_to_tracklets.keys())
                self.rng.shuffle(self.available_videos)
                self.available_tracklets = {k: list(v) for k, v in self.video_to_tracklets.items()}
                for k in self.available_tracklets:
                    self.rng.shuffle(self.available_tracklets[k])
                    
            def get_tracklets(self, k: int):
                sampled = []
                while len(sampled) < k:
                    valid_videos = [v for v in self.available_videos if self.available_tracklets[v]]
                    if not valid_videos:
                        self.reset()
                        valid_videos = [v for v in self.available_videos if self.available_tracklets[v]]
                        if not valid_videos:
                            raise ValueError("No tracklets available even after reset.")
                    
                    used_videos = {ds.tracklets[tid].video_name for tid in sampled}
                    unused_valid_videos = [v for v in valid_videos if v not in used_videos]
                    
                    if unused_valid_videos:
                        chosen_video = self.rng.choice(unused_valid_videos)
                    else:
                        chosen_video = self.rng.choice(valid_videos)
                        
                    tracklet = self.available_tracklets[chosen_video].pop()
                    sampled.append(tracklet)
                    
                return sampled

        pool = TrackletPool(self.dataset.video_to_tracklets, rng)
        ds = self.dataset
        batch_count = 0
        virtual_seq_counter = 0

        while batch_count < batches_to_yield:
            batch_items: list[VideoReIDItem] = []
            
            try:
                pos_tracklet_ids = pool.get_tracklets(self.num_identities_per_batch)
            except ValueError:
                break
                
            for pos_id in pos_tracklet_ids:
                pos_sequences = self._split_tracklet(rng, pos_id)
                for seq_indices in pos_sequences:
                    seq_id = virtual_seq_counter
                    virtual_seq_counter += 1
                    for idx in seq_indices:
                        item = ds.get_item(pos_id, idx, seq_id, sample_index=0)
                        batch_items.append(item)
                        
                negatives = ds.tracklets[pos_id].negative_global_ids
                if negatives:
                    pos_prompt = ds.tracklets[pos_id].prompt
                    same_prompt_negs = [n for n in negatives if ds.tracklets[n].prompt == pos_prompt]
                    diff_prompt_negs = [n for n in negatives if ds.tracklets[n].prompt != pos_prompt]
                    
                    rng.shuffle(same_prompt_negs)
                    rng.shuffle(diff_prompt_negs)
                    
                    sorted_negs = same_prompt_negs + diff_prompt_negs
                    sampled_negs = sorted_negs[:self.num_hard_negatives_per_positive]
                else:
                    sampled_negs = []
                    
                for neg_id in sampled_negs:
                    neg_sequences = self._split_tracklet(rng, neg_id)
                    for seq_indices in neg_sequences:
                        seq_id = virtual_seq_counter
                        virtual_seq_counter += 1
                        for idx in seq_indices:
                            item = ds.get_item(neg_id, idx, seq_id, sample_index=0)
                            batch_items.append(item)

            yield batch_items
            batch_count += 1

    def __len__(self) -> int:
        if self.batches_per_epoch is None:
            raise TypeError("This IterableDataset is set to infinite mode and has no length.")
        return self.batches_per_epoch
