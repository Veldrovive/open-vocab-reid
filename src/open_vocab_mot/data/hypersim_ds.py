from pathlib import Path
from typing import Iterator, NamedTuple
import json
import random
import torch
import h5py
import numpy as np
from PIL import Image
from pydantic import BaseModel
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms.functional import crop

from open_vocab_mot.data.video_reid_abc import VideoReIDItem, DatasetSplit, AbstractVideoReIDDataset

class HypersimObjectData(BaseModel):
    semantic_label: int
    asset_id: str = ""
    cameras: dict[str, list[str]]

class HypersimSceneData(BaseModel):
    objects: dict[str, HypersimObjectData]

class HypersimSidecarData(BaseModel):
    scenes: dict[str, HypersimSceneData]

class GlobalIdentityInfo(NamedTuple):
    global_id: int
    scene_name: str
    original_obj_id: str
    semantic_label: int
    asset_id: str
    cameras: dict[str, list[Path]]

class HypersimVideoReIDDataset(AbstractVideoReIDDataset):
    def __init__(
        self,
        ds_root: Path | str,
        sidecar_path: Path | str,
        split: DatasetSplit = DatasetSplit.TRAIN,
        val_split_ratio: float = 0.1,
        split_seed: int = 42,
        min_frames_per_sequence: int = 4,
        load_image_pil: bool = False,
        load_image_tensor: bool = False,
        image_tensor_dtype: torch.dtype | None = None,
        load_segmentations: bool = False,
        crop_to_object: bool = False,
        crop_padding: int = 15,
        apply_mask: bool = False,
        transform=None,
    ):
        self.ds_root = Path(ds_root)
        self.sidecar_path = Path(sidecar_path)
        self.split = split
        self.val_split_ratio = val_split_ratio
        self.split_seed = split_seed
        self.min_frames_per_sequence = min_frames_per_sequence
        
        self.load_image_pil = load_image_pil
        self.load_image_tensor = load_image_tensor
        self.image_tensor_dtype = image_tensor_dtype
        self.load_segmentations = load_segmentations
        self.crop_to_object = crop_to_object
        self.crop_padding = crop_padding
        self.apply_mask = apply_mask
        self.transform = transform

        with open(self.sidecar_path, 'r') as f:
            sidecar_dict = json.load(f)
            
        self.sidecar = HypersimSidecarData(**sidecar_dict)
        
        self.global_identities: dict[int, GlobalIdentityInfo] = {}
        self.scene_to_global_ids: dict[str, list[int]] = {}
        self.flat_frames: list[tuple[int, str, int, int]] = [] # global_id, cam_name, frame_idx, seq_id
        self._unique_identities: list[int] = []
        self._sequence_map: dict[int, dict[int, list[int]]] = {}
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        global_id_counter = 0
        
        all_scenes = sorted(list(self.sidecar.scenes.keys()))
        rng = random.Random(self.split_seed)
        rng.shuffle(all_scenes)
        
        num_val = int(len(all_scenes) * self.val_split_ratio)
        val_scenes = set(all_scenes[:num_val])
        train_scenes = set(all_scenes[num_val:])
        
        target_scenes = train_scenes if self.split == DatasetSplit.TRAIN else val_scenes
        
        for scene_name in target_scenes:
            scene_data = self.sidecar.scenes[scene_name]
            valid_ids_for_scene = []
            
            for obj_id_str, obj_data in scene_data.objects.items():
                cameras_with_paths = {}
                for cam_name, frames in obj_data.cameras.items():
                    if len(frames) < self.min_frames_per_sequence:
                        continue
                        
                    frame_paths = []
                    for frame_name in frames:
                        # Use final_preview color.jpg for RGB image
                        img_path = self.ds_root / scene_name / "images" / f"scene_{cam_name}_final_preview" / f"{frame_name}.color.jpg"
                        frame_paths.append(img_path)
                        
                    if len(frame_paths) >= self.min_frames_per_sequence:
                        cameras_with_paths[cam_name] = frame_paths
                        
                if len(cameras_with_paths) > 0:
                    global_id = global_id_counter
                    global_id_counter += 1
                    
                    self.global_identities[global_id] = GlobalIdentityInfo(
                        global_id=global_id,
                        scene_name=scene_name,
                        original_obj_id=obj_id_str,
                        semantic_label=obj_data.semantic_label,
                        asset_id=obj_data.asset_id or obj_id_str,
                        cameras=cameras_with_paths
                    )
                    valid_ids_for_scene.append(global_id)
                    self._unique_identities.append(global_id)
                    self._sequence_map[global_id] = {}
                    
                    seq_id_counter = 0
                    for cam_name, frame_paths in cameras_with_paths.items():
                        seq_id = seq_id_counter
                        seq_id_counter += 1
                        self._sequence_map[global_id][seq_id] = []
                        for idx in range(len(frame_paths)):
                            sample_index = len(self.flat_frames)
                            self._sequence_map[global_id][seq_id].append(sample_index)
                            self.flat_frames.append((global_id, cam_name, idx, seq_id))
                            
            if valid_ids_for_scene:
                self.scene_to_global_ids[scene_name] = valid_ids_for_scene

        print(f"Loaded Hypersim Video ReID Dataset: {len(self.global_identities)} identities across {len(self.scene_to_global_ids)} scenes.")

    @property
    def unique_identities(self) -> list[int]:
        return self._unique_identities

    @property
    def sequence_map(self) -> dict[int, dict[int, list[int]]]:
        return self._sequence_map

    def __len__(self) -> int:
        return len(self.flat_frames)

    def __getitem__(self, index: int) -> VideoReIDItem:
        global_id, cam_name, frame_idx, sequence_id = self.flat_frames[index]
        return self.get_item(global_id, cam_name, frame_idx, sequence_id, index)

    def get_item(self, global_id: int, cam_name: str, frame_idx: int, sequence_id: int, sample_index: int) -> VideoReIDItem:
        identity = self.global_identities[global_id]
        frame_path = identity.cameras[cam_name][frame_idx]
        
        # Original frame name e.g. "frame.0000" extracted from frame_path
        # frame_path looks like /z/dat/hypersim/data/frames/ai_001_001/images/scene_cam_00_final_preview/frame.0000.color.jpg
        # We need the semantic_instance.hdf5
        frame_base = frame_path.name.split(".color.")[0] # "frame.0000"
        
        mask_path = self.ds_root / identity.scene_name / "images" / f"scene_{cam_name}_geometry_hdf5" / f"{frame_base}.semantic_instance.hdf5"
        
        # Load object mask if needed for cropping or applying mask or returning segmentations
        needs_mask = self.crop_to_object or self.apply_mask or self.load_segmentations
        mask_np = None
        obj_mask = None
        bounding_box = None # (ymin, xmin, ymax, xmax)
        
        if needs_mask and mask_path.exists():
            with h5py.File(mask_path, "r") as f:
                mask_np = f["dataset"][:]
                obj_mask = (mask_np == int(identity.original_obj_id))
                
                if self.crop_to_object and np.any(obj_mask):
                    rows = np.any(obj_mask, axis=1)
                    cols = np.any(obj_mask, axis=0)
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]
                    
                    ymin = max(0, ymin - self.crop_padding)
                    ymax = min(mask_np.shape[0], ymax + self.crop_padding + 1)
                    xmin = max(0, xmin - self.crop_padding)
                    xmax = min(mask_np.shape[1], xmax + self.crop_padding + 1)
                    
                    bounding_box = (ymin, xmin, ymax, xmax)
                    
        image_pil = None
        if self.load_image_pil:
            image_pil = Image.open(frame_path).convert("RGB")
            
            if self.apply_mask and obj_mask is not None:
                img_np = np.array(image_pil)
                img_np[~obj_mask] = 0
                image_pil = Image.fromarray(img_np)
                
            if self.crop_to_object and bounding_box is not None:
                ymin, xmin, ymax, xmax = bounding_box
                image_pil = image_pil.crop((xmin, ymin, xmax, ymax))
            
        image_tensor = None
        if self.load_image_tensor:
            image_tensor = read_image(str(frame_path), ImageReadMode.RGB)
            
            if self.apply_mask and obj_mask is not None:
                mask_tensor = torch.from_numpy(obj_mask).unsqueeze(0).to(image_tensor.device)
                image_tensor = image_tensor * mask_tensor
                
            if self.crop_to_object and bounding_box is not None:
                ymin, xmin, ymax, xmax = bounding_box
                image_tensor = crop(image_tensor, ymin, xmin, ymax - ymin, xmax - xmin)
                
            if self.image_tensor_dtype is not None:
                image_tensor = image_tensor.to(self.image_tensor_dtype) / 255.0
            else:
                image_tensor = image_tensor.float() / 255.0
                
        segmentation_tensor = None
        if self.load_segmentations and obj_mask is not None:
            segmentation_tensor = torch.from_numpy(obj_mask).bool()
            if self.crop_to_object and bounding_box is not None:
                ymin, xmin, ymax, xmax = bounding_box
                segmentation_tensor = segmentation_tensor[ymin:ymax, xmin:xmax]
        
        if self.transform is not None:
            if image_tensor is not None:
                image_tensor = tv_tensors.Image(image_tensor)
                image_tensor = self.transform(image_tensor)

        return VideoReIDItem(
            sample_index=sample_index,
            identity_id=global_id,
            sequence_id=sequence_id,
            frame_id=f"{cam_name}_{frame_path.name}",
            frame_path=frame_path,
            frame=image_pil,
            frame_tensor=image_tensor,
            segmentation_path=mask_path,
            segmentation_tensor=segmentation_tensor,
        )

class HypersimBatchIterableDataset(IterableDataset[list[VideoReIDItem]]):
    def __init__(
        self,
        dataset: HypersimVideoReIDDataset,
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

    def _split_tracklet(self, rng: random.Random, global_id: int) -> list[tuple[str, list[int]]]:
        identity = self.dataset.global_identities[global_id]
        
        sequences = []
        # Bias towards picking different cameras for hard positives
        cam_names = list(identity.cameras.keys())
        rng.shuffle(cam_names)
        
        cameras_to_use = cam_names * (self.num_sequences_per_tracklet // len(cam_names))
        cameras_to_use += cam_names[:self.num_sequences_per_tracklet % len(cam_names)]
        
        for cam_name in cameras_to_use:
            num_frames = len(identity.cameras[cam_name])
            available_indices = list(range(num_frames))
            
            # Subsample frames randomly
            if len(available_indices) >= self.num_frames_per_sequence:
                sampled_indices = sorted(rng.sample(available_indices, self.num_frames_per_sequence))
            else:
                sampled_indices = sorted(rng.choices(available_indices, k=self.num_frames_per_sequence))
                
            sequences.append((cam_name, sampled_indices))
            
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

        class ScenePool:
            def __init__(self, scene_to_global_ids, r):
                self.scene_to_global_ids = {k: list(v) for k, v in scene_to_global_ids.items() if len(v) > 0}
                self.rng = r
                self.available_scenes = list(self.scene_to_global_ids.keys())
                self.rng.shuffle(self.available_scenes)
                
            def get_scene(self) -> str:
                if not self.available_scenes:
                    self.available_scenes = list(self.scene_to_global_ids.keys())
                    self.rng.shuffle(self.available_scenes)
                return self.available_scenes.pop()

        pool = ScenePool(self.dataset.scene_to_global_ids, rng)
        ds = self.dataset
        batch_count = 0
        virtual_seq_counter = 0

        while batch_count < batches_to_yield:
            batch_items: list[VideoReIDItem] = []
            
            # Select a scene
            scene_name = pool.get_scene()
            scene_identities = ds.scene_to_global_ids[scene_name]
            
            if len(scene_identities) < self.num_identities_per_batch:
                # Fallback if scene doesn't have enough identities
                pos_tracklet_ids = scene_identities
            else:
                pos_tracklet_ids = rng.sample(scene_identities, self.num_identities_per_batch)
                
            for pos_id in pos_tracklet_ids:
                pos_sequences = self._split_tracklet(rng, pos_id)
                for cam_name, seq_indices in pos_sequences:
                    seq_id = virtual_seq_counter
                    virtual_seq_counter += 1
                    for idx in seq_indices:
                        item = ds.get_item(pos_id, cam_name, idx, seq_id, sample_index=0)
                        batch_items.append(item)
                        
                pos_identity = ds.global_identities[pos_id]
                pos_label = pos_identity.semantic_label
                
                # Find negatives
                pos_asset_id = pos_identity.asset_id
                other_ids_in_scene = [
                    i for i in scene_identities 
                    if i != pos_id and ds.global_identities[i].asset_id != pos_asset_id
                ]
                same_label_negs = [i for i in other_ids_in_scene if ds.global_identities[i].semantic_label == pos_label]
                diff_label_negs = [i for i in other_ids_in_scene if ds.global_identities[i].semantic_label != pos_label]
                
                rng.shuffle(same_label_negs)
                rng.shuffle(diff_label_negs)
                
                # Bias towards hard negatives (same semantic label)
                sorted_negs = same_label_negs + diff_label_negs
                sampled_negs = sorted_negs[:self.num_hard_negatives_per_positive]
                    
                for neg_id in sampled_negs:
                    neg_sequences = self._split_tracklet(rng, neg_id)
                    for cam_name, seq_indices in neg_sequences:
                        seq_id = virtual_seq_counter
                        virtual_seq_counter += 1
                        for idx in seq_indices:
                            item = ds.get_item(neg_id, cam_name, idx, seq_id, sample_index=0)
                            batch_items.append(item)

            if len(batch_items) > 0:
                yield batch_items
                batch_count += 1

    def __len__(self) -> int:
        if self.batches_per_epoch is None:
            raise TypeError("This IterableDataset is set to infinite mode and has no length.")
        return self.batches_per_epoch

