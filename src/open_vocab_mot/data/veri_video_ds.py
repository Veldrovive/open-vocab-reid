import os
from pathlib import Path
from enum import Enum

import torch
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from tqdm import tqdm

from open_vocab_mot.data.video_reid_abc import AbstractVideoReIDDataset, VideoReIDItem, IdentityId, SequenceId

VeRiVehicleId = int
VeRiCameraId = int
VeRiFrameName = str

class VeRiSplit(Enum):
    TRAIN = 0
    QUERY = 1
    GALLERY = 2

class VeRiVideoDataset(AbstractVideoReIDDataset):
    def __init__(
        self,
        ds_root: Path | str,
        main_split: VeRiSplit,
        sidecar_root: Path | str | None = None,
        load_image_pil: bool = False,
        load_image_tensor: bool = False,
        image_tensor_dtype: torch.dtype | None = None,
        load_segmentations: bool = False,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.load_image_pil = load_image_pil
        self.load_image_tensor = load_image_tensor
        self.load_segmentations = load_segmentations
        self.image_tensor_dtype = image_tensor_dtype

        if sidecar_root is not None:
            self.sidecar_root = Path(sidecar_root)
            assert self.sidecar_root.exists(), f"VeRi Dataset sidecar root {self.sidecar_root} does not exist."
        else:
            self.sidecar_root = None

        if self.load_segmentations:
            assert self.sidecar_root is not None, "VeRi Dataset sidecar root must be specified for loading segmentations."

        self.ds_root = Path(ds_root)
        assert self.ds_root.exists(), f"VeRi Dataset root {self.ds_root} does not exist."
        
        self.train_path = self.ds_root / "image_train"
        assert self.train_path.exists(), f"VeRi Dataset train folder {self.train_path} does not exist."

        self.query_path = self.ds_root / "image_query"
        assert self.query_path.exists(), f"VeRi Dataset query folder {self.query_path} does not exist."

        self.gallery_path = self.ds_root / "image_test"
        assert self.gallery_path.exists(), f"VeRi Dataset gallery folder {self.gallery_path} does not exist."

        self.main_split = main_split
        self.split_paths: dict[VeRiSplit, Path] = {
            VeRiSplit.TRAIN: self.train_path,
            VeRiSplit.QUERY: self.query_path,
            VeRiSplit.GALLERY: self.gallery_path,
        }

        self.train_frame_map, self.train_frame_list, self.train_sequence_map = self._process_frames(VeRiSplit.TRAIN, verbose=self.verbose)
        self.query_frame_map, self.query_frame_list, self.query_sequence_map = self._process_frames(VeRiSplit.QUERY, verbose=self.verbose)
        self.gallery_frame_map, self.gallery_frame_list, self.gallery_sequence_map = self._process_frames(VeRiSplit.GALLERY, verbose=self.verbose)

    def _process_frames(self, split: VeRiSplit | None = None, verbose: bool = False) -> tuple[
        dict[VeRiVehicleId, dict[VeRiCameraId, list[VeRiFrameName]]], 
        list[tuple[VeRiVehicleId, VeRiCameraId, VeRiFrameName]],
        dict[IdentityId, dict[SequenceId, list[int]]]
    ]:
        if split is None:
            split = self.main_split

        if split == VeRiSplit.TRAIN:
            list_file = self.ds_root / "name_train.txt"
        elif split == VeRiSplit.QUERY:
            list_file = self.ds_root / "name_query.txt"
        elif split == VeRiSplit.GALLERY:
            list_file = self.ds_root / "name_test.txt"
        else:
            raise ValueError(f"Invalid split: {split}")

        assert list_file.exists(), f"List file {list_file} does not exist."

        with open(list_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if verbose:
            progress = tqdm(lines, desc=f"Loading split {split.name}")
        else:
            progress = lines

        frames_by_person: dict[VeRiVehicleId, dict[VeRiCameraId, list[VeRiFrameName]]] = {}
        frame_list: list[tuple[VeRiVehicleId, VeRiCameraId, VeRiFrameName]] = []
        sequence_map: dict[IdentityId, dict[SequenceId, list[int]]] = {}

        # First pass to group by vehicle and camera
        # Store a tuple of (frame_id, frame_name) to allow sorting
        temp_grouping: dict[VeRiVehicleId, dict[VeRiCameraId, list[tuple[int, VeRiFrameName]]]] = {}

        for frame_name in progress:
            # e.g., 0002_c002_00030600_0.jpg
            parts = frame_name.split('_')
            if len(parts) < 3:
                continue
            
            try:
                vehicle_id = int(parts[0])
                camera_id = int(parts[1][1:]) # skip 'c'
                frame_id = int(parts[2])
            except ValueError:
                continue
            
            if vehicle_id not in temp_grouping:
                temp_grouping[vehicle_id] = {}
            if camera_id not in temp_grouping[vehicle_id]:
                temp_grouping[vehicle_id][camera_id] = []
            
            temp_grouping[vehicle_id][camera_id].append((frame_id, frame_name))

        # Second pass to populate sorted frame_list and sequence_map
        for vehicle_id in sorted(temp_grouping.keys()):
            frames_by_person[vehicle_id] = {}
            sequence_map[vehicle_id] = {}
            for camera_id in sorted(temp_grouping[vehicle_id].keys()):
                # Sort frames chronologically by frame_id
                sorted_frames = sorted(temp_grouping[vehicle_id][camera_id], key=lambda x: x[0])
                frame_names = [x[1] for x in sorted_frames]
                
                frames_by_person[vehicle_id][camera_id] = frame_names
                
                start_idx = len(frame_list)
                frame_list.extend(
                    [(vehicle_id, camera_id, fname) for fname in frame_names]
                )
                end_idx = len(frame_list)
                
                sequence_map[vehicle_id][camera_id] = list(range(start_idx, end_idx))
                
        return frames_by_person, frame_list, sequence_map

    # --- AbstractVideoReIDDataset Properties ---

    @property
    def unique_identities(self) -> list[IdentityId]:
        return list(self.sequence_map.keys())

    @property
    def sequence_map(self) -> dict[IdentityId, dict[SequenceId, list[int]]]:
        if self.main_split == VeRiSplit.TRAIN:
            return self.train_sequence_map
        elif self.main_split == VeRiSplit.QUERY:
            return self.query_sequence_map
        elif self.main_split == VeRiSplit.GALLERY:
            return self.gallery_sequence_map
        else:
            raise ValueError(f"Invalid split: {self.main_split}")

    # --- Internal Properties & Loaders ---

    @property
    def frame_list(self, split: VeRiSplit | None = None) -> list[tuple[VeRiVehicleId, VeRiCameraId, VeRiFrameName]]:
        split = split or self.main_split
        if split == VeRiSplit.TRAIN:
            return self.train_frame_list
        elif split == VeRiSplit.QUERY:
            return self.query_frame_list
        elif split == VeRiSplit.GALLERY:
            return self.gallery_frame_list
        else:
            raise ValueError(f"Invalid split: {split}")

    @property
    def frame_map(self, split: VeRiSplit | None = None) -> dict[VeRiVehicleId, dict[VeRiCameraId, list[VeRiFrameName]]]:
        split = split or self.main_split
        if split == VeRiSplit.TRAIN:
            return self.train_frame_map
        elif split == VeRiSplit.QUERY:
            return self.query_frame_map
        elif split == VeRiSplit.GALLERY:
            return self.gallery_frame_map
        else:
            raise ValueError(f"Invalid split: {split}")

    def _get_frame_path(self, unique_vehicle_id: VeRiVehicleId, camera_id: VeRiCameraId, frame_name: VeRiFrameName, split: VeRiSplit | None = None) -> Path | None:
        split = split or self.main_split
        frame_path = self.split_paths[split] / frame_name
        
        if not frame_path.exists():
            print(f"Warning: Frame path {frame_path} does not exist.")
            return None
        return frame_path

    def _load_frame_pil(self, unique_vehicle_id: VeRiVehicleId, camera_id: VeRiCameraId, frame_name: VeRiFrameName, split: VeRiSplit | None = None) -> Image.Image | None:
        frame_path = self._get_frame_path(unique_vehicle_id, camera_id, frame_name, split)
        return Image.open(frame_path) if frame_path is not None else None

    def _load_frame_tensor(self, unique_vehicle_id: VeRiVehicleId, camera_id: VeRiCameraId, frame_name: VeRiFrameName, split: VeRiSplit | None = None) -> torch.Tensor | None:
        frame_path = self._get_frame_path(unique_vehicle_id, camera_id, frame_name, split)
        if frame_path is None:
            return None
        
        frame_tensor = read_image(str(frame_path), ImageReadMode.RGB)
        if self.image_tensor_dtype is not None:
            frame_tensor = frame_tensor.to(self.image_tensor_dtype) / 255.0
        else:
            frame_tensor = frame_tensor.float() / 255.0
        return frame_tensor

    def _get_segmentation_path(self, unique_vehicle_id: VeRiVehicleId, camera_id: VeRiCameraId, frame_name: VeRiFrameName, split: VeRiSplit | None = None) -> Path | None:
        frame_path = self._get_frame_path(unique_vehicle_id, camera_id, frame_name, split)
        if frame_path is None:
            return None
        
        relative_parent_path = frame_path.parent.relative_to(self.ds_root)
        segmentation_path = self.sidecar_root / relative_parent_path / f"{frame_path.stem}_major_mask.png"
        
        if not segmentation_path.exists():
            print(f"Warning: Segmentation path {segmentation_path} does not exist.")
            return None
        return segmentation_path

    def _load_segmentation_tensor(self, unique_vehicle_id: VeRiVehicleId, camera_id: VeRiCameraId, frame_name: VeRiFrameName, split: VeRiSplit | None = None) -> torch.Tensor | None:
        segmentation_path = self._get_segmentation_path(unique_vehicle_id, camera_id, frame_name, split)
        if segmentation_path is None:
            return None
        
        return read_image(str(segmentation_path), ImageReadMode.GRAY)

    def __len__(self) -> int:
        return len(self.frame_list)

    def __getitem__(self, index: int) -> VideoReIDItem:
        vehicle_id, camera_id, frame_name = self.frame_list[index]
        frame_path = self._get_frame_path(vehicle_id, camera_id, frame_name)
        
        image_pil = self._load_frame_pil(vehicle_id, camera_id, frame_name) if self.load_image_pil else None
        image_tensor = self._load_frame_tensor(vehicle_id, camera_id, frame_name) if self.load_image_tensor else None
        segmentation_path = self._get_segmentation_path(vehicle_id, camera_id, frame_name) if self.sidecar_root else None
        segmentation_tensor = self._load_segmentation_tensor(vehicle_id, camera_id, frame_name) if self.load_segmentations else None

        return VideoReIDItem(
            sample_index=index,
            identity_id=vehicle_id,
            sequence_id=camera_id,
            frame_id=frame_name,
            frame_path=frame_path,
            frame=image_pil,
            frame_tensor=image_tensor,
            segmentation_path=segmentation_path,
            segmentation_tensor=segmentation_tensor,
        )
