import os
import pickle
from pathlib import Path
from enum import Enum

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from typing import Callable, Any
from PIL import Image
from tqdm import tqdm

from open_vocab_mot.data.video_reid_abc import AbstractVideoReIDDataset, VideoReIDItem, IdentityId, SequenceId, DatasetSplit

VRAIPersonId = int
VRAICameraId = int
VRAIFrameName = str



class VRAITestSet(Enum):
    DEV = 0
    FULL = 1

class VRAIDataset(AbstractVideoReIDDataset):
    def __init__(
        self,
        ds_root: Path | str,
        main_split: DatasetSplit,
        test_set: VRAITestSet = VRAITestSet.FULL,
        sidecar_root: Path | str | None = None,
        load_image_pil: bool = False,
        load_image_tensor: bool = False,
        image_tensor_dtype: torch.dtype | None = None,
        load_segmentations: bool = False,
        verbose: bool = False,
        transform: Callable | None = None,
    ):
        self.verbose = verbose
        self.load_image_pil = load_image_pil
        self.load_image_tensor = load_image_tensor
        self.load_segmentations = load_segmentations
        self.image_tensor_dtype = image_tensor_dtype
        self.transform = transform

        if sidecar_root is not None:
            self.sidecar_root = Path(sidecar_root)
            assert self.sidecar_root.exists(), f"VRAI Dataset sidecar root {self.sidecar_root} does not exist."
        else:
            self.sidecar_root = None

        if self.load_segmentations:
            assert self.sidecar_root is not None, "VRAI Dataset sidecar root must be specified for loading segmentations."

        self.ds_root = Path(ds_root)
        assert self.ds_root.exists(), f"VRAI Dataset root {self.ds_root} does not exist."
        
        self.images_train_path = self.ds_root / "images_train"
        assert self.images_train_path.exists(), f"VRAI Dataset train folder {self.images_train_path} does not exist. Please unpack images_train.tar."

        self.images_dev_path = self.ds_root / "images_dev"
        assert self.images_dev_path.exists(), f"VRAI Dataset dev folder {self.images_dev_path} does not exist. Please unpack images_dev.tar."

        self.main_split = main_split
        self.test_set = test_set

        # Determine string to integer ID mapping for the requested test set
        test_pkl_name = "test_dev_annotation.pkl" if test_set == VRAITestSet.DEV else "test_annotation.pkl"
        test_pkl_path = self.ds_root / test_pkl_name
        assert test_pkl_path.exists(), f"Test annotation file {test_pkl_path} does not exist."
        
        with open(test_pkl_path, 'rb') as f:
            self.test_annotations = pickle.load(f)
            
        unique_test_str_ids = set()
        for frame_name in self.test_annotations.keys():
            person_id_str = frame_name.split('_')[0]
            unique_test_str_ids.add(person_id_str)
            
        self.test_str_to_int_id = {
            str_id: 1000000 + idx 
            for idx, str_id in enumerate(sorted(list(unique_test_str_ids)))
        }

        self.train_frame_map, self.train_frame_list, self.train_sequence_map = self._process_train_frames(verbose=self.verbose)
        self.query_frame_map, self.query_frame_list, self.query_sequence_map = self._process_test_frames(DatasetSplit.QUERY, self.test_annotations, verbose=self.verbose)
        self.gallery_frame_map, self.gallery_frame_list, self.gallery_sequence_map = self._process_test_frames(DatasetSplit.GALLERY, self.test_annotations, verbose=self.verbose)

    def _process_train_frames(self, verbose: bool = False) -> tuple[
        dict[VRAIPersonId, dict[VRAICameraId, list[VRAIFrameName]]], 
        list[tuple[VRAIPersonId, VRAICameraId, VRAIFrameName]],
        dict[IdentityId, dict[SequenceId, list[int]]]
    ]:
        frames_by_person: dict[VRAIPersonId, dict[VRAICameraId, list[VRAIFrameName]]] = {}
        frame_list: list[tuple[VRAIPersonId, VRAICameraId, VRAIFrameName]] = []
        sequence_map: dict[IdentityId, dict[SequenceId, list[int]]] = {}
        
        train_ann_path = self.ds_root / "train_annotation.pkl"
        with open(train_ann_path, 'rb') as f:
            train_ann = pickle.load(f)
            train_im_names = train_ann['train_im_names']
            
        grouped_frames = {}
        for frame_name in train_im_names:
            parts = frame_name.split('_')
            person_id = int(parts[0])
            camera_id = int(parts[1])
            
            if person_id not in grouped_frames:
                grouped_frames[person_id] = {}
            if camera_id not in grouped_frames[person_id]:
                grouped_frames[person_id][camera_id] = []
            grouped_frames[person_id][camera_id].append(frame_name)
            
        if verbose:
            progress = tqdm(total=len(train_im_names), desc="Loading split TRAIN")
            
        for person_id, cameras in grouped_frames.items():
            frames_by_person[person_id] = {}
            sequence_map[person_id] = {}
            for camera_id, frames in cameras.items():
                frames.sort()
                frames_by_person[person_id][camera_id] = frames
                
                start_idx = len(frame_list)
                frame_list.extend([(person_id, camera_id, f) for f in frames])
                end_idx = len(frame_list)
                sequence_map[person_id][camera_id] = list(range(start_idx, end_idx))
                
                if verbose:
                    progress.update(len(frames))
                    
        return frames_by_person, frame_list, sequence_map

    def _process_test_frames(self, split: DatasetSplit, annotations: dict[str, int], verbose: bool = False) -> tuple[
        dict[VRAIPersonId, dict[VRAICameraId, list[VRAIFrameName]]], 
        list[tuple[VRAIPersonId, VRAICameraId, VRAIFrameName]],
        dict[IdentityId, dict[SequenceId, list[int]]]
    ]:
        frames_by_person: dict[VRAIPersonId, dict[VRAICameraId, list[VRAIFrameName]]] = {}
        frame_list: list[tuple[VRAIPersonId, VRAICameraId, VRAIFrameName]] = []
        sequence_map: dict[IdentityId, dict[SequenceId, list[int]]] = {}
        
        # Query == 1, Gallery == 0
        target_val = 1 if split == DatasetSplit.QUERY else 0
        
        grouped_frames = {}
        for frame_name, is_query in annotations.items():
            if is_query != target_val:
                continue
                
            person_id_str, cam_ext = frame_name.split('_')
            person_id = self.test_str_to_int_id[person_id_str]
            camera_id = int(cam_ext.split('.')[0].replace('C', ''))
            
            if person_id not in grouped_frames:
                grouped_frames[person_id] = {}
            if camera_id not in grouped_frames[person_id]:
                grouped_frames[person_id][camera_id] = []
            grouped_frames[person_id][camera_id].append(frame_name)
            
        total_frames = sum(len(frames) for cameras in grouped_frames.values() for frames in cameras.values())
        if verbose:
            progress = tqdm(total=total_frames, desc=f"Loading split {split.name}")
            
        for person_id, cameras in grouped_frames.items():
            frames_by_person[person_id] = {}
            sequence_map[person_id] = {}
            for camera_id, frames in cameras.items():
                frames.sort()
                frames_by_person[person_id][camera_id] = frames
                
                start_idx = len(frame_list)
                frame_list.extend([(person_id, camera_id, f) for f in frames])
                end_idx = len(frame_list)
                sequence_map[person_id][camera_id] = list(range(start_idx, end_idx))
                
                if verbose:
                    progress.update(len(frames))
                    
        return frames_by_person, frame_list, sequence_map

    # --- AbstractVideoReIDDataset Properties ---

    @property
    def unique_identities(self) -> list[IdentityId]:
        return list(self.sequence_map.keys())

    @property
    def sequence_map(self) -> dict[IdentityId, dict[SequenceId, list[int]]]:
        if self.main_split == DatasetSplit.TRAIN:
            return self.train_sequence_map
        elif self.main_split == DatasetSplit.QUERY:
            return self.query_sequence_map
        elif self.main_split == DatasetSplit.GALLERY:
            return self.gallery_sequence_map
        else:
            raise ValueError(f"Invalid split: {self.main_split}")

    # --- Internal Properties & Loaders ---

    @property
    def frame_list(self, split: DatasetSplit | None = None) -> list[tuple[VRAIPersonId, VRAICameraId, VRAIFrameName]]:
        split = split or self.main_split
        if split == DatasetSplit.TRAIN:
            return self.train_frame_list
        elif split == DatasetSplit.QUERY:
            return self.query_frame_list
        elif split == DatasetSplit.GALLERY:
            return self.gallery_frame_list
        else:
            raise ValueError(f"Invalid split: {split}")

    @property
    def frame_map(self, split: DatasetSplit | None = None) -> dict[VRAIPersonId, dict[VRAICameraId, list[VRAIFrameName]]]:
        split = split or self.main_split
        if split == DatasetSplit.TRAIN:
            return self.train_frame_map
        elif split == DatasetSplit.QUERY:
            return self.query_frame_map
        elif split == DatasetSplit.GALLERY:
            return self.gallery_frame_map
        else:
            raise ValueError(f"Invalid split: {split}")

    def _get_frame_path(self, unique_person_id: VRAIPersonId, camera_id: VRAICameraId, frame_name: VRAIFrameName, split: DatasetSplit | None = None) -> Path | None:
        split = split or self.main_split
        if split == DatasetSplit.TRAIN:
            frame_path = self.images_train_path / frame_name
        else:
            frame_path = self.images_dev_path / frame_name
        
        if not frame_path.exists():
            print(f"Warning: Frame path {frame_path} does not exist.")
            return None
        return frame_path

    def _load_frame_pil(self, unique_person_id: VRAIPersonId, camera_id: VRAICameraId, frame_name: VRAIFrameName, split: DatasetSplit | None = None) -> Image.Image | None:
        frame_path = self._get_frame_path(unique_person_id, camera_id, frame_name, split)
        return Image.open(frame_path) if frame_path is not None else None

    def _load_frame_tensor(self, unique_person_id: VRAIPersonId, camera_id: VRAICameraId, frame_name: VRAIFrameName, split: DatasetSplit | None = None) -> torch.Tensor | None:
        frame_path = self._get_frame_path(unique_person_id, camera_id, frame_name, split)
        if frame_path is None:
            return None
        
        frame_tensor = read_image(str(frame_path), ImageReadMode.RGB)
        if self.image_tensor_dtype is not None:
            frame_tensor = frame_tensor.to(self.image_tensor_dtype) / 255.0
        else:
            frame_tensor = frame_tensor.float() / 255.0
        return frame_tensor

    def _get_segmentation_path(self, unique_person_id: VRAIPersonId, camera_id: VRAICameraId, frame_name: VRAIFrameName, split: DatasetSplit | None = None) -> Path | None:
        frame_path = self._get_frame_path(unique_person_id, camera_id, frame_name, split)
        if frame_path is None or self.sidecar_root is None:
            return None
        
        relative_parent_path = Path("images_train") if (split or self.main_split) == DatasetSplit.TRAIN else Path("images_dev")
        segmentation_path = self.sidecar_root / relative_parent_path / f"{frame_path.stem}_major_mask.png"
        
        if not segmentation_path.exists():
            print(f"Warning: Segmentation path {segmentation_path} does not exist.")
            return None
        return segmentation_path

    def _load_segmentation_tensor(self, unique_person_id: VRAIPersonId, camera_id: VRAICameraId, frame_name: VRAIFrameName, split: DatasetSplit | None = None) -> torch.Tensor | None:
        segmentation_path = self._get_segmentation_path(unique_person_id, camera_id, frame_name, split)
        if segmentation_path is None:
            return None
        
        return read_image(str(segmentation_path), ImageReadMode.GRAY)

    def __len__(self) -> int:
        return len(self.frame_list)

    def __getitem__(self, index: int) -> VideoReIDItem:
        person_id, camera_id, frame_name = self.frame_list[index]
        frame_path = self._get_frame_path(person_id, camera_id, frame_name)
        
        image_pil = self._load_frame_pil(person_id, camera_id, frame_name) if self.load_image_pil else None
        image_tensor = self._load_frame_tensor(person_id, camera_id, frame_name) if self.load_image_tensor else None
        segmentation_path = self._get_segmentation_path(person_id, camera_id, frame_name) if self.sidecar_root else None
        segmentation_tensor = self._load_segmentation_tensor(person_id, camera_id, frame_name) if self.load_segmentations else None

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
            sample_index=index,
            identity_id=person_id,
            sequence_id=camera_id,
            frame_id=frame_name,
            frame_path=frame_path,
            frame=image_pil,
            frame_tensor=image_tensor,
            segmentation_path=segmentation_path,
            segmentation_tensor=segmentation_tensor,
        )
