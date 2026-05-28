import os
from pathlib import Path
from enum import Enum

import torch
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from tqdm import tqdm

from open_vocab_mot.data.video_reid_abc import AbstractVideoReIDDataset, VideoReIDItem, IdentityId, SequenceId

DukePersonId = int
DukeCameraId = int
DukeFrameName = str

class DukeSplit(Enum):
    TRAIN = 0
    QUERY = 1
    GALLERY = 2

class DukeMTMCVideoDataset(AbstractVideoReIDDataset):
    def __init__(
        self,
        ds_root: Path | str,
        main_split: DukeSplit,
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
            assert self.sidecar_root.exists(), f"Duke MTMC Video Dataset sidecar root {self.sidecar_root} does not exist."
        else:
            self.sidecar_root = None

        if self.load_segmentations:
            assert self.sidecar_root is not None, "Duke MTMC Video Dataset sidecar root must be specified for loading segmentations."

        self.ds_root = Path(ds_root)
        assert self.ds_root.exists(), f"Duke MTMC Video Dataset root {self.ds_root} does not exist."
        
        self.train_path = self.ds_root / "train"
        assert self.train_path.exists(), f"Duke MTMC Video Dataset train folder {self.train_path} does not exist."

        self.query_path = self.ds_root / "query"
        assert self.query_path.exists(), f"Duke MTMC Video Dataset query folder {self.query_path} does not exist."

        self.gallery_path = self.ds_root / "gallery"
        assert self.gallery_path.exists(), f"Duke MTMC Video Dataset gallery folder {self.gallery_path} does not exist."

        self.main_split = main_split
        self.split_paths: dict[DukeSplit, Path] = {
            DukeSplit.TRAIN: self.train_path,
            DukeSplit.QUERY: self.query_path,
            DukeSplit.GALLERY: self.gallery_path,
        }

        self.train_frame_map, self.train_frame_list, self.train_sequence_map = self._process_frames(DukeSplit.TRAIN, verbose=self.verbose)
        self.query_frame_map, self.query_frame_list, self.query_sequence_map = self._process_frames(DukeSplit.QUERY, verbose=self.verbose)
        self.gallery_frame_map, self.gallery_frame_list, self.gallery_sequence_map = self._process_frames(DukeSplit.GALLERY, verbose=self.verbose)

    def _process_frames(self, split: DukeSplit | None = None, verbose: bool = False) -> tuple[
        dict[DukePersonId, dict[DukeCameraId, list[DukeFrameName]]], 
        list[tuple[DukePersonId, DukeCameraId, DukeFrameName]],
        dict[IdentityId, dict[SequenceId, list[int]]]
    ]:
        if split is None:
            split = self.main_split

        if verbose:
            progress = tqdm(desc=f"Loading split {split.name}")

        split_folder = self.split_paths[split]
        frames_by_person: dict[DukePersonId, dict[DukeCameraId, list[DukeFrameName]]] = {}
        frame_list: list[tuple[DukePersonId, DukeCameraId, DukeFrameName]] = []
        sequence_map: dict[IdentityId, dict[SequenceId, list[int]]] = {}
        
        with os.scandir(split_folder) as subject_it:
            for subject_folder in subject_it:
                if not subject_folder.is_dir():
                    continue
                person_id = int(subject_folder.name)
                frames_by_person[person_id] = {}
                sequence_map[person_id] = {}
                
                with os.scandir(subject_folder.path) as camera_it:
                    for camera_folder in camera_it:
                        if not camera_folder.is_dir():
                            continue
                        camera_id = int(camera_folder.name)
                        
                        frames = []
                        with os.scandir(camera_folder.path) as frame_it:
                            for frame in frame_it:
                                if frame.is_file():
                                    frames.append(frame.name)
                        
                        # Generate flat indices mapping for the Abstract Dataset property
                        start_idx = len(frame_list)
                        frame_list.extend(
                            [(person_id, camera_id, frame_name) for frame_name in frames]
                        )
                        end_idx = len(frame_list)
                        
                        frames_by_person[person_id][camera_id] = frames
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
        if self.main_split == DukeSplit.TRAIN:
            return self.train_sequence_map
        elif self.main_split == DukeSplit.QUERY:
            return self.query_sequence_map
        elif self.main_split == DukeSplit.GALLERY:
            return self.gallery_sequence_map
        else:
            raise ValueError(f"Invalid split: {self.main_split}")

    # --- Internal Properties & Loaders ---

    @property
    def frame_list(self, split: DukeSplit | None = None) -> list[tuple[DukePersonId, DukeCameraId, DukeFrameName]]:
        split = split or self.main_split
        if split == DukeSplit.TRAIN:
            return self.train_frame_list
        elif split == DukeSplit.QUERY:
            return self.query_frame_list
        elif split == DukeSplit.GALLERY:
            return self.gallery_frame_list
        else:
            raise ValueError(f"Invalid split: {split}")

    @property
    def frame_map(self, split: DukeSplit | None = None) -> dict[DukePersonId, dict[DukeCameraId, list[DukeFrameName]]]:
        split = split or self.main_split
        if split == DukeSplit.TRAIN:
            return self.train_frame_map
        elif split == DukeSplit.QUERY:
            return self.query_frame_map
        elif split == DukeSplit.GALLERY:
            return self.gallery_frame_map
        else:
            raise ValueError(f"Invalid split: {split}")

    def _get_frame_path(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> Path | None:
        split = split or self.main_split
        person_folder_name = f"{unique_person_id:04d}"
        camera_folder_name = f"{camera_id:04d}"

        person_folder = self.split_paths[split] / person_folder_name
        camera_folder = person_folder / camera_folder_name
        frame_path = camera_folder / frame_name
        
        if not frame_path.exists():
            print(f"Warning: Frame path {frame_path} does not exist.")
            return None
        return frame_path

    def _load_frame_pil(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> Image.Image | None:
        frame_path = self._get_frame_path(unique_person_id, camera_id, frame_name, split)
        return Image.open(frame_path) if frame_path is not None else None

    def _load_frame_tensor(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> torch.Tensor | None:
        frame_path = self._get_frame_path(unique_person_id, camera_id, frame_name, split)
        if frame_path is None:
            return None
        
        frame_tensor = read_image(str(frame_path), ImageReadMode.RGB)
        if self.image_tensor_dtype is not None:
            frame_tensor = frame_tensor.to(self.image_tensor_dtype) / 255.0
        else:
            frame_tensor = frame_tensor.float() / 255.0
        return frame_tensor

    def _get_segmentation_path(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> Path | None:
        frame_path = self._get_frame_path(unique_person_id, camera_id, frame_name, split)
        if frame_path is None:
            return None
        
        relative_parent_path = frame_path.parent.relative_to(self.ds_root)
        segmentation_path = self.sidecar_root / relative_parent_path / f"{frame_path.stem}_major_mask.png"
        
        if not segmentation_path.exists():
            print(f"Warning: Segmentation path {segmentation_path} does not exist.")
            return None
        return segmentation_path

    def _load_segmentation_tensor(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> torch.Tensor | None:
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