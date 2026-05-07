import os
from torch.utils.data import Dataset
import torch

from pathlib import Path
from enum import Enum
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from jaxtyping import Int

DukePersonId = int
DukeCameraId = int
DukeFrameName = str

class DukeSplit(Enum):
    TRAIN = 0
    QUERY = 1
    GALLERY = 2

class DukeMTMCItem(NamedTuple):
    person_id: DukePersonId
    camera_id: DukeCameraId
    frame_name: DukeFrameName
    frame_path: Path
    frame: Image.Image | None

class DukeMTMCItemBatch(NamedTuple):
    person_ids: Int[torch.Tensor, "b"]
    camera_ids: Int[torch.Tensor, "b"]
    frame_names: list[DukeFrameName]
    frame_paths: list[Path]
    frames: list[Image.Image] | list[None]

def collate_duke_mtmc_video_ds(batch: list[DukeMTMCItem]) -> DukeMTMCItemBatch:
    person_ids = torch.tensor([item.person_id for item in batch])
    camera_ids = torch.tensor([item.camera_id for item in batch])
    frame_names = [item.frame_name for item in batch]
    frame_paths = [item.frame_path for item in batch]
    frames = [item.frame for item in batch]
    return DukeMTMCItemBatch(person_ids=person_ids, camera_ids=camera_ids, frame_names=frame_names, frame_paths=frame_paths, frames=frames)

class DukeMTMCVideoDataset(Dataset):
    def __init__(self, ds_root: Path | str, main_split: DukeSplit, load_image: bool = False, verbose: bool = False):
        self.verbose = verbose
        self.load_image = load_image

        self.ds_root = Path(ds_root)
        assert self.ds_root.exists(), f"Duke MTMC Video Dataset root {self.ds_root} does not exist."
        
        self.train_path = Path(ds_root) / "train"
        assert self.train_path.exists(), f"Duke MTMC Video Dataset train folder {self.ds_root} does not exist."

        self.query_path = Path(ds_root) / "query"
        assert self.query_path.exists(), f"Duke MTMC Video Dataset query folder {self.ds_root} does not exist."

        self.gallery_path = Path(ds_root) / "gallery"
        assert self.gallery_path.exists(), f"Duke MTMC Video Dataset gallery folder {self.ds_root} does not exist."

        self.main_split = main_split
        self.split_paths: dict[DukeSplit, Path] = {
            DukeSplit.TRAIN: self.train_path,
            DukeSplit.QUERY: self.query_path,
            DukeSplit.GALLERY: self.gallery_path,
        }

        self.train_frame_map, self.train_frame_list = self._process_frames(DukeSplit.TRAIN, verbose=self.verbose)
        self.query_frame_map, self.query_frame_list = self._process_frames(DukeSplit.QUERY, verbose=self.verbose)
        self.gallery_frame_map, self.gallery_frame_list = self._process_frames(DukeSplit.GALLERY, verbose=self.verbose)

    def _process_frames(self, split: DukeSplit | None = None, verbose: bool = False) -> tuple[dict[DukePersonId, dict[DukeCameraId, list[DukeFrameName]]], list[tuple[DukePersonId, DukeCameraId, DukeFrameName]]]:
        if split is None:
            split = self.main_split

        if verbose:
            progress = tqdm(desc=f"Loading split {split.name}")

        split_folder = self.split_paths[split]
        frames_by_person: dict[DukePersonId, dict[DukeCameraId, list[DukeFrameName]]] = {}
        frame_list: list[tuple[DukePersonId, DukeCameraId, DukeFrameName]] = []
        
        with os.scandir(split_folder) as subject_it:
            for subject_folder in subject_it:
                if not subject_folder.is_dir():
                    continue
                person_id = int(subject_folder.name)
                frames_by_person[person_id] = {}
                
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
                                    
                        frames_by_person[person_id][camera_id] = frames
                        frame_list.extend(
                            [(person_id, camera_id, frame_name) for frame_name in frames]
                        )

                        if verbose:
                            progress.update(len(frames))
                            
        return frames_by_person, frame_list

    @property
    def frame_list(self, split: DukeSplit | None = None) -> list[tuple[DukePersonId, DukeCameraId, DukeFrameName]]:
        if split is None:
            split = self.main_split
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
        if split is None:
            split = self.main_split
        if split == DukeSplit.TRAIN:
            return self.train_frame_map
        elif split == DukeSplit.QUERY:
            return self.query_frame_map
        elif split == DukeSplit.GALLERY:
            return self.gallery_frame_map
        else:
            raise ValueError(f"Invalid split: {split}")

    def _get_frame_path(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> Path | None:
        if split is None:
            split = self.main_split

        person_folder_name = f"{unique_person_id:04d}"
        camera_folder_name = f"{camera_id:04d}"
        frame_name = frame_name

        person_folder = self.split_paths[split] / person_folder_name
        camera_folder = person_folder / camera_folder_name
        frame_path = camera_folder / frame_name
        if not frame_path.exists():
            print(f"Warning: Frame path {frame_path} does not exist.")
            return None
        return frame_path

    def _load_frame_pil(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> Image.Image | None:
        if split is None:
            split = self.main_split

        frame_path = self._get_frame_path(unique_person_id, camera_id, frame_name, split)
        if frame_path is None:
            return None
        return Image.open(frame_path)

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index: int) -> DukeMTMCItem:
        person_id, camera_id, frame_name = self.frame_list[index]
        frame_path = self._get_frame_path(person_id, camera_id, frame_name)
        if self.load_image:
            image = self._load_frame_pil(person_id, camera_id, frame_name)
        else:
            image = None
        return DukeMTMCItem(person_id=person_id, camera_id=camera_id, frame_name=frame_name, frame_path=frame_path, frame=image)



