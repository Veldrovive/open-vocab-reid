import os
from torch.utils.data import Dataset, Sampler, default_collate
from torchvision.io import read_image, ImageReadMode
import torch

from pathlib import Path
from enum import Enum
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from jaxtyping import Int
import random

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
    frame_tensor: torch.Tensor | None

    segmentation_path: Path | None
    segmentation_tensor: torch.Tensor | None

class DukeMTMCItemBatch(NamedTuple):
    person_ids: Int[torch.Tensor, "b"]
    camera_ids: Int[torch.Tensor, "b"]
    frame_names: list[DukeFrameName]
    frame_paths: list[Path]
    frames: list[Image.Image] | list[None]
    frame_tensors: list[torch.Tensor] | list[None]

    segmentation_paths: list[Path] | list[None]
    segmentations: list[torch.Tensor] | list[None]

def collate_duke_mtmc_video_ds(batch: list[DukeMTMCItem]) -> DukeMTMCItemBatch:
    person_ids = torch.tensor([item.person_id for item in batch])
    camera_ids = torch.tensor([item.camera_id for item in batch])
    frame_names = [item.frame_name for item in batch]
    frame_paths = [item.frame_path for item in batch]
    frames = [item.frame for item in batch]
    frame_tensors = [item.frame_tensor for item in batch]

    segmentation_paths = [item.segmentation_path for item in batch]
    segmentations = [item.segmentation_tensor for item in batch]
    return DukeMTMCItemBatch(person_ids=person_ids, camera_ids=camera_ids, frame_names=frame_names, frame_paths=frame_paths, frames=frames, frame_tensors=frame_tensors, segmentation_paths=segmentation_paths, segmentations=segmentations)

class DukeMTMCVideoDataset(Dataset):
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
            assert self.sidecar_root is not None, f"Duke MTMC Video Dataset sidecar root must be specified for loading segmentations."

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

    def _load_frame_tensor(self, unique_person_id: DukePersonId, camera_id: DukeCameraId, frame_name: DukeFrameName, split: DukeSplit | None = None) -> torch.Tensor | None:
        if split is None:
            split = self.main_split

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
        # The segmentation path is at the same relative path as the frame path. So we can just take the frame path and modify it to get the segmentation path.

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
        if split is None:
            split = self.main_split

        segmentation_path = self._get_segmentation_path(unique_person_id, camera_id, frame_name, split)
        if segmentation_path is None:
            return None
        
        segmentation_tensor = read_image(str(segmentation_path), ImageReadMode.GRAY)
        return segmentation_tensor

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index: int) -> DukeMTMCItem:
        person_id, camera_id, frame_name = self.frame_list[index]
        frame_path = self._get_frame_path(person_id, camera_id, frame_name)
        if self.load_image_pil:
            image_pil = self._load_frame_pil(person_id, camera_id, frame_name)
        else:
            image_pil = None

        if self.load_image_tensor:
            image_tensor = self._load_frame_tensor(person_id, camera_id, frame_name)
        else:
            image_tensor = None

        if self.sidecar_root is not None:
            segmentation_path = self._get_segmentation_path(person_id, camera_id, frame_name)
        else:
            segmentation_path = None

        if self.load_segmentations:
            segmentation_tensor = self._load_segmentation_tensor(person_id, camera_id, frame_name)
        else:
            segmentation_tensor = None

        return DukeMTMCItem(
            person_id=person_id,
            camera_id=camera_id,
            frame_name=frame_name,
            frame_path=frame_path,
            frame=image_pil,
            frame_tensor=image_tensor,
            segmentation_path=segmentation_path,
            segmentation_tensor=segmentation_tensor,
        )

class DukeMTMCVideoDatasetVideoKPFBatchSampler(Sampler):
    """
    For our training process we need to have K people with P views per person. Within each view we sample F frames to form the batch.
    """

    def __init__(
        self, 
        dataset: DukeMTMCVideoDataset,
        batches_per_epoch: int | None,
        num_people_per_batch: int,
        num_views_per_person: int,
        num_frames_per_view: int,
        allow_same_person_same_view: bool = False,
        allow_reduced_views_per_person: bool = False,  # If true, allow sampling fewer views than num_views_per_person for people who do not have enough views
        allow_resampling_sample_indices: bool = False,  # Allow sampling the same frame more than once in the same view if the view has fewer frames than num_frames_per_view
        epoch_deterministic: bool = False,  # If true, seed is reset every epoch
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.seed = seed
        self.epoch_deterministic = epoch_deterministic
        self.rng = random.Random(self.seed) if self.seed is not None else random.Random()

        self.ds_frame_list = dataset.frame_list
        self.person_to_frame_idx_map: dict[DukePersonId, dict[DukeCameraId, list[int]]] = {}
        for frame_idx, (person_id, camera_id, _) in enumerate(self.ds_frame_list):
            if person_id not in self.person_to_frame_idx_map:
                self.person_to_frame_idx_map[person_id] = {}
            if camera_id not in self.person_to_frame_idx_map[person_id]:
                self.person_to_frame_idx_map[person_id][camera_id] = []
            self.person_to_frame_idx_map[person_id][camera_id].append(frame_idx)
        self.all_person_ids = sorted(list(self.person_to_frame_idx_map.keys()))

        self.dataset = dataset
        self.batches_per_epoch = batches_per_epoch
        self.num_people_per_batch = num_people_per_batch
        self.num_views_per_person = num_views_per_person
        self.num_frames_per_view = num_frames_per_view
        self.allow_same_person_same_view = allow_same_person_same_view
        self.allow_reduced_views_per_person = allow_reduced_views_per_person
        self.allow_resampling_sample_indices = allow_resampling_sample_indices

        # Ensure that no view has less than self.num_frames_per_view frames. This is absolutely unsupported
        for person_id in self.all_person_ids:
            for camera_id in self.person_to_frame_idx_map[person_id]:
                if len(self.person_to_frame_idx_map[person_id][camera_id]) < self.num_frames_per_view:
                    if not self.allow_resampling_sample_indices:
                        raise ValueError(f"Person {person_id} has fewer than {self.num_frames_per_view} frames in camera view {camera_id}. Set allow_resampling_sample_indices to True if you want to allow sampling the same frame more than once in the same view if the view has fewer frames than num_frames_per_view.")
                    else:
                        print(f"WARNING: Person {person_id} has fewer than {self.num_frames_per_view} ({len(self.person_to_frame_idx_map[person_id][camera_id])}) frames in camera view {camera_id}.")

        assert not(self.allow_reduced_views_per_person and self.allow_same_person_same_view), "Setting both allow_reduced_views_per_person and allow_same_person_same_view is not allowed."
        if not self.allow_reduced_views_per_person and not self.allow_same_person_same_view:
            # Check to make sure all people have at least num_views_per_person views.
            for person_id in self.all_person_ids:
                if len(self.person_to_frame_idx_map[person_id]) < self.num_views_per_person:
                    raise ValueError(f"Person {person_id} has fewer than {self.num_views_per_person} views. Set allow_reduced_views_per_person to True if you want to allow sampling fewer views than num_views_per_person for people who do not have enough views.")
        elif self.allow_reduced_views_per_person:
            print(f"WARNING: allow_reduced_views_per_person is set to True, so people with fewer than {self.num_views_per_person} views will be sampled with fewer views. This will create ragged batch sizes.")
        elif self.allow_same_person_same_view:
            print(f"WARNING: allow_same_person_same_view is set to True, so the same view may be sampled multiple times for the same person.")

        assert self.num_people_per_batch <= len(self.all_person_ids), f"Number of people per batch ({self.num_people_per_batch}) must be less than or equal to the number of people in the dataset ({len(self.all_person_ids)})."

        self.frames_per_batch = self.num_people_per_batch * self.num_views_per_person * self.num_frames_per_view
        if verbose:
            print(f"Created a new {__class__.__name__} with {self.frames_per_batch} frames per batch and epoch length of {self.batches_per_epoch}")
            print(f"This will mean processing {self.frames_per_batch * self.batches_per_epoch} frames per epoch.")

    def sample_camera_views(self, available_camera_ids: list[DukeCameraId]) -> list[DukeCameraId]:
        if self.allow_same_person_same_view:
            # Then we grab all views as many times as possible and then sample without replacement for the rest
            k = self.num_views_per_person
            n_avail = len(available_camera_ids)

            sampled_views = available_camera_ids * (k // n_avail)
            sampled_views += self.rng.sample(available_camera_ids, k % n_avail)
            self.rng.shuffle(sampled_views)
            
            return sampled_views

        # Otherwise we need to ensure that we sample without replacement
        # However, we also need to account for reduced views if allow_reduced_views_per_person is true
        if not self.allow_reduced_views_per_person:
            assert len(available_camera_ids) >= self.num_views_per_person, f"Not enough camera views available to sample {self.num_views_per_person} views."
            true_num_views_to_sample = self.num_views_per_person
        else:
            true_num_views_to_sample = min(self.num_views_per_person, len(available_camera_ids))

        # Now we sample without replacement
        return self.rng.sample(available_camera_ids, k=true_num_views_to_sample)

    def sample_batch(self) -> list[int]:
        """
        Returns a list of sample indices from the dataset to be loaded in a single batch.
        The list is of size B = num_people_per_batch * num_views_per_person * num_frames_per_view
        """
        batch_sample_indices: list[int] = []

        # We first sample the individuals that will make up the batch
        batch_person_ids: list[DukePersonId] = self.rng.sample(self.all_person_ids, self.num_people_per_batch)

        # For each of these individuals, we sample the actual frames
        for person_id in batch_person_ids:
            person_camera_dict: dict[DukeCameraId, list[int]] = self.person_to_frame_idx_map[person_id]
            person_camera_ids: list[DukeCameraId] = list(person_camera_dict.keys())

            # We have a helper that manages the sampling of camera views for us based on the class parameters
            sampled_camera_views: list[DukeCameraId] = self.sample_camera_views(person_camera_ids)
            
            # Note that we can have repeated camera views in some cases. For sampling individual frames, we
            # attempt to ensure that frames are disjoint between different indices in the sampled_camera_views even if
            # we repeat a view. We do this by maintaining a set of used frames and adding used frames once we sample them
            # for each camera view. If we run out we print a warning and clear the set to allow reuse of frames
            used_view_sample_indices: dict[DukeCameraId, set[int]] = {camera_id: set() for camera_id in set(sampled_camera_views)}
            for internal_view_id, camera_id in enumerate(sampled_camera_views):
                view_sample_indices: list[int] = person_camera_dict[camera_id]  # List of samples that correspond to the sampled person and camera view

                # To get the frames we can sample without having repeat frames between
                already_used_frame_views: set[int] = used_view_sample_indices[camera_id]
                # Check if we have enough frames to sample
                if len(already_used_frame_views) + self.num_frames_per_view > len(view_sample_indices):
                    # Print a warning and clear the set to allow reuse of frames
                    print(f"Warning: Not enough frames available to sample {self.num_frames_per_view} frames from camera view {camera_id} without replacement. Reusing frames.")
                    already_used_frame_views.clear()
                available_frame_indices = [frame_idx for frame_idx in view_sample_indices if frame_idx not in already_used_frame_views]

                # Now we can sample the frames for this camera view without replacement
                if len(available_frame_indices) < self.num_frames_per_view:
                    assert self.allow_resampling_sample_indices, f"Not enough frames available to sample {self.num_frames_per_view} frames from camera view {camera_id} for person {person_id} without replacement. Only {len(available_frame_indices)} frames are available."
                    selected_sample_indices = self.rng.choices(available_frame_indices, k=self.num_frames_per_view)
                else:
                    selected_sample_indices = self.rng.sample(available_frame_indices, self.num_frames_per_view)
                # Update the set of used frames for this camera view
                used_view_sample_indices[camera_id].update(selected_sample_indices)

                # Now add the selected samples to the batch
                batch_sample_indices.extend(selected_sample_indices)

        return batch_sample_indices

    def __len__(self) -> int:
        if self.batches_per_epoch is None:
            raise TypeError("This BatchSampler is set to infinite mode and has no length.")
        else:
            return self.batches_per_epoch

    def __iter__(self):
        if self.epoch_deterministic:
            # This is useful for eval loops where we want exactly the same samples each epoch
            # so that a model run on epoch 0 can be compared to a model run on epoch 1
            assert self.seed is not None, "seed must be set for deterministic mode"
            print(f"Resetting RNG for epoch with seed {self.seed}")
            self.rng = random.Random(self.seed)

        batch_count = 0
        while self.batches_per_epoch is None or batch_count < self.batches_per_epoch:
            yield self.sample_batch()
            batch_count += 1
