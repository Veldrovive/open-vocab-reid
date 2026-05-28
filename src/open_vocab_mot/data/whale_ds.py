import os
from pathlib import Path
from enum import Enum
import numpy as np

import torch
from torchvision.io import read_image, ImageReadMode
from PIL import Image
from tqdm import tqdm

from wildlife_datasets import datasets
from open_vocab_mot.data.video_reid_abc import AbstractVideoReIDDataset, VideoReIDItem, IdentityId, SequenceId

class WhaleSplit(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class WhaleDataset(AbstractVideoReIDDataset):
    def __init__(
        self,
        ds_root: Path | str,
        split: WhaleSplit,
        sidecar_root: Path | str | None = None,
        load_image_pil: bool = False,
        load_image_tensor: bool = False,
        image_tensor_dtype: torch.dtype | None = None,
        load_segmentations: bool = False,
        download_if_missing: bool = False,
        verbose: bool = False,
        min_num_images: int = 0,
        val_split_frac: float = 0.2,
        val_split_seed: int = 42
    ):
        self.ds_root = Path(ds_root)
        if sidecar_root is not None:
            self.sidecar_root = Path(sidecar_root)
        else:
            self.sidecar_root = None
            
        self.load_image_pil = load_image_pil
        self.load_image_tensor = load_image_tensor
        self.image_tensor_dtype = image_tensor_dtype
        self.load_segmentations = load_segmentations
        self.download_if_missing = download_if_missing
        self.verbose = verbose
        self.min_num_images = min_num_images
        self.split = split
        self.val_split_frac = val_split_frac
        self.val_split_seed = val_split_seed

        if self.load_segmentations:
            assert self.sidecar_root is not None, "WhaleDataset sidecar root must be specified for loading segmentations."

        if self.download_if_missing:
            self._download_dataset()
        else:
            assert self.ds_root.exists(), f"Whale dataset not found at {self.ds_root}"

        # Initialize wildlife_datasets instance
        self.whales = datasets.HumpbackWhaleID(str(self.ds_root))
        
        # Parse the dataframe to populate the sequence map and frame list
        self._prepare_dataset()

    def _download_dataset(self):
        datasets.HumpbackWhaleID.get_data(str(self.ds_root))

    def _prepare_dataset(self):
        """
        Parses the dataframe to build the sequence_map and frame_list.
        Filters out 'unknown' identities and maps string identities to integers.
        """
        df = self.whales.df
        
        if self.min_num_images > 0:
            counts = df['identity'].value_counts()
            valid_identities = counts[counts >= self.min_num_images].index
            df = df[df['identity'].isin(valid_identities)].copy()

        # Filter down to the specified split (train/test)
        # To split between train and val, we find the set of identities, shuffle it using a seeded shuffle
        # and then split it
        unique_identities = df['identity'].unique()
        rng = np.random.default_rng(self.val_split_seed)
        rng.shuffle(unique_identities)
        val_split_idx = int(len(unique_identities) * (1-self.val_split_frac))
        if self.split == WhaleSplit.TRAIN:
            train_val_df = df[df['original_split'] == 'train'].copy()
            train_identities = unique_identities[:val_split_idx]
            df = train_val_df[train_val_df['identity'].isin(train_identities)].copy()
        elif self.split == WhaleSplit.VAL:
            train_val_df = df[df['original_split'] == 'train'].copy()
            val_identities = unique_identities[val_split_idx:]
            df = train_val_df[train_val_df['identity'].isin(val_identities)].copy()
        elif self.split == WhaleSplit.TEST:
            df = df[df['original_split'] == 'test'].copy()
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Wildlife datasets use string identities. We will map them to integers
        # for consistency with standard ReID integer IDs.
        unique_id_strs = df['identity'].unique()
        self.identity_str_to_id = {name: i for i, name in enumerate(unique_id_strs)}
        
        self._frame_list: list[dict] = []
        self._sequence_map: dict[IdentityId, dict[SequenceId, list[int]]] = {}
        
        iterator = df.iterrows()
        if self.verbose:
            iterator = tqdm(iterator, total=len(df), desc="Processing WhaleDataset")
            
        for _, row in iterator:
            ident_str = row['identity']
            ident_id = self.identity_str_to_id[ident_str]
            
            # Whales dataset consists of independent images rather than distinct video cameras.
            # We map all images for an identity to sequence_id 0 to satisfy the abstract structure.
            seq_id = 0 
            
            frame_id = row['image_id']
            rel_path = row['path']
            
            flat_idx = len(self._frame_list)
            
            self._frame_list.append({
                'identity_id': ident_id,
                'sequence_id': seq_id,
                'frame_id': frame_id,
                'rel_path': rel_path
            })
            
            if ident_id not in self._sequence_map:
                self._sequence_map[ident_id] = {}
            if seq_id not in self._sequence_map[ident_id]:
                self._sequence_map[ident_id][seq_id] = []
                
            self._sequence_map[ident_id][seq_id].append(flat_idx)

    # --- AbstractVideoReIDDataset Properties ---

    @property
    def unique_identities(self) -> list[IdentityId]:
        return list(self._sequence_map.keys())

    @property
    def sequence_map(self) -> dict[IdentityId, dict[SequenceId, list[int]]]:
        return self._sequence_map

    # --- Standard Magic Methods ---

    def __len__(self) -> int:
        return len(self._frame_list)

    def __getitem__(self, index: int) -> VideoReIDItem:
        item_info = self._frame_list[index]
        
        identity_id = item_info['identity_id']
        sequence_id = item_info['sequence_id']
        frame_id = item_info['frame_id']
        rel_path = item_info['rel_path']
        
        frame_path = self.ds_root / rel_path
        
        # Load PIL Image
        image_pil = None
        if self.load_image_pil:
            image_pil = Image.open(frame_path).convert("RGB")
            
        # Load Tensor
        image_tensor = None
        if self.load_image_tensor:
            image_tensor = read_image(str(frame_path), ImageReadMode.RGB)
            if self.image_tensor_dtype is not None:
                image_tensor = image_tensor.to(self.image_tensor_dtype) / 255.0
            else:
                image_tensor = image_tensor.float() / 255.0
                
        # Handle Segmentations
        segmentation_path = None
        segmentation_tensor = None
        if self.sidecar_root:
            segmentation_path = self.sidecar_root / rel_path
            
            # Fallback check in case masks are saved as .png instead of the original .jpg
            if not segmentation_path.exists() and segmentation_path.suffix != '.png':
                alt_path = segmentation_path.with_suffix('.png')
                if alt_path.exists():
                    segmentation_path = alt_path

            # We verify exists() in case a particular mask is missing from the sidecar
            if not segmentation_path.exists():
                segmentation_path = None
            elif self.load_segmentations:
                segmentation_tensor = read_image(str(segmentation_path), ImageReadMode.GRAY)
                # Normalize segmentation masks to [0, 1] if needed
                segmentation_tensor = segmentation_tensor.float() / 255.0

        return VideoReIDItem(
            sample_index=index,
            identity_id=identity_id,
            sequence_id=sequence_id,
            frame_id=frame_id,
            frame_path=frame_path,
            frame=image_pil,
            frame_tensor=image_tensor,
            segmentation_path=segmentation_path,
            segmentation_tensor=segmentation_tensor,
        )