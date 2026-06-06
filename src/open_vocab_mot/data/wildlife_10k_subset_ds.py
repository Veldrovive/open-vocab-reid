"""
ReID dataset built to extract a subset of the full wildlife 10k that is known to contain hard negatives
because they come from the same underlying dataset.
"""

import pandas
from typing import Literal
import os
from pathlib import Path
from enum import Enum
import numpy as np

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision import tv_tensors
from typing import Callable, Any
from PIL import Image
from tqdm import tqdm

from wildlife_datasets import datasets
from open_vocab_mot.data.video_reid_abc import AbstractVideoReIDDataset, VideoReIDItem, IdentityId, SequenceId, DatasetSplit

Wildlife10KDatasets = Literal[
    'AAUZebraFish', 'AerialCattle2017',
    'AmvrakikosTurtles', 'ATRW',
    'BelugaID', 'BirdIndividualID',
    'CatIndividualImages', 'Chicks4FreeID',
    'CowDataset', 'Cows2021',
    'CTai', 'CZoo',
    'DogFaceNet', 'FriesianCattle2015',
    'FriesianCattle2017', 'Giraffes',
    'GiraffeZebraID', 'HyenaID2022',
    'IPanda50', 'LeopardID2022',
    'MPDD', 'MultiCamCows2024',
    'NDD20', 'NyalaData',
    'OpenCows2020', 'PolarBearVidID',
    'PrimFace', 'ReunionTurtles',
    'SealID', 'SeaStarReID2023',
    'SeaTurtleID2022', 'SMALST',
    'SouthernProvinceTurtles', 'StripeSpotter',
    'WhaleSharkID', 'ZakynthosTurtles',
    'ZindiTurtleRecall'
]



class Wildlife10KSubsetDataset(AbstractVideoReIDDataset):
    ds: datasets.WildlifeReID10k

    def __init__(
        self,
        ds_root: Path | str,
        dataset_name: Wildlife10KDatasets,
        split: DatasetSplit,
        sidecar_root: Path | str | None = None,
        load_image_pil: bool = False,
        load_image_tensor: bool = False,
        image_tensor_dtype: torch.dtype | None = None,
        load_segmentations: bool = False,
        pre_loaded_ds: datasets.WildlifeReID10k | None = None,
        download_if_missing: bool = False,
        verbose: bool = False,
        min_num_images: int = 0,
        val_split_frac: float = 0.2,
        val_split_seed: int = 42,
        transform: Callable | None = None,
    ):
        self.ds_root = Path(ds_root)
        if sidecar_root is not None:
            self.sidecar_root = Path(sidecar_root)
        else:
            self.sidecar_root = None
        self.dataset_name = dataset_name

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
        self.transform = transform

        if self.load_segmentations:
            assert self.sidecar_root is not None, "Wildlife10KSubsetDataset sidecar root must be specified for loading segmentations."

        if pre_loaded_ds is not None:
            self.ds = pre_loaded_ds
        else:
            if self.download_if_missing:
                self._download_dataset()
            
            assert self.ds_root.exists(), f"Wildlife10KSubsetDataset not found at {self.ds_root}"

            self.ds = datasets.WildlifeReID10k(str(self.ds_root))

        self._prepare_dataset()

    def _download_dataset(self):
        datasets.WildlifeReID10k.get_data(str(self.ds_root))

    def _prepare_dataset(self):
        df: pandas.DataFrame = self.ds.df

        # Remove the identities that have too few images
        if self.min_num_images > 0:
            counts = df['identity'].value_counts()
            valid_identities = counts[counts >= self.min_num_images].index
            df = df[df['identity'].isin(valid_identities)].copy()

        # Filter down to just the dataset we want
        df = df[df['dataset'] == self.dataset_name].copy()

        # Then split out into the train and test dataframes
        train_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'] == 'test'].copy()

        unique_train_identities = train_df['identity'].unique()
        unique_test_identities = test_df['identity'].unique()
        
        if self.verbose:
            print(f"Preparring WildlifeReID10k subset {self.dataset_name} with {len(unique_train_identities)} train identities and {len(unique_test_identities)} test identities above the image count threshold")

        # Split into train and val
        rng = np.random.default_rng(self.val_split_seed)
        rng.shuffle(unique_train_identities)

        val_split_idx = int(len(unique_train_identities) * (1 - self.val_split_frac))
        if self.split == DatasetSplit.TRAIN:
            train_identities = unique_train_identities[:val_split_idx]
            df = train_df[train_df['identity'].isin(train_identities)].copy()
        elif self.split == DatasetSplit.VAL:
            val_identities = unique_train_identities[val_split_idx:]
            df = train_df[train_df['identity'].isin(val_identities)].copy()
        else:
            raise ValueError(f"Invalid split for Wildlife10KSubsetDataset: {self.split}")

        # All identities are strings. We map them to indices
        unique_id_strings = df['identity'].unique()
        self.identity_str_to_id = {name: i for i, name in enumerate(unique_id_strings)}
        
        self._frame_list: list[dict] = []
        self._sequence_map: dict[IdentityId, dict[SequenceId, list[int]]] = {}
        self.seq_id_str_to_id = {"NaN": 0}
        
        iterator = df.iterrows()
        if self.verbose:
            iterator = tqdm(iterator, total=len(df), desc=f"Processing WildlifeReID10k subset {self.dataset_name}")
            
        for _, row in iterator:
            ident_str = row['identity']
            ident_id = self.identity_str_to_id[ident_str]
            
            # # Wildlife10k dataset consists of independent images rather than distinct video cameras.
            # # We map all images for an identity to sequence_id 0 to satisfy the abstract structure.
            # seq_id = 0 
            # Nevermind! We do have a sequence id. The column cluster_id is a string that is unique between sequences
            # We can convert that to a unique integer to get a sequence id
            # When there is no sequence, this defaults to NaN which we will just call sequence 0
            seq_id_str = row['cluster_id']
            if seq_id_str not in self.seq_id_str_to_id:
                self.seq_id_str_to_id[seq_id_str] = len(self.seq_id_str_to_id)
            seq_id = self.seq_id_str_to_id[seq_id_str]

            frame_id = row['image_id']
            rel_path = row['path']
            base_dataset = row['dataset']
            species = row['species']
            
            flat_idx = len(self._frame_list)
            
            self._frame_list.append({
                'identity_id': ident_id,
                'sequence_id': seq_id,
                'frame_id': frame_id,
                'rel_path': rel_path,
                'base_dataset': base_dataset,
                'species': species
            })
            
            if ident_id not in self._sequence_map:
                self._sequence_map[ident_id] = {}
            if seq_id not in self._sequence_map[ident_id]:
                self._sequence_map[ident_id][seq_id] = []
                
            self._sequence_map[ident_id][seq_id].append(flat_idx)

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
            identity_id=identity_id,
            sequence_id=sequence_id,
            frame_id=frame_id,
            frame_path=frame_path,
            frame=image_pil,
            frame_tensor=image_tensor,
            segmentation_path=segmentation_path,
            segmentation_tensor=segmentation_tensor,
        )