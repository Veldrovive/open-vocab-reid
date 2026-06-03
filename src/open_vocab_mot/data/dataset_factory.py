import torch
from pydantic import BaseModel, ConfigDict
from typing import Optional, Literal, Tuple, Dict, Any, List

from open_vocab_mot.data.whale_ds import WhaleDataset, WhaleSplit
from open_vocab_mot.data.duke_mtmc_video_ds import DukeMTMCVideoDataset, DukeSplit
from open_vocab_mot.data.wildlife_10k_subset_ds import Wildlife10KSubsetDataset, Wildlife10KSplit, Wildlife10KDatasets
from open_vocab_mot.data.veri_video_ds import VeRiVideoDataset, VeRiSplit
from open_vocab_mot.data.vrai_ds import VRAIDataset, VRAISplit
from open_vocab_mot.data.video_reid_abc import VideoReIDKPFBatchIterableDataset, AbstractVideoReIDDataset

from open_vocab_mot.definitions import (
    DUKEMTMC_VIDEO_REID_PATH, 
    DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
    WHALE_DATASET_PATH,
    WHALE_DATASET_SIDECAR_PATH,
    WILDLIFE_10K_PATH,
    WILDLIFE_10K_SIDECAR_PATH,
    VERI_DATASET_PATH,
    VERI_DATASET_SIDECAR_PATH,
    VRAI_DATASET_PATH,
    VRAI_DATASET_SIDECAR_PATH
)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    use_for_training: bool = True
    weight: float = 1.0
    frames_per_video: int = 4
    people_per_batch: int = 16
    views_per_person: int = 3
    use_for_video_eval: bool = False
    use_for_video_mini_eval: bool = False
    use_for_frame_eval: bool = False
    use_for_frame_mini_eval: bool = False

class DukeDatasetConfig(DatasetConfig):
    pass

class WhaleDatasetConfig(DatasetConfig):
    min_num_images: int = 0

class Wildlife10kSubsetDatasetConfig(DatasetConfig):
    subset_dataset: Wildlife10KDatasets
    min_num_images: int = 0

class VeRiDatasetConfig(DatasetConfig):
    query_collapse_sequences: bool = False

class VRAIDatasetConfig(DatasetConfig):
    pass


def load_dataset_for_training(
    dataset_type: Literal["duke", "whale", "wildlife10k_subset", "veri", "vrai"],
    config: DatasetConfig,
    seed: int,
    verbose: bool = False
) -> VideoReIDKPFBatchIterableDataset:
    
    kwargs = config.model_dump(exclude={"use_for_training", "weight", "frames_per_video", "people_per_batch", "views_per_person", "use_for_video_eval", "use_for_video_mini_eval", "use_for_frame_eval", "use_for_frame_mini_eval", "subset_dataset", "min_num_images"}, exclude_unset=True, exclude_none=True)
    
    if dataset_type == "duke":
        ds = DukeMTMCVideoDataset(
            ds_root=DUKEMTMC_VIDEO_REID_PATH,
            main_split=DukeSplit.TRAIN,
            sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            **kwargs
        )
    elif dataset_type == "whale":
        # config is a WhaleDatasetConfig
        ds = WhaleDataset(
            ds_root=WHALE_DATASET_PATH,
            split=WhaleSplit.TRAIN,
            sidecar_root=WHALE_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            min_num_images=getattr(config, "min_num_images", 0),
            verbose=verbose,
            **kwargs
        )
    elif dataset_type == "wildlife10k_subset":
        ds = Wildlife10KSubsetDataset(
            ds_root=WILDLIFE_10K_PATH,
            dataset_name=getattr(config, "subset_dataset"),
            split=Wildlife10KSplit.TRAIN,
            sidecar_root=WILDLIFE_10K_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            min_num_images=getattr(config, "min_num_images", 0),
            verbose=verbose,
            **kwargs
        )
    elif dataset_type == "veri":
        kwargs.pop("query_collapse_sequences", None)
        kwargs.pop("collapse_sequences", None)
        ds = VeRiVideoDataset(
            ds_root=VERI_DATASET_PATH,
            main_split=VeRiSplit.TRAIN,
            sidecar_root=VERI_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            collapse_sequences=False,
            **kwargs
        )
    elif dataset_type == "vrai":
        ds = VRAIDataset(
            ds_root=VRAI_DATASET_PATH,
            main_split=VRAISplit.TRAIN,
            sidecar_root=VRAI_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    return VideoReIDKPFBatchIterableDataset(
        ds,
        batches_per_epoch=None,
        num_identities_per_batch=config.people_per_batch,
        num_sequences_per_identity=config.views_per_person,
        num_frames_per_sequence=config.frames_per_video,
        allow_same_identity_same_sequence=True,
        allow_reduced_sequences_per_identity=False,
        allow_resampling_sample_indices=True,
        epoch_deterministic=False,
        seed=seed,
        verbose=verbose
    )

def load_dataset_for_eval(
    dataset_type: Literal["duke", "whale", "wildlife10k_subset", "veri", "vrai"],
    config: DatasetConfig,
    verbose: bool = False
) -> Dict[str, AbstractVideoReIDDataset]:
    kwargs = config.model_dump(exclude={"use_for_training", "weight", "frames_per_video", "people_per_batch", "views_per_person", "use_for_video_eval", "use_for_video_mini_eval", "use_for_frame_eval", "use_for_frame_mini_eval", "subset_dataset", "min_num_images"}, exclude_unset=True, exclude_none=True)
    
    if dataset_type == "duke":
        query_ds = DukeMTMCVideoDataset(
            ds_root=DUKEMTMC_VIDEO_REID_PATH,
            main_split=DukeSplit.QUERY,
            sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            **kwargs
        )
        gallery_ds = DukeMTMCVideoDataset(
            ds_root=DUKEMTMC_VIDEO_REID_PATH,
            main_split=DukeSplit.GALLERY,
            sidecar_root=DUKEMTMC_VIDEO_REID_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            **kwargs
        )
        return {"query": query_ds, "gallery": gallery_ds}
        
    elif dataset_type == "whale":
        val_ds = WhaleDataset(
            ds_root=WHALE_DATASET_PATH,
            split=WhaleSplit.VAL,
            sidecar_root=WHALE_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            min_num_images=getattr(config, "min_num_images", 0),
            verbose=verbose,
            **kwargs
        )
        return {"val": val_ds}
        
    elif dataset_type == "wildlife10k_subset":
        val_ds = Wildlife10KSubsetDataset(
            ds_root=WILDLIFE_10K_PATH,
            dataset_name=getattr(config, "subset_dataset"),
            split=Wildlife10KSplit.VAL,
            sidecar_root=WILDLIFE_10K_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            min_num_images=getattr(config, "min_num_images", 0),
            verbose=verbose,
            **kwargs
        )
        return {"val": val_ds}
        
    elif dataset_type == "veri":
        query_collapse_sequences = getattr(config, "query_collapse_sequences", False)
        kwargs.pop("query_collapse_sequences", None)
        kwargs.pop("collapse_sequences", None)
        query_ds = VeRiVideoDataset(
            ds_root=VERI_DATASET_PATH,
            main_split=VeRiSplit.QUERY,
            sidecar_root=VERI_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            collapse_sequences=query_collapse_sequences,
            **kwargs
        )
        gallery_ds = VeRiVideoDataset(
            ds_root=VERI_DATASET_PATH,
            main_split=VeRiSplit.GALLERY,
            sidecar_root=VERI_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            collapse_sequences=False,
            **kwargs
        )
        return {"query": query_ds, "gallery": gallery_ds}
        
    elif dataset_type == "vrai":
        val_ds = VRAIDataset(
            ds_root=VRAI_DATASET_PATH,
            main_split=VRAISplit.QUERY,
            sidecar_root=VRAI_DATASET_SIDECAR_PATH,
            load_image_pil=False,
            load_image_tensor=True,
            load_segmentations=True,
            verbose=verbose,
            **kwargs
        )
        return {"val": val_ds}
        
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")
