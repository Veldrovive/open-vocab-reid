"""
Docs:

In annotations/train.json
Note that annotations/validation.json and annotations/tao_test_annotations.json also exist
root: dict with keys 'videos', 'annotations', 'tracks', 'images', 'info', 'categories', 'licenses'
    videos: list of dict with keys 'id', 'width', 'height', 'neg_category_ids', 'not_exhaustive_category_ids', 'name', 'metadata'
    500 entries total in train
        id: int
        width: int
        height: int
        neg_category_ids: list[int]
        not_exhaustive_category_ids: list[int]
        name: str (Seems like a path or id within a subdataset. One was train/YFCC100M/v_f69ebe5b731d3e87c1a3992ee39c3b7e)
        metadata: at least sometimes dict with keys 'dataset', 'user_id', 'username'
            dataset: str (aligns with subdataset names like YFCC100M)
            user_id: str (in one case was 22634709@N00. No idea what this is. Maybe something to do with scale data annotation)
            username: str (in one case was Amsterdamized. No idea what this is. Maybe something to do with scale data annotation)
    annotations: list of dict with keys 'segmentation', 'bbox', 'area', 'iscrowd', 'id', 'image_id', 'category_id', 'track_id', '_scale_uuid', 'scale_category', 'video_id'
    54639 entries total in train
        segmentation: list[list[int]] (in one case was [[114, 166, 181, 166, 181, 237, 114, 237]]. What is this? Run length encoding? Very small for that.)
        bbox: list[int] (in one case was [114, 166, 67, 71])
        area: int (equal to bbox width * bbox height)
        iscrowd: 0 or 1 (not sure what this means. Should look it up)
        id: int (I think internal id of the sample)
        image_id: int (Must correspond to id from images)
        category_id: int 
        track_id: int (Pretty sure this must correspond to the track id below)
        _scale_uuid: str (Was '5a32709e-44a0-47b9-85af-b01286adea67' in one case)
        scale_category: str (Was 'moving object' in one case)
        video_id: int (Pretty sure this must correspond with the video["id"] from above)
    tracks: list of dict with keys 'id', 'category_id', 'video_id'
    2647 entries total in train
        id: int (I guess used to identify which annotations belong to the same object)
        category_id: int (Duplicates the category id from the annotation sample)
        video_id: int (Duplicates the video id from the annotation)
    images: list of dict with keys 'id', 'video', '_scale_task_id', 'width', 'height', 'file_name', 'frame_index', 'license', 'video_id'
    18274 entries total in train
        id: int
        video: str (Looks to be the "name" field from the video dict. One example was 'train/YFCC100M/v_f69ebe5b731d3e87c1a3992ee39c3b7e')
        _scale_task_id: str (Was '5de800eddb2c18001a56aa11' in one case)
        width: int (same as in video)
        height: int (same as in video)
        file_name: str (Looks like a subpath of video. One example is 'train/YFCC100M/v_f69ebe5b731d3e87c1a3992ee39c3b7e/frame0391.jpg')
        frame_index: int (True frame index. File name appears to be 1 indexed, but that probably isn't consistent. Same frame from above is 390)
        license: int
        video_id: 0 (Looks to be a way to index the video without searching by name)
    info: dataset metadata
        {'year': 2020, 'version': '0.1.20200120', 'description': 'Annotations imported from Scale', 'contributor': '', 'url': '', 'date_created': '2020-01-20 15:49:53.519740'}
    categories: list of dict with keys 'frequency', 'id', 'synset', 'image_count', 'instance_count', 'synonyms', 'def', 'name'
    1230 entries total in train
        frequency: str (Was the character 'r' in one case. Perhaps this means "rare"?)
        id: int (Must correspond with above ids. Note that the index in the list is not the same as id)
        synset: str (In one case was acorn.n.01)
        image_count: int (Was 0 for acorn so not all categories listed are represented)
        instance_count: int (Was 0 for acorn so not all categories listed are represented)
        synonyms: list[str] (Just words that mean the same thing. For 'acorn' the list was ['acorn'] so I guess we always duplicate the current name)
        def: str (Definition of term in natural language)
        name: str (Name of object in natural language like 'acorn')
    licenses: list of str
        The only thing in this list is the single string ["Unknown"] lol

The frame root is just 'frames' and in that are the folders 'test', 'train', and 'val'.
Inside each of those is one directory per sub dataset and inside that are video folders.
I don't think we need to deal with the paths though because the video name and frame file_name seem to be paths with the
root at 'frames'
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from open_vocab_mot import TAO_DATASET_PATH


# ==========================================
# 1. PYDANTIC MODELS (DATA VALIDATION)
# ==========================================

class VideoMetadata(BaseModel):
    dataset: Optional[str] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    
    model_config = {"extra": "allow"}

class Video(BaseModel):
    id: int
    width: int
    height: int
    neg_category_ids: List[int] = Field(default_factory=list)
    not_exhaustive_category_ids: List[int] = Field(default_factory=list)
    name: str
    metadata: Optional[VideoMetadata] = None

class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    track_id: int
    video_id: int
    bbox: List[float]
    area: float
    iscrowd: int
    segmentation: Optional[List[List[float]]] = None
    scale_uuid: Optional[str] = Field(None, alias="_scale_uuid")
    scale_category: Optional[str] = None

class Track(BaseModel):
    id: int
    category_id: int
    video_id: int

class Image(BaseModel):
    id: int
    video: str
    width: int
    height: int
    file_name: str
    frame_index: int
    video_id: int
    scale_task_id: Optional[str] = Field(None, alias="_scale_task_id")
    license: Optional[int] = None

class DatasetInfo(BaseModel):
    year: Optional[int] = None
    version: Optional[str] = None
    description: Optional[str] = None
    contributor: Optional[str] = None
    url: Optional[str] = None
    date_created: Optional[str] = None

class Category(BaseModel):
    id: int
    name: str
    synset: Optional[str] = None
    frequency: Optional[str] = None
    image_count: int = 0
    instance_count: int = 0
    synonyms: List[str] = Field(default_factory=list)
    definition: str = Field(default="", alias="def")  # 'def' is a reserved keyword in Python

class TAOAnnotationFile(BaseModel):
    videos: List[Video]
    annotations: List[Annotation]
    tracks: List[Track]
    images: List[Image]
    info: DatasetInfo
    categories: List[Category]
    licenses: List[Any] = Field(default_factory=list)


# ==========================================
# 2. DATASET HANDLER
# ==========================================

class TAODatasetHandler:
    """
    A unified harness to wrap the TAO dataset, validating JSON structures using Pydantic, 
    and building O(1) indices for common Data Loading, MOT, and ReID operations.
    """
    video_by_id: dict[int, Video]
    image_by_id: dict[int, Image]
    track_by_id: dict[int, Track]
    category_by_id: dict[int, Category]
    annotation_by_id: dict[int, Annotation]
    images_by_video: dict[int, List[Image]]
    annotations_by_image: dict[int, List[Annotation]]
    annotations_by_track: dict[int, List[Annotation]]
    annotations_by_category: dict[int, List[Annotation]]
    tracks_by_category: dict[int, List[Track]]

    def __init__(self, split: str = 'train', dataset_root: str | Path = TAO_DATASET_PATH):
        self.dataset_root = Path(dataset_root)
        self.annotation_path = {
            "train": dataset_root / "annotations/train.json",
            "val": dataset_root / "annotations/validation.json",
            "test": dataset_root / "annotations/tao_test_annotations.json"
        }[split]
        
        print(f"Loading and validating JSON from {self.annotation_path}...")
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)
            
        self.data = TAOAnnotationFile(**data)
        self._build_indices()
        print("Dataset loaded and indices built successfully.")

    def _build_indices(self):
        """Build dictionary lookups for fast retrieval."""
        # 1-to-1 mappings
        self.video_by_id: Dict[int, Video] = {v.id: v for v in self.data.videos}
        self.image_by_id: Dict[int, Image] = {i.id: i for i in self.data.images}
        self.track_by_id: Dict[int, Track] = {t.id: t for t in self.data.tracks}
        self.category_by_id: Dict[int, Category] = {c.id: c for c in self.data.categories}
        self.annotation_by_id: Dict[int, Annotation] = {a.id: a for a in self.data.annotations}

        # 1-to-Many mappings
        self.images_by_video: Dict[int, List[Image]] = defaultdict(list)
        self.annotations_by_image: Dict[int, List[Annotation]] = defaultdict(list)
        self.annotations_by_track: Dict[int, List[Annotation]] = defaultdict(list)
        self.annotations_by_category: Dict[int, List[Annotation]] = defaultdict(list)
        self.tracks_by_category: Dict[int, List[Track]] = defaultdict(list)
        
        # Populate 1-to-Many mappings
        for img in self.data.images:
            self.images_by_video[img.video_id].append(img)
            
        for ann in self.data.annotations:
            self.annotations_by_image[ann.image_id].append(ann)
            self.annotations_by_track[ann.track_id].append(ann)
            self.annotations_by_category[ann.category_id].append(ann)

        for trk in self.data.tracks:
            self.tracks_by_category[trk.category_id].append(trk)

        # Sort grouped data by time/frame_index for temporal consistency (Crucial for Video ReID)
        for vid_id in self.images_by_video:
            self.images_by_video[vid_id].sort(key=lambda x: x.frame_index)
            
        for track_id in self.annotations_by_track:
            # Sort annotations within a track by the frame_index of their corresponding image
            self.annotations_by_track[track_id].sort(
                key=lambda a: self.image_by_id[a.image_id].frame_index
            )

    # --- BASIC RETRIEVAL METHODS ---

    def get_video(self, video_id: int) -> Video:
        return self.video_by_id[video_id]

    def get_image(self, image_id: int) -> Image:
        return self.image_by_id[image_id]
        
    def get_category(self, category_id: int) -> Category:
        return self.category_by_id[category_id]

    def get_track(self, track_id: int) -> Track:
        return self.track_by_id[track_id]

    def get_annotation(self, annotation_id: int) -> Annotation:
        return self.annotation_by_id[annotation_id]

    def get_image_absolute_path(self, image: Image) -> Path:
        """Resolves the physical absolute path for a given image."""
        return self.dataset_root / "frames" / image.file_name

    # --- ITERATION & CLASS METHODS ---

    def get_video_frames(self, video_id: int) -> List[Image]:
        """Returns all frame definitions for a video, sorted chronologically."""
        return self.images_by_video.get(video_id, [])

    def get_annotations_in_image(self, image_id: int) -> List[Annotation]:
        """Returns all bounding boxes/samples within a single frame."""
        return self.annotations_by_image.get(image_id, [])

    def get_annotations_by_category(self, category_id: int) -> List[Annotation]:
        """Returns all annotation samples belonging to a specific class."""
        return self.annotations_by_category.get(category_id, [])

    # --- VIDEO RE-IDENTIFICATION (ReID) SPECIFIC METHODS ---
    
    def get_track_sequence(self, track_id: int) -> List[Tuple[Image, Annotation]]:
        """
        Crucial for ReID: Returns the chronological sequence of an object's appearances.
        Returns a list of (Image, Annotation) tuples tracking a single object over time.
        """
        annotations = self.annotations_by_track.get(track_id, [])
        return [(self.get_image(ann.image_id), ann) for ann in annotations]

    def get_reid_candidates(self, category_id: int, min_sequence_length: int = 5) -> List[int]:
        """
        For ReID training, you only want tracks that last long enough to form positive pairs.
        Returns a list of track_ids for a specific category that have >= `min_sequence_length` frames.
        """
        tracks = self.tracks_by_category.get(category_id, [])
        valid_track_ids = []
        for track in tracks:
            ann_count = len(self.annotations_by_track.get(track.id, []))
            if ann_count >= min_sequence_length:
                valid_track_ids.append(track.id)
        return valid_track_ids

    def generate_reid_pairs(self, category_id: int, min_sequence_length: int = 5):
        """
        Hypothetical Generator for Video ReID positive pairs.
        Yields (track_id, [crop_sequence_1], [crop_sequence_2])
        """
        import random
        valid_tracks = self.get_reid_candidates(category_id, min_sequence_length)
        
        for track_id in valid_tracks:
            sequence = self.get_track_sequence(track_id)
            
            # ReID often involves splitting a track into two temporally distinct tracklets
            midpoint = len(sequence) // 2
            tracklet_1 = sequence[:midpoint]
            tracklet_2 = sequence[midpoint:]
            
            yield track_id, tracklet_1, tracklet_2