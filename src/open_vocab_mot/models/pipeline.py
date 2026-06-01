import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import TypedDict
from jaxtyping import Float

class PipelineOutput(TypedDict):
    video_contrastive_embeddings: Float[torch.Tensor, "num_videos video_contrastive_dim"] | None
    frame_contrastive_embeddings: list[Float[torch.Tensor, "frames_in_video frame_contrastive_dim"]] | None

class AbstractVideoReIDPipeline(nn.Module, ABC):
    """
    Abstract wrapper that provides a unified interface for Video ReID models.
    It takes raw image frames and their optional segmentations and produces
    video and optionally frame-level contrastive embeddings.
    """
    @abstractmethod
    def forward(
        self,
        frames: list[list[torch.Tensor]],
        segmentations: list[list[torch.Tensor | None]],
        extract_video: bool = True,
        extract_frames: bool = False
    ) -> PipelineOutput:
        """
        Args:
            frames: A list of videos, where each video is a list of frame image tensors.
            segmentations: A list of videos, where each video is a list of segmentation mask tensors
                           (or None if no segmentation is available for that frame).
            extract_video: Whether to compute and return video-level contrastive embeddings.
            extract_frames: Whether to compute and return frame-level contrastive embeddings.
            
        Returns:
            A PipelineOutput dictionary containing the requested embeddings.
        """
        pass

from aidan_lib.models.dino_lib_compiled import DINOv3CompiledHarness
from open_vocab_mot.models.reid_transformer import HierarchicalVideoReIDTransformer, NestedHierarchicalVideoReIDTransformer

class HierarchicalReIDPipeline(AbstractVideoReIDPipeline):
    """
    Concrete pipeline that wraps DINO and the HierarchicalVideoReIDTransformer.
    """
    def __init__(
        self,
        dino_harness: DINOv3CompiledHarness,
        reid_model: HierarchicalVideoReIDTransformer | NestedHierarchicalVideoReIDTransformer,
        dino_batch_split: int = 1
    ):
        super().__init__()
        self.dino_harness = dino_harness
        self.reid_model = reid_model
        self.dino_batch_split = dino_batch_split

    @torch.no_grad()
    def forward(
        self,
        frames: list[list[torch.Tensor]],
        segmentations: list[list[torch.Tensor | None]],
        extract_video: bool = True,
        extract_frames: bool = False
    ) -> PipelineOutput:
        device = next(self.reid_model.parameters()).device
        
        all_dino_embeddings: list[list[torch.Tensor]] = []
        
        for video_idx, (video_frames, video_segs) in enumerate(zip(frames, segmentations)):
            valid_imgs = []
            valid_segs = []
            valid_indices = []
            
            for idx, (img, seg) in enumerate(zip(video_frames, video_segs)):
                if seg is not None:
                    valid_imgs.append(img)
                    valid_segs.append(seg)
                    valid_indices.append(idx)

            dino_embeddings = [[] for _ in range(len(video_frames))]
            
            if len(valid_imgs) > 0:
                if self.dino_batch_split > 1:
                    valid_embs = []
                    num_valid = len(valid_imgs)
                    for i in range(self.dino_batch_split):
                        start_idx = i * num_valid // self.dino_batch_split
                        end_idx = (i + 1) * num_valid // self.dino_batch_split
                        if start_idx < end_idx:
                            split_imgs = [e.to(device) for e in valid_imgs[start_idx:end_idx]]
                            split_segs = valid_segs[start_idx:end_idx]
                            valid_embs.extend(self.dino_harness.match_bool_segmentations_to_dino(split_imgs, split_segs))
                            del split_imgs
                else:
                    imgs_torch = [e.to(device) for e in valid_imgs]
                    valid_embs = self.dino_harness.match_bool_segmentations_to_dino(imgs_torch, valid_segs)
                    del imgs_torch
                
                for valid_idx, emb in zip(valid_indices, valid_embs):
                    dino_embeddings[valid_idx] = emb

            dino_embedding_video: list[torch.Tensor] = []
            for frame_dino_embedding in dino_embeddings:
                if len(frame_dino_embedding) == 0:
                    continue
                dino_embedding_video.append(frame_dino_embedding[0].dino_embeddings)
                
            all_dino_embeddings.append(dino_embedding_video)

        # Filter out empty videos
        valid_video_indices = [i for i, video in enumerate(all_dino_embeddings) if len(video) > 0]
        if len(valid_video_indices) == 0:
            return PipelineOutput(
                video_contrastive_embeddings=None if not extract_video else torch.empty((0, self.reid_model.video_contrastive_dim), device=device),
                frame_contrastive_embeddings=None if not extract_frames else []
            )
            
        valid_dino_embeddings = [all_dino_embeddings[i] for i in valid_video_indices]
        
        # 2. Embed frames
        frame_embedding_data = self.reid_model.embed_frames(valid_dino_embeddings, device=device)
        
        output_frame_embeddings = None
        if extract_frames:
            # frame_embedding_data["video_frame_contrastive_embeddings"] is a list of tensors of shape (frames_in_video, dim)
            output_frame_embeddings = []
            valid_idx_ptr = 0
            for i in range(len(frames)):
                if i in valid_video_indices:
                    output_frame_embeddings.append(frame_embedding_data["video_frame_contrastive_embeddings"][valid_idx_ptr])
                    valid_idx_ptr += 1
                else:
                    output_frame_embeddings.append(torch.empty((0, self.reid_model.frame_contrastive_dim), device=device))
        
        output_video_embeddings = None
        if extract_video:
            video_embedding_data = self.reid_model.embed_video_frames(
                frame_embedding_data["video_frame_cls_tokens"], device=device
            )
            valid_video_emb = video_embedding_data["video_contrastive_embeddings"]
            
            output_video_embeddings = torch.zeros(
                (len(frames), self.reid_model.video_contrastive_dim),
                dtype=valid_video_emb.dtype,
                device=device
            )
            for i, idx in enumerate(valid_video_indices):
                output_video_embeddings[idx] = valid_video_emb[i]

        return PipelineOutput(
            video_contrastive_embeddings=output_video_embeddings,
            frame_contrastive_embeddings=output_frame_embeddings
        )

