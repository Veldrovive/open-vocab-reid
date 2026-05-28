from typing import TypedDict
from jaxtyping import Float

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


# Define output types for the hierarchical model
class ReIDFrameOutput(TypedDict):
    video_frame_contrastive_embeddings: list[Float[torch.Tensor, "frames_in_video frame_constrastive_dim"]]
    video_frame_cls_tokens: list[Float[torch.Tensor, "frames_in_video frame_transformer_dim"]]

class RIDVideoOutput(TypedDict):
    video_contrastive_embeddings: Float[torch.Tensor, "num_videos video_contrastive_dim"]
    video_cls_tokens: Float[torch.Tensor, "num_videos video_transformer_dim"]

class ReIDOutput(TypedDict):
    video_frame_contrastive_embeddings: list[Float[torch.Tensor, "frames_in_video frame_constrastive_dim"]]
    video_frame_cls_tokens: list[Float[torch.Tensor, "frames_in_video frame_transformer_dim"]]
    video_contrastive_embeddings: Float[torch.Tensor, "num_videos video_contrastive_dim"]
    video_cls_tokens: Float[torch.Tensor, "num_videos video_transformer_dim"]


class HierarchicalVideoReIDTransformer(nn.Module):
    """
    Model that processes frames individually followed by together over the whole video
    """

    def __init__(
        self,
        input_dim: int,
        frame_transformer_dim: int,
        frame_contrastive_dim: int,
        frame_num_heads: int,
        frame_num_layers: int,
        video_transformer_dim: int,
        video_contrastive_dim: int,
        video_num_heads: int,
        video_num_layers: int,
        dropout: float = 0.1,
        max_frame_patches: int = 1024
    ):
        super().__init__()

        self.input_dim = input_dim
        self.max_frame_patches = max_frame_patches
        self.frame_transformer_dim = frame_transformer_dim
        self.frame_contrastive_dim = frame_contrastive_dim
        self.frame_num_heads = frame_num_heads
        self.frame_num_layers = frame_num_layers

        self.video_transformer_dim = video_transformer_dim
        self.video_contrastive_dim = video_contrastive_dim
        self.video_num_heads = video_num_heads
        self.video_num_layers = video_num_layers

        # The input does not need to be the same size as the frame transformer dimension so we need to project into it
        self.input_projection = nn.Linear(input_dim, frame_transformer_dim)

        # Note that position embeddings are not needed since DINO provides position embeddings
        # We also do not need temporal embeddings because we treat frames as a bag of embeddings

        # Now we can construct the frame transformer
        self.frame_cls_token = nn.Parameter(torch.randn(1, 1, frame_transformer_dim))
        frame_encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_transformer_dim,
            nhead=frame_num_heads,
            dim_feedforward=frame_transformer_dim * frame_num_heads,
            dropout=dropout,
            activation='gelu',
            batch_first=True # Crucial: Expects input as (Batch, Seq, Feature)
        )
        self.frame_transformer = nn.TransformerEncoder(frame_encoder_layer, num_layers=frame_num_layers)

        # At the output of the frame transformer we need to both project into the contrastive space
        # and into the space of the video transformer
        self.frame_contrastive_projection = nn.Linear(frame_transformer_dim, frame_contrastive_dim)
        self.frame_to_video_projection = nn.Linear(frame_transformer_dim, video_transformer_dim)

        # Then we can construct the video transformer
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, video_transformer_dim))
        video_encoder_layer = nn.TransformerEncoderLayer(
            d_model=video_transformer_dim,
            nhead=video_num_heads,
            dim_feedforward=video_transformer_dim * video_num_heads,
            dropout=dropout,
            activation='gelu',
            batch_first=True # Crucial: Expects input as (Batch, Seq, Feature)
        )
        self.video_transformer = nn.TransformerEncoder(video_encoder_layer, num_layers=video_num_layers)

        # And then at the end of the video transformer we project into the contrastive space
        self.video_contrastive_projection = nn.Linear(video_transformer_dim, video_contrastive_dim)

    def embed_frames(self, video_embeddings: list[list[torch.Tensor]], device: str) -> ReIDFrameOutput:
        # In order to efficiently pass these through the transformer, we flatten 2d list into a jagged 1D list
        # and pad to be a consistent length
        video_indices: list[int] = []
        frame_embeddings_list: list[torch.Tensor] = []
        for video_index in range(len(video_embeddings)):
            frame_embeddings = video_embeddings[video_index]
            video_indices.extend([video_index for _ in range(len(frame_embeddings))])
            
            for embed in frame_embeddings:
                if embed.size(0) > self.max_frame_patches:
                    # Uniformly subsample patches to prevent OOM
                    indices = torch.linspace(0, embed.size(0) - 1, steps=self.max_frame_patches, device=embed.device).long()
                    embed = embed[indices]
                frame_embeddings_list.append(embed)
                
        total_videos = len(video_embeddings)
        padded_frame_embeddings = pad_sequence(frame_embeddings_list, batch_first=True)
        total_frames = padded_frame_embeddings.size(0)
        # This is now (total_frames, max_len, input_dim)

        # We also need a mask for the attention to ignore the padding
        lengths = torch.tensor([e.size(0) for e in frame_embeddings_list], device=device)
        max_len = lengths.max()
        frame_mask = torch.arange(max_len, device=device).expand(len(padded_frame_embeddings), max_len) >= lengths.unsqueeze(1)

        # We now have what we need to run the frame level transformer
        projected_frame_embeddings = self.input_projection(padded_frame_embeddings)
        # This is now (total_frames, max_len, frame_transformer_dim)

        # Prepend [CLS] token to every sequence in the batch
        cls_tokens = self.frame_cls_token.expand(total_frames, -1, -1) # (total_frames, 1, transformer_dim)
        frame_transformer_input = torch.cat((cls_tokens, projected_frame_embeddings), dim=1) # (total_frames, max_len + 1, frame_transformer_dim)

        # The [CLS] token at index 0 is always valid, so we prepend False
        cls_mask = torch.zeros((total_frames, 1), dtype=torch.bool, device=device)
        frame_padding_mask = torch.cat((cls_mask, frame_mask), dim=1) # (total_frames, max_len + 1)

        # frame_transformer_out = self.frame_transformer(frame_transformer_input, src_key_padding_mask=frame_padding_mask)
        frame_transformer_out = frame_transformer_input
        
        # Ensure requires_grad is True to trigger the backward pass properly
        if not frame_transformer_out.requires_grad:
            frame_transformer_out.requires_grad_(True)
            
        # Iterate through the internal layers of the TransformerEncoder
        for layer in self.frame_transformer.layers:
            frame_transformer_out = checkpoint(
                layer,
                frame_transformer_out,
                use_reentrant=False,
                src_key_padding_mask=frame_padding_mask
            )

        # Extract the state of the [CLS] token
        frame_cls_out = frame_transformer_out[:, 0, :] # (total_frames, frame_transformer_dim)

        # Now we project the class tokens into the contrastive space as well
        frame_contrastive_embeddings = self.frame_contrastive_projection(frame_cls_out)
        # This is (total_frames, frame_contrastive_dim)

        # And finally we re-package back into the original videos
        video_frame_contrastive_embeddings_list: list[list[torch.Tensor]] = [[] for _ in range(total_videos)]
        video_frame_cls_tokens_list: list[list[torch.Tensor]] = [[] for _ in range(total_videos)]
        for i in range(total_frames):
            video_index = video_indices[i]
            
            video_frame_contrastive_embeddings_list[video_index].append(
                frame_contrastive_embeddings[i]
            )

            video_frame_cls_tokens_list[video_index].append(
                frame_cls_out[i]
            )

        # Stack each
        video_frame_contrastive_embeddings = [torch.stack(frame_embeddings) for frame_embeddings in video_frame_contrastive_embeddings_list]
        video_frame_cls_tokens = [torch.stack(frame_embeddings) for frame_embeddings in video_frame_cls_tokens_list]
        # These are now in the same jagged shape as the original video_embeddings
        # (num_videos, frames_in_video, embedding_size) where frames_in_video may vary between videos

        return ReIDFrameOutput(
            video_frame_contrastive_embeddings = video_frame_contrastive_embeddings,
            video_frame_cls_tokens = video_frame_cls_tokens
        )

    def embed_video_frames(self, video_frame_cls_tokens: list[torch.Tensor], device: str) -> RIDVideoOutput:
        # video_frame_cls_tokens is (num_videos, frames_in_video, frame_transformer_dim)
        # We want to pad so that the input is of size (num_videos, max_frames_in_video, frame_transformer_dim)
        padded_video_frame_tokens = pad_sequence(video_frame_cls_tokens, batch_first=True)
        num_videos = len(video_frame_cls_tokens)

        # Like with the frame level we now needs to create a mask for the transformer
        lengths = torch.tensor([len(e) for e in video_frame_cls_tokens], device=device)
        max_len = lengths.max()
        frame_mask = torch.arange(max_len, device=device).expand(len(padded_video_frame_tokens), max_len) >= lengths.unsqueeze(1)

        # Now we can project from the frame transformer dimension to the video transformer dimension
        projected_embeddings = self.frame_to_video_projection(padded_video_frame_tokens)
        # This is now (num_videos, max_frames_in_video, video_transformer_dim)

        # Prepend the [CLS] token
        cls_tokens = self.video_cls_token.expand(num_videos, -1, -1)
        video_transformer_input = torch.cat((cls_tokens, projected_embeddings), dim=1)

        # The [CLS] token at index 0 is always valid, so we prepend False
        cls_mask = torch.zeros((num_videos, 1), dtype=torch.bool, device=device)
        frame_padding_mask = torch.cat((cls_mask, frame_mask), dim=1) # (total_frames, max_len + 1)

        video_transformer_out = self.video_transformer(video_transformer_input, src_key_padding_mask=frame_padding_mask)

        # Extract the class token embeddings
        video_cls_out = video_transformer_out[:, 0, :]  # (num_videos, video_transformer_dim)

        # And now we project into the contrastive space
        video_contrastive_embeddings = self.video_contrastive_projection(video_cls_out)
        # (num_videos, video_contrastive_dim)

        return RIDVideoOutput(
            video_contrastive_embeddings = video_contrastive_embeddings,
            video_cls_tokens = video_cls_out
        )

    def forward(self, video_embeddings: list[list[torch.Tensor]]) -> ReIDOutput:
        device = video_embeddings[0][0].device

        video_frame_transformer_output = self.embed_frames(video_embeddings, device)

        video_transformer_output = self.embed_video_frames(
            video_frame_transformer_output["video_frame_cls_tokens"],
            device
        )
        
        out = ReIDOutput(
            **video_frame_transformer_output,
            **video_transformer_output
        )

        return out
