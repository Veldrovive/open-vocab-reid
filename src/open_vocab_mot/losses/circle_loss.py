import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLossWithUnknowns(nn.Module):
    def __init__(self, m=0.25, gamma=256):
        """
        Circle Loss with a 3-tier relationship state: Positive, Negative, and Unknown.
        """
        super().__init__()
        self.m = m
        self.gamma = gamma
        
        # Optimums
        self.O_p = 1 + m
        self.O_n = -m
        
        # Margins
        self.Delta_p = 1 - m
        self.Delta_n = m

    def forward(self, embeddings: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings (torch.Tensor): Shape (N, D), where N is batch size, D is feature dim.
            pos_mask (torch.Tensor): Boolean mask of shape (N, N). True for positive pairs.
            neg_mask (torch.Tensor): Boolean mask of shape (N, N). True for negative pairs.
        """
        # 1. Normalize embeddings to unit length for Cosine Similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 2. Compute pairwise similarity matrix (N x N)
        sim_mat = torch.matmul(embeddings, embeddings.t())

        # 3. Calculate weighting factors (alpha)
        alpha_p = torch.clamp(self.O_p - sim_mat, min=0.0)
        alpha_n = torch.clamp(sim_mat - self.O_n, min=0.0)

        # 4. Calculate weighted similarities (logits)
        logit_p = -self.gamma * alpha_p * (sim_mat - self.Delta_p)
        logit_n = self.gamma * alpha_n * (sim_mat - self.Delta_n)

        # 5. Masking
        INF = 1e9
        logit_p_masked = torch.where(pos_mask, logit_p, torch.tensor(-INF, device=embeddings.device))
        logit_n_masked = torch.where(neg_mask, logit_n, torch.tensor(-INF, device=embeddings.device))

        # 6. LogSumExp aggregation per anchor
        lse_p = torch.logsumexp(logit_p_masked, dim=1)
        lse_n = torch.logsumexp(logit_n_masked, dim=1)

        # 7. Compute final loss
        loss_per_anchor = F.softplus(lse_p + lse_n)

        # 8. Filter out anchors that have NO valid positive AND negative pairs
        valid_anchors_mask = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)

        if valid_anchors_mask.sum() > 0:
            return loss_per_anchor[valid_anchors_mask].mean()
        else:
            return (embeddings * 0).sum()
