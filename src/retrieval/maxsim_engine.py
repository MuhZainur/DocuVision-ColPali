"""Vectorized MaxSim late interaction scoring and patch alignment engine."""

from __future__ import annotations

import torch
import numpy as np


def compute_maxsim(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    normalize: bool = True,
) -> tuple[float, torch.Tensor]:
    """Compute vectorized MaxSim late-interaction score between query and document.

    Args:
        query_embeddings: Tensor of shape (num_query_tokens, embedding_dim).
        doc_embeddings: Tensor of shape (num_doc_patches, embedding_dim).
        normalize: Whether to L2-normalize embeddings before dot product (cosine similarity).

    Returns:
        Tuple of:
            - scalar float score: sum over query tokens of max patch similarity.
            - patch_max_sims: 1D Tensor of shape (num_doc_patches,) representing
              the maximum alignment of each image patch across all query tokens.
    """
    if query_embeddings.ndim != 2 or doc_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D tensors, got query={query_embeddings.shape}, doc={doc_embeddings.shape}"
        )

    if normalize:
        q_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        d_norm = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
    else:
        q_norm = query_embeddings
        d_norm = doc_embeddings

    # Similarity matrix shape: (|Q|, N_patches)
    sim_matrix = torch.matmul(q_norm, d_norm.transpose(0, 1))

    # Token-wise max over patches -> shape (|Q|,)
    token_max_scores, _ = torch.max(sim_matrix, dim=1)
    total_score = float(torch.sum(token_max_scores).item())

    # Patch-wise max over query tokens -> shape (N_patches,)
    patch_max_sims, _ = torch.max(sim_matrix, dim=0)

    return total_score, patch_max_sims


def batch_maxsim(
    query_embeddings: torch.Tensor,
    doc_batch: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch compute MaxSim scores for multiple documents against a single query.

    Args:
        query_embeddings: Tensor of shape (num_query_tokens, embedding_dim).
        doc_batch: Tensor of shape (batch_size, num_doc_patches, embedding_dim).
        normalize: Whether to L2-normalize vectors.

    Returns:
        Tuple of:
            - scores: 1D Tensor of shape (batch_size,) with total MaxSim scores.
            - patch_heatmaps: Tensor of shape (batch_size, num_doc_patches) with patch alignments.
    """
    if doc_batch.ndim != 3:
        raise ValueError(f"Expected 3D tensor for doc_batch, got shape {doc_batch.shape}")

    if normalize:
        q_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        d_norm = torch.nn.functional.normalize(doc_batch, p=2, dim=-1)
    else:
        q_norm = query_embeddings
        d_norm = doc_batch

    # Sim shape: (batch_size, num_query_tokens, num_doc_patches)
    # q_norm: (1, Q, D), d_norm: (B, N, D)
    q_expanded = q_norm.unsqueeze(0).expand(doc_batch.size(0), -1, -1)
    sim_tensor = torch.bmm(q_expanded, d_norm.transpose(1, 2))

    # Score: max over patches (dim=2), sum over query tokens (dim=1) -> shape (B,)
    token_max, _ = torch.max(sim_tensor, dim=2)
    scores = torch.sum(token_max, dim=1)

    # Patch heatmaps: max over query tokens (dim=1) -> shape (B, N)
    patch_heatmaps, _ = torch.max(sim_tensor, dim=1)

    return scores, patch_heatmaps
