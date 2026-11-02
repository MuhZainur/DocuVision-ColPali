"""Unit tests for vectorized MaxSim calculation and ColPali indexer."""

from pathlib import Path
from PIL import Image
import torch
import numpy as np
import pytest

from src.retrieval.maxsim_engine import batch_maxsim, compute_maxsim
from src.retrieval.colpali_indexer import ColPaliIndexer, ColPaliEmbedder


def test_compute_maxsim_vectorized() -> None:
    # 4 query tokens, 16 patch tokens, 32-dim embeddings
    q_emb = torch.randn(4, 32)
    d_emb = torch.randn(16, 32)

    score, patch_heat = compute_maxsim(q_emb, d_emb, normalize=True)

    assert isinstance(score, float)
    assert patch_heat.shape == (16,)
    # Similarity values after normalization should be within [-1.0, 1.0]
    assert patch_heat.min().item() >= -1.0
    assert patch_heat.max().item() <= 1.0


def test_batch_maxsim() -> None:
    # 3 query tokens, batch of 5 documents with 16 patches each
    q_emb = torch.randn(3, 32)
    doc_batch = torch.randn(5, 16, 32)

    scores, heatmaps = batch_maxsim(q_emb, doc_batch, normalize=True)

    assert scores.shape == (5,)
    assert heatmaps.shape == (5, 16)


def test_colpali_indexer_flow(tmp_path: Path) -> None:
    embedder = ColPaliEmbedder(embedding_dim=64, patch_grid_size=8)
    indexer = ColPaliIndexer(index_dir=tmp_path / "index", embedder=embedder)

    img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))

    indexer.index_page("doc1", 0, img1, str(tmp_path / "p1.png"), 100, 100, 150)
    indexer.index_page("doc1", 1, img2, str(tmp_path / "p2.png"), 100, 100, 150)
    indexer.save()

    # Verify reloading
    reloaded_indexer = ColPaliIndexer(index_dir=tmp_path / "index", embedder=embedder)
    assert len(reloaded_indexer.pages) == 2
    assert len(reloaded_indexer.embeddings) == 2

    # Verify search
    hits = reloaded_indexer.search("financial summary table", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert len(hits[0].patch_heatmap) == 64
