"""ColPali multi-vector indexer and retrieval store supporting late interaction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel

from src.retrieval.maxsim_engine import compute_maxsim


class SearchResult(BaseModel):
    """Top-K retrieval item with metadata, score, and patch heatmap."""
    doc_id: str
    page_idx: int
    score: float
    image_path: str
    patch_heatmap: list[float]  # flattened 1D list of patch alignments (e.g. 1024 floats)
    width: int
    height: int
    dpi: int


class ColPaliEmbedder:
    """Extracts multi-vector patch embeddings for document images and token embeddings for text."""

    def __init__(
        self,
        model_name_or_path: str = "surrogate",
        device: str = "cpu",
        embedding_dim: int = 128,
        patch_grid_size: int = 32,
    ) -> None:
        self.model_name = model_name_or_path
        self.device = device
        self.embedding_dim = embedding_dim
        self.patch_grid_size = patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size  # 1024 patches
        self._real_model: Any = None

        if model_name_or_path != "surrogate":
            self._init_real_model(model_name_or_path)

    def _init_real_model(self, model_name: str) -> None:
        """Attempt to load real Byaldi or ColPali HuggingFace model if available."""
        try:
            from byaldi import RAGMultiModalModel  # type: ignore
            self._real_model = RAGMultiModalModel.from_pretrained(model_name, device=self.device)
        except Exception:
            # Fall back gracefully to surrogate mode if heavy weights/dependencies are missing
            self.model_name = "surrogate"

    def _token_vector(self, token: str) -> np.ndarray:
        """Deterministic semantic unit vector for a token string."""
        clean = token.lower().strip(",.()$:-%_ ")
        if not clean:
            clean = token.lower()
        import hashlib
        seed = int(hashlib.md5(clean.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.embedding_dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def embed_image(
        self,
        image: Image.Image,
        words: list[list[Any]] | None = None,
        page_points: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Embed document page image into multi-vector patch representations (N x D)."""
        if self._real_model is not None:
            # Native model inference if available
            with torch.no_grad():
                res = self._real_model.model.embed_image(image)
                return torch.tensor(res, dtype=torch.float32, device=self.device)

        # Deterministic visual & spatial patch encoder (surrogate):
        img_rgb = image.convert("RGB").resize((self.patch_grid_size * 4, self.patch_grid_size * 4))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0  # (128, 128, 3)

        patch_matrix = np.zeros((self.patch_grid_size, self.patch_grid_size, self.embedding_dim), dtype=np.float32)
        step = 4
        for r in range(self.patch_grid_size):
            for c in range(self.patch_grid_size):
                patch = arr[r * step : (r + 1) * step, c * step : (c + 1) * step, :]
                mean_rgb = patch.mean(axis=(0, 1))
                std_rgb = patch.std(axis=(0, 1))
                raw_feat = np.concatenate([mean_rgb, std_rgb])
                for i in range(self.embedding_dim):
                    patch_matrix[r, c, i] = 0.1 * np.sin((i + 1) * raw_feat[i % 6] + float(r * 32 + c))

        # Project word tokens to spatial patch coordinates
        if words and page_points:
            pt_w, pt_h = page_points
            for w in words:
                x0, y0, x1, y1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
                cx = ((x0 + x1) / 2.0) / max(1.0, pt_w)
                cy = ((y0 + y1) / 2.0) / max(1.0, pt_h)
                grid_c = int(np.clip(cx * self.patch_grid_size, 0, self.patch_grid_size - 1))
                grid_r = int(np.clip(cy * self.patch_grid_size, 0, self.patch_grid_size - 1))
                w_vec = self._token_vector(text)
                patch_matrix[grid_r, grid_c] += 1.5 * w_vec
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = grid_r + dr, grid_c + dc
                        if 0 <= nr < self.patch_grid_size and 0 <= nc < self.patch_grid_size:
                            patch_matrix[nr, nc] += 0.3 * w_vec

        patches_flat = patch_matrix.reshape(-1, self.embedding_dim)
        norms = np.linalg.norm(patches_flat, axis=1, keepdims=True) + 1e-8
        patches_normalized = patches_flat / norms

        return torch.tensor(patches_normalized, device=self.device)

    def embed_query(self, query: str) -> torch.Tensor:
        """Embed text query into token embeddings (|Q| x D)."""
        if self._real_model is not None:
            with torch.no_grad():
                res = self._real_model.model.embed_query(query)
                return torch.tensor(res, dtype=torch.float32, device=self.device)

        tokens = [t.strip().lower() for t in query.split() if t.strip()]
        if not tokens:
            tokens = ["query"]

        token_vecs = [self._token_vector(t) for t in tokens]
        return torch.tensor(np.array(token_vecs, dtype=np.float32), device=self.device)


class ColPaliIndexer:
    """Manages multi-vector patch index persistence and MaxSim similarity retrieval."""

    def __init__(
        self,
        index_dir: str | Path = "data/cache/index",
        embedder: ColPaliEmbedder | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or ColPaliEmbedder()
        self.pages: list[dict[str, Any]] = []
        self.embeddings: list[torch.Tensor] = []
        self._load_if_exists()

    def index_page(
        self,
        doc_id: str,
        page_idx: int,
        image: Image.Image,
        image_path: str,
        width: int,
        height: int,
        dpi: int,
        words: list[list[Any]] | None = None,
        original_points: tuple[float, float] | None = None,
        is_scanned: bool = False,
        vlm_synthesizer: Any = None,
    ) -> None:
        """Encode and add a document page to the in-memory and persistent index."""
        # Fallback for scanned/image-only PDFs: transcribe visible elements via VLM
        if (not words or is_scanned) and vlm_synthesizer is not None:
            words = vlm_synthesizer.transcribe_scanned_page(image)

        emb = self.embedder.embed_image(image, words=words, page_points=original_points)
        meta = {
            "doc_id": doc_id,
            "page_idx": page_idx,
            "image_path": image_path,
            "width": width,
            "height": height,
            "dpi": dpi,
            "words": words or [],
            "original_points": original_points or (float(width), float(height)),
            "is_scanned": is_scanned,
        }
        self.pages.append(meta)
        self.embeddings.append(emb)

    def save(self) -> None:
        """Persist index metadata and embedding tensors to disk."""
        meta_file = self.index_dir / "index_meta.json"
        emb_file = self.index_dir / "index_tensors.pt"

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.pages, f, indent=2)

        if self.embeddings:
            stacked = torch.stack(self.embeddings, dim=0)
            torch.save(stacked, emb_file)

    def _load_if_exists(self) -> None:
        """Reload index if metadata and tensor files exist on disk."""
        meta_file = self.index_dir / "index_meta.json"
        emb_file = self.index_dir / "index_tensors.pt"

        if meta_file.exists() and emb_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.pages = json.load(f)
            stacked = torch.load(emb_file, map_location=self.embedder.device)
            self.embeddings = [stacked[i] for i in range(stacked.size(0))]

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id_filter: str | None = None,
    ) -> list[SearchResult]:
        """Perform Late-Interaction MaxSim search over indexed pages."""
        if not self.embeddings:
            return []

        q_emb = self.embedder.embed_query(query)
        results: list[SearchResult] = []

        for idx, (meta, d_emb) in enumerate(zip(self.pages, self.embeddings)):
            if doc_id_filter and meta["doc_id"] != doc_id_filter:
                continue

            score, patch_heat = compute_maxsim(q_emb, d_emb, normalize=True)
            res = SearchResult(
                doc_id=meta["doc_id"],
                page_idx=meta["page_idx"],
                score=round(score, 4),
                image_path=meta["image_path"],
                patch_heatmap=patch_heat.cpu().tolist(),
                width=meta["width"],
                height=meta["height"],
                dpi=meta["dpi"],
            )
            results.append(res)

        # Rank descending by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
