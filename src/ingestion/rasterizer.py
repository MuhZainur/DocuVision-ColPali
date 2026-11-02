"""Document rasterization pipeline using PyMuPDF (fitz) with DPI scaling and caching."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field


class RasterizerError(Exception):
    """Base exception for PDF rasterization errors."""
    pass


class PDFNotFoundError(RasterizerError):
    """Raised when the specified PDF file cannot be found."""
    pass


class PDFEncryptedError(RasterizerError):
    """Raised when attempting to rasterize a password-protected PDF."""
    pass


class PDFCorruptedError(RasterizerError):
    """Raised when PDF bytes or file structure is corrupt."""
    pass


class PageMetadata(BaseModel):
    """Metadata describing a rasterized document page."""
    doc_id: str
    page_idx: int
    width: int
    height: int
    dpi: int
    image_path: str
    original_points: tuple[float, float] = Field(
        description="Physical (width, height) in 72-dpi PDF point space."
    )
    words: list[list[Any]] = Field(
        default_factory=list,
        description="Extracted word bounding boxes [x0, y0, x1, y1, word_text].",
    )
    is_scanned: bool = Field(
        default=False,
        description="True if the page contains no embedded text layer (scanned image or photo).",
    )


class PDFRasterizer:
    """High-fidelity PDF rasterizer utilizing PyMuPDF with structured caching."""

    def __init__(self, cache_dir: str | Path = "data/cache/pages") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_doc_id(file_bytes_or_path: bytes | Path | str) -> str:
        """Compute unique SHA-256 hash for document tracking."""
        if isinstance(file_bytes_or_path, (str, Path)):
            with open(file_bytes_or_path, "rb") as f:
                content = f.read()
        else:
            content = file_bytes_or_path
        return hashlib.sha256(content).hexdigest()[:16]

    def rasterize(
        self,
        pdf_source: str | Path | bytes,
        dpi: int = 150,
        doc_id: str | None = None,
    ) -> tuple[str, list[PageMetadata], list[Image.Image]]:
        """Rasterize PDF to uncompressed RGB PIL images with metadata.

        Args:
            pdf_source: File path or raw bytes of the PDF.
            dpi: Target rendering resolution (default 150 for standard, 300 for high-res).
            doc_id: Optional document identifier; defaults to SHA-256 hash.

        Returns:
            Tuple of (doc_id, list of PageMetadata, list of PIL Images).
        """
        if isinstance(pdf_source, (str, Path)):
            p = Path(pdf_source)
            if not p.is_file():
                raise PDFNotFoundError(f"PDF file not found: {p}")
            with open(p, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(pdf_source, bytes):
            raw_bytes = pdf_source
        else:
            raise RasterizerError(f"Unsupported pdf_source type: {type(pdf_source)}")

        if not raw_bytes.startswith(b"%PDF"):
            raise PDFCorruptedError("Provided input does not contain a valid %PDF header.")

        if doc_id is None:
            doc_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
        except Exception as e:
            raise PDFCorruptedError(f"Failed to open PDF stream: {e}") from e

        try:
            if doc.is_encrypted:
                raise PDFEncryptedError(f"PDF {doc_id} is password-protected and cannot be rasterized.")

            pages_meta: list[PageMetadata] = []
            images: list[Image.Image] = []

            target_dir = self.cache_dir / doc_id
            target_dir.mkdir(parents=True, exist_ok=True)

            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                rect = page.rect  # width, height in points (1/72 inch)
                pix = page.get_pixmap(matrix=matrix, alpha=False)

                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

                img_path = target_dir / f"page_{page_idx}.png"
                img.save(img_path, format="PNG")

                raw_words = page.get_text("words")
                extracted_words = [
                    [round(float(w[0]), 1), round(float(w[1]), 1), round(float(w[2]), 1), round(float(w[3]), 1), str(w[4])]
                    for w in raw_words
                ]
                is_scanned = len(extracted_words) == 0

                meta = PageMetadata(
                    doc_id=doc_id,
                    page_idx=page_idx,
                    width=pix.width,
                    height=pix.height,
                    dpi=dpi,
                    image_path=str(img_path.resolve()),
                    original_points=(float(rect.width), float(rect.height)),
                    words=extracted_words,
                    is_scanned=is_scanned,
                )
                pages_meta.append(meta)

            return doc_id, pages_meta, images
        finally:
            doc.close()
