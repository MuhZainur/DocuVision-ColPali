"""Unit tests for PDF rasterization and error handling."""

from pathlib import Path
import fitz
import pytest
from src.ingestion.rasterizer import (
    PDFCorruptedError,
    PDFEncryptedError,
    PDFNotFoundError,
    PDFRasterizer,
)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a clean 2-page PDF in memory using fitz."""
    doc = fitz.open()
    # Page 1
    p1 = doc.new_page(width=300, height=400)
    p1.insert_text((50, 50), "DocuVision Page 1: Annual Financial Report 2024", fontsize=12)
    # Page 2
    p2 = doc.new_page(width=300, height=400)
    p2.insert_text((50, 50), "DocuVision Page 2: Income Statement Table", fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_rasterizer_valid_pdf(tmp_path: Path, sample_pdf_bytes: bytes) -> None:
    rasterizer = PDFRasterizer(cache_dir=tmp_path / "cache")
    doc_id, pages_meta, images = rasterizer.rasterize(sample_pdf_bytes, dpi=144)

    assert len(doc_id) == 16
    assert len(pages_meta) == 2
    assert len(images) == 2
    # At 144 DPI (2x 72 DPI), a 300x400 pt page renders to 600x800 px
    assert pages_meta[0].width == 600
    assert pages_meta[0].height == 800
    assert pages_meta[0].dpi == 144
    assert Path(pages_meta[0].image_path).exists()


def test_rasterizer_corrupt_bytes(tmp_path: Path) -> None:
    rasterizer = PDFRasterizer(cache_dir=tmp_path / "cache")
    corrupt_bytes = b"NOT_A_REAL_PDF_STREAM_DATA"
    with pytest.raises(PDFCorruptedError):
        rasterizer.rasterize(corrupt_bytes)


def test_rasterizer_missing_file(tmp_path: Path) -> None:
    rasterizer = PDFRasterizer(cache_dir=tmp_path / "cache")
    with pytest.raises(PDFNotFoundError):
        rasterizer.rasterize(tmp_path / "non_existent.pdf")
