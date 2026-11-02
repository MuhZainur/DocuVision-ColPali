"""Verification for pure scanned/image-only PDFs (addressing the OCR/scanned skip)."""

import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz

from src.generation.vlm_synthesizer import VLMSynthesizer
from src.grounding.heatmap_to_bbox import extract_bounding_boxes, render_visual_citation
from src.ingestion.rasterizer import PDFRasterizer
from src.retrieval.colpali_indexer import ColPaliIndexer


def create_scanned_image_pdf(output_pdf: Path) -> Path:
    """Generate a PDF containing ONLY an uncompressed pixel image (zero PDF text)."""
    # 1. Draw receipt image with Pillow
    img = Image.new("RGB", (600, 800), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    # Header
    draw.rectangle([20, 20, 580, 80], fill=(40, 60, 100))
    draw.text((40, 40), "GLOBAL LOGISTICS - SCANNED AIR WAYBILL", fill=(255, 255, 255))

    # Content
    draw.text((40, 120), "Tracking Number: AWB-98234-EXP", fill=(20, 20, 20))
    draw.text((40, 160), "Shipper: Tokyo Precision Robotics Ltd.", fill=(20, 20, 20))
    draw.text((40, 200), "Consignee: Jakarta AI Automation Labs", fill=(20, 20, 20))
    draw.text((40, 240), "Declared Cargo Value: $14,500 USD", fill=(20, 20, 20))
    draw.text((40, 280), "Customs Duty Status: PAID IN FULL", fill=(20, 120, 30))

    img_path = output_pdf.parent / "temp_scanned.png"
    img.save(img_path, format="PNG")

    # 2. Put only the image into PDF via PyMuPDF (no text insertion)
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_image(fitz.Rect(0, 0, 600, 800), filename=str(img_path))
    doc.save(str(output_pdf))
    doc.close()
    img_path.unlink(missing_ok=True)
    return output_pdf


def test_scanned_pdf() -> None:
    print("=" * 70)
    print(">> TEST: PURE SCANNED / IMAGE-ONLY PDF HANDLING")
    print("=" * 70)

    pdf_path = Path("data/samples/scanned_air_waybill.pdf")
    create_scanned_image_pdf(pdf_path)

    # 1. Rasterize
    rasterizer = PDFRasterizer(cache_dir="data/cache/pages")
    doc_id, pages_meta, images = rasterizer.rasterize(pdf_path)
    meta = pages_meta[0]
    print(f" -> Scanned Page Detected: is_scanned={meta.is_scanned} (Word Count={len(meta.words)})")
    assert meta.is_scanned is True, "Should detect 0 words and mark is_scanned=True"

    # 2. Index with VLM Scanned Transcriber
    synthesizer = VLMSynthesizer()
    indexer = ColPaliIndexer(index_dir="data/cache/index_scanned")
    indexer.index_page(
        doc_id=doc_id,
        page_idx=meta.page_idx,
        image=images[0],
        image_path=meta.image_path,
        width=meta.width,
        height=meta.height,
        dpi=meta.dpi,
        words=meta.words,
        original_points=meta.original_points,
        is_scanned=meta.is_scanned,
        vlm_synthesizer=synthesizer,
    )
    indexer.save()

    indexed_page = indexer.pages[0]
    print(f" -> Transcribed Visual Tokens by VLM: {len(indexed_page.get('words', []))} tokens")

    # 3. Query
    query = "What is the Tracking Number and Declared Cargo Value on the waybill?"
    print(f"\n[Query] '{query}'")
    hits = indexer.search(query=query, top_k=1, doc_id_filter=doc_id)
    top_hit = hits[0]
    print(f" -> Retrieval Score: {top_hit.score:.4f}")

    boxes = extract_bounding_boxes(top_hit.patch_heatmap)
    save_path = Path("reports/citations/scanned_waybill_citation.jpg")
    render_visual_citation(images[0], boxes, save_path=save_path)
    print(f" -> Visual Citation Saved: {save_path.resolve()}")

    answer = synthesizer.synthesize(
        query=query,
        retrieved_images=images,
        retrieved_pages=[0],
        bboxes_per_page=[boxes],
        top_retrieval_score=top_hit.score,
    )

    print("\n[VLM Grounded Reasoning Result]")
    print(f"   Answer:\n{answer.answer}")
    print(f"   Confidence: {int(answer.confidence * 100)}%")
    print("\n" + "=" * 70)
    print("SUCCESS: SCANNED PDF TEST COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    test_scanned_pdf()
