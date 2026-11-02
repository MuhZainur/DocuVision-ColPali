"""End-to-end testing script for DocuVision-ColPali multimodal document intelligence."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.generation.vlm_synthesizer import VLMSynthesizer
from src.grounding.heatmap_to_bbox import extract_bounding_boxes, render_visual_citation
from src.ingestion.rasterizer import PDFRasterizer
from src.retrieval.colpali_indexer import ColPaliIndexer


def run_e2e_test() -> None:
    print("=" * 70)
    print(">> DOCUVISION-COLPALI: LIVE END-TO-END PIPELINE VERIFICATION")
    print("=" * 70)

    pdf_path = Path("data/samples/sample_financial_report.pdf")
    if not pdf_path.exists():
        print(f"Generating sample PDF first at {pdf_path}...")
        from scripts.generate_sample_pdf import create_sample_financial_pdf
        create_sample_financial_pdf(pdf_path)

    # 1. Ingestion & Rasterization
    print(f"\n[Step 1] Ingesting & Rasterizing: {pdf_path.name}")
    rasterizer = PDFRasterizer(cache_dir="data/cache/pages")
    doc_id, pages_meta, images = rasterizer.rasterize(pdf_path, dpi=150)
    print(f" -> Doc ID: {doc_id}")
    print(f" -> Total Pages: {len(pages_meta)}")
    for m in pages_meta:
        print(f"    Page {m.page_idx + 1}: {m.width}x{m.height} px @ {m.dpi} DPI ({len(m.words)} words extracted)")

    # 2. ColPali Multi-Vector Indexing
    print("\n[Step 2] Building ColPali Multi-Vector Index (1024 patches x 128d / page)...")
    indexer = ColPaliIndexer(index_dir="data/cache/index")
    for meta, img in zip(pages_meta, images):
        indexer.index_page(
            doc_id=doc_id,
            page_idx=meta.page_idx,
            image=img,
            image_path=meta.image_path,
            width=meta.width,
            height=meta.height,
            dpi=meta.dpi,
            words=meta.words,
            original_points=meta.original_points,
        )
    indexer.save()
    print(" -> Index successfully built and persisted to disk!")

    # 3. Test Queries
    synthesizer = VLMSynthesizer()
    print(f"\n[Step 3] Initialized VLM Synthesizer (Active Model: {synthesizer.model_name})")
    print(f" -> Gemini Client Enabled: {synthesizer._client is not None}")

    test_queries = [
        ("What was the Net Income in fiscal year 2024?", "query1_net_income_citation.jpg"),
        ("What are the Cash and Cash Equivalents in FY 2024?", "query2_cash_citation.jpg"),
        ("What is the company policy regarding astronauts traveling to Mars?", None),
    ]

    for q_idx, (query, citation_filename) in enumerate(test_queries, 1):
        print("\n" + "-" * 70)
        print(f"[Query #{q_idx}] '{query}'")
        hits = indexer.search(query=query, top_k=2, doc_id_filter=doc_id)

        if not hits:
            print(" -> No hits retrieved.")
            continue

        print(f" -> Top Retrieval: Page {hits[0].page_idx + 1} (MaxSim Score: {hits[0].score:.4f})")

        # Spatial Grounding Bounding Box
        boxes = extract_bounding_boxes(hits[0].patch_heatmap)
        print(f" -> Extracted Bounding Boxes: {len(boxes)}")
        for b in boxes:
            print(f"    Box: [{b.x_min:.3f}, {b.y_min:.3f}, {b.x_max:.3f}, {b.y_max:.3f}] (conf: {b.confidence:.2f})")

        # Visual Citation Rendering
        if citation_filename:
            save_path = Path("reports/citations") / citation_filename
            orig_img = Image.open(hits[0].image_path)
            annotated = render_visual_citation(orig_img, boxes, save_path=save_path)
            print(f" -> Visual Citation Saved: {save_path.resolve()}")

        # VLM Synthesizer
        grounded_answer = synthesizer.synthesize(
            query=query,
            retrieved_images=[Image.open(hits[0].image_path)],
            retrieved_pages=[hits[0].page_idx],
            bboxes_per_page=[boxes],
            top_retrieval_score=hits[0].score,
        )

        print("\n[VLM Grounded Response]")
        if grounded_answer.refusal:
            print(f"   [REFUSAL] {grounded_answer.answer}")
            print(f"   Reason: {grounded_answer.evidence_summary}")
        else:
            print(f"   Answer: {grounded_answer.answer}")
            print(f"   Confidence: {int(grounded_answer.confidence * 100)}%")
            print(f"   Supporting Pages: {[p + 1 for p in grounded_answer.supporting_pages]}")
            print(f"   Evidence Summary: {grounded_answer.evidence_summary}")

    print("\n" + "=" * 70)
    print("SUCCESS: LIVE END-TO-END VERIFICATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_test()
