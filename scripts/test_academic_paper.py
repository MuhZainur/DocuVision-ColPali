"""Test DocuVision-ColPali against a real 12-page Springer academic research paper."""

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


def test_paper() -> None:
    paper_path = Path(r"C:\Users\acer\Downloads\s11554-026-01884-2 (1).pdf")
    if not paper_path.exists():
        print(f"Error: File not found at {paper_path}")
        return

    print("=" * 70)
    print(">> REAL-WORLD ACADEMIC PAPER TEST: RTNet (Springer Journal, 12 Pages)")
    print("=" * 70)

    # 1. Ingestion
    print(f"\n[1] Ingesting & Rasterizing 12-page PDF @ 150 DPI...")
    rasterizer = PDFRasterizer(cache_dir="data/cache/pages")
    doc_id, pages_meta, images = rasterizer.rasterize(paper_path, dpi=150)
    print(f" -> Doc ID: {doc_id}")
    print(f" -> Successfully rasterized {len(pages_meta)} pages.")

    # 2. Indexing
    print("\n[2] Indexing all 12 pages with ColPali multi-vector representation...")
    indexer = ColPaliIndexer(index_dir="data/cache/index_rtnet")
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
    print(" -> Index ready.")

    # 3. VLM Synthesizer
    synthesizer = VLMSynthesizer()
    print(f"\n[3] VLM Synthesizer initialized (Active model: {synthesizer.model_name})")

    queries = [
        (
            "What are the main branches in the RTNet architecture and how do they fuse RGB and Thermal features?",
            "rtnet_architecture_citation.jpg",
        ),
        (
            "What is the mIoU and inference speed FPS achieved by RTNet compared to other methods in the experimental table?",
            "rtnet_benchmark_table_citation.jpg",
        ),
    ]

    for q_idx, (query, citation_file) in enumerate(queries, 1):
        print("\n" + "-" * 70)
        print(f"[Query #{q_idx}] '{query}'")
        hits = indexer.search(query=query, top_k=2, doc_id_filter=doc_id)
        if not hits:
            print(" -> No hits retrieved.")
            continue

        top_hit = hits[0]
        print(f" -> Top Retrieved Page: Page {top_hit.page_idx + 1} (MaxSim Score: {top_hit.score:.4f})")

        # Grounding & Bounding Boxes
        boxes = extract_bounding_boxes(top_hit.patch_heatmap)
        print(f" -> Salient Bounding Boxes Detected: {len(boxes)}")
        for b in boxes:
            print(f"    Region Box: [{b.x_min:.3f}, {b.y_min:.3f}, {b.x_max:.3f}, {b.y_max:.3f}] (conf: {b.confidence:.2f})")

        # Render Visual Citation
        save_path = Path("reports/citations") / citation_file
        orig_img = Image.open(top_hit.image_path)
        render_visual_citation(orig_img, boxes, save_path=save_path)
        print(f" -> Visual Citation Overlay Saved: {save_path.resolve()}")

        # VLM Synthesizer
        answer = synthesizer.synthesize(
            query=query,
            retrieved_images=[orig_img],
            retrieved_pages=[top_hit.page_idx],
            bboxes_per_page=[boxes],
            top_retrieval_score=top_hit.score,
        )

        print("\n[VLM Grounded Reasoning Result]")
        if answer.refusal:
            print(f"   [REFUSAL] {answer.answer}")
            print(f"   Reason: {answer.evidence_summary}")
        else:
            print(f"   Answer:\n{answer.answer}")
            print(f"   Supporting Pages: {[p + 1 for p in answer.supporting_pages]}")
            print(f"   Confidence: {int(answer.confidence * 100)}%")

    print("\n" + "=" * 70)
    print("SUCCESS: REAL-WORLD PAPER TEST COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    test_paper()
