"""Interactive Streamlit UI for DocuVision-ColPali multimodal document intelligence."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path for Streamlit Cloud and nested executions
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st
from PIL import Image

from src.generation.vlm_synthesizer import VLMSynthesizer
from src.grounding.heatmap_to_bbox import extract_bounding_boxes, render_visual_citation
from src.ingestion.rasterizer import PDFRasterizer
from src.retrieval.colpali_indexer import ColPaliIndexer

st.set_page_config(
    page_title="DocuVision-ColPali | Multimodal Document RAG",
    page_icon="👁️",
    layout="wide",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .badge-chip { background-color: #EEF2F6; color: #1E293B; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    </style>
    <div class="main-title">DocuVision-ColPali 👁️📄</div>
    <div class="sub-title">Enterprise Multimodal Visual Document Intelligence via Vision-Language Late Interaction</div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_services() -> tuple[PDFRasterizer, ColPaliIndexer, VLMSynthesizer]:
    """Initialize cached service instances."""
    rasterizer = PDFRasterizer(cache_dir="data/cache/pages")
    indexer = ColPaliIndexer(index_dir="data/cache/index")
    synthesizer = VLMSynthesizer()
    return rasterizer, indexer, synthesizer


rasterizer, indexer, synthesizer = get_services()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Document Ingestion")
    dpi_choice = st.select_slider(
        "Rendering Resolution (DPI)",
        options=[100, 150, 200, 300],
        value=150,
        help="Higher DPI improves OCR and micro-table fidelity at higher raster memory.",
    )
    top_k = st.slider("Top-K Retrieved Pages", min_value=1, max_value=5, value=2)

    st.markdown("---")
    st.subheader("Document Index Status")
    st.write(f"**Total Indexed Pages:** `{len(indexer.pages)}`")

    uploaded_file = st.file_uploader("Upload Target PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("🚀 Ingest & Index Document", use_container_width=True):
            with st.spinner("Rasterizing PDF with PyMuPDF and computing patch embeddings..."):
                file_bytes = uploaded_file.read()
                doc_id, pages_meta, images = rasterizer.rasterize(file_bytes, dpi=dpi_choice)
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
                        is_scanned=meta.is_scanned,
                        vlm_synthesizer=synthesizer,
                    )
                indexer.save()
                st.session_state["current_doc_id"] = doc_id
                st.success(f"Indexed {len(pages_meta)} page(s) successfully! (Doc ID: `{doc_id[:8]}`)")

# --- Main Split-Screen Workspace ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🔍 Visual Late Interaction Search")
    query_input = st.text_input(
        "Enter your query (e.g. 'What is the net revenue in 2024?', 'Summarize chart trends'):",
        placeholder="Type a multimodal or tabular question...",
    )
    search_clicked = st.button("Search & Reason", type="primary", use_container_width=True)

    if search_clicked and query_input.strip():
        curr_doc_id = st.session_state.get("current_doc_id")
        with st.spinner("Executing ColPali MaxSim retrieval and VLM synthesis..."):
            hits = indexer.search(query=query_input, top_k=top_k, doc_id_filter=curr_doc_id)

            if not hits:
                st.warning("No matching documents or pages found in the active index.")
            else:
                retrieved_images: list[Image.Image] = []
                retrieved_pages: list[int] = []
                bboxes_per_page = []

                for hit in hits:
                    img_p = Path(hit.image_path)
                    if img_p.exists():
                        img = Image.open(img_p)
                        retrieved_images.append(img)
                        retrieved_pages.append(hit.page_idx)
                        bboxes = extract_bounding_boxes(hit.patch_heatmap)
                        bboxes_per_page.append(bboxes)

                top_score = hits[0].score if hits else 0.0
                answer = synthesizer.synthesize(
                    query=query_input,
                    retrieved_images=retrieved_images,
                    retrieved_pages=retrieved_pages,
                    bboxes_per_page=bboxes_per_page,
                    top_retrieval_score=top_score,
                )

                st.session_state["last_query"] = query_input
                st.session_state["last_answer"] = answer
                st.session_state["last_hits"] = hits
                st.session_state["last_bboxes"] = bboxes_per_page

    # Display Answering & Evidence
    if "last_answer" in st.session_state:
        ans = st.session_state["last_answer"]
        st.markdown("### 🤖 Factual Grounded Answer")
        if ans.refusal:
            st.error(f"⚠️ **Refusal:** {ans.answer}")
            st.caption(ans.evidence_summary)
        else:
            st.info(ans.answer)
            st.markdown(
                f"<span class='badge-chip'>Confidence: {int(ans.confidence * 100)}%</span> "
                f"<span class='badge-chip'>Pages: {', '.join(str(p + 1) for p in ans.supporting_pages)}</span>",
                unsafe_allow_html=True,
            )
            st.caption(ans.evidence_summary)

        st.markdown("---")
        st.markdown("### 📑 Retrieved Citations")
        for idx, hit in enumerate(st.session_state.get("last_hits", [])):
            with st.expander(f"Rank #{idx+1} | Page {hit.page_idx + 1} (MaxSim Score: {hit.score})"):
                st.write(f"Image Resolution: `{hit.width}x{hit.height}` @ `{hit.dpi} DPI`")
                st.write(f"Source file: `{hit.image_path}`")

with col_right:
    st.subheader("🖼️ Visual Grounding Preview")
    if "last_hits" in st.session_state and st.session_state["last_hits"]:
        hits = st.session_state["last_hits"]
        bboxes_list = st.session_state.get("last_bboxes", [])

        # Page selector if multiple pages retrieved
        page_options = [f"Rank #{i+1} - Page {h.page_idx + 1}" for i, h in enumerate(hits)]
        selected_idx = st.selectbox("Inspect Citation Page:", range(len(page_options)), format_func=lambda i: page_options[i])

        chosen_hit = hits[selected_idx]
        chosen_bboxes = bboxes_list[selected_idx] if selected_idx < len(bboxes_list) else []

        img_path = Path(chosen_hit.image_path)
        if img_path.exists():
            orig_img = Image.open(img_path)
            show_overlay = st.checkbox("Show Bounding Box Highlights", value=True)
            if show_overlay and chosen_bboxes:
                annotated_img = render_visual_citation(orig_img, chosen_bboxes)
                st.image(annotated_img, caption=f"Page {chosen_hit.page_idx + 1} with Attention Bounding Boxes", use_container_width=True)
            else:
                st.image(orig_img, caption=f"Page {chosen_hit.page_idx + 1} (Raw Raster)", use_container_width=True)
        else:
            st.error(f"Image not found at {img_path}")
    else:
        st.info("Upload a document and run a query to inspect visual grounding and bounding boxes.")
