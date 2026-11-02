# Implementation Plan & Milestone Roadmap
## DocuVision-ColPali: Multimodal Visual Document RAG

---

### Phase 1: Environment Setup & Document Rasterization Pipeline (Days 1–2)
- [ ] Initialize Python 3.11 virtual environment with `uv`.
- [ ] Implement `src/ingestion/rasterizer.py`:
  - Read arbitrary multi-page PDF documents.
  - Render each page to an uncompressed RGB PIL Image at 150 DPI and 300 DPI.
  - Store rendered page images in structured directory cache (`data/cache/pages/{doc_id}/page_{idx}.png`).
  - Unit tests: verify rasterization fidelity, DPI scaling, and memory cleanup.

---

### Phase 2: ColPali Multi-Vector Indexing & Late Interaction Retrieval (Days 3–5)
- [ ] Integrate ColPali / ColQwen2 backbone via `byaldi` or native Hugging Face transformers.
- [ ] Implement `src/retrieval/colpali_indexer.py`:
  - Batch extract multi-vector patch embeddings ($1024 \times 128$ dimensions per page).
  - Build persistent vector index on disk with Faiss/HNSW backend.
- [ ] Implement `src/retrieval/maxsim_engine.py`:
  - Tokenize query and encode text tokens into identical embedding space.
  - Compute vectorized MaxSim similarity score.
  - Return Top-$K$ ranked pages with patch-level similarity heatmaps.
- [ ] Benchmark retrieval latency and memory consumption across varying batch sizes.

---

### Phase 3: Spatial Bounding-Box Grounding & Visual Citation (Days 6–7)
- [ ] Implement `src/grounding/heatmap_to_bbox.py`:
  - Reshape 1D patch similarity scores to 2D topological matrix $(32 \times 32)$.
  - Apply Gaussian smoothing, adaptive thresholding, and morphological closing.
  - Extract bounding boxes covering high-attention regions ($> 85\text{th percentile}$).
  - Map normalized coordinates back to source PDF DPI coordinates.
- [ ] Implement visual overlay generator:
  - Draw semi-transparent bounding box overlays on page images using Pillow/OpenCV.
  - Save visual citation artifact (`reports/citations/query_{id}_page_{num}.jpg`).

---

### Phase 4: VLM Synthesis & Grounded Structured Reasoning (Days 8–9)
- [ ] Implement `src/generation/vlm_synthesizer.py`:
  - Construct prompt containing user query, retrieved page image(s), and cropped focus regions.
  - Invoke Vision-Language Model (Qwen2-VL local quantized or API).
  - Enforce strict JSON schema: `answer`, `confidence`, `supporting_pages`, `bounding_boxes`.
  - Add refusal logic: if visual evidence is absent, respond with `NOT_FOUND_IN_DOCUMENT`.

---

### Phase 5: FastAPI Production Service & Streamlit Interactive UI (Days 10–11)
- [ ] Implement `src/api/main.py`:
  - `POST /api/v1/documents/upload`: Upload PDF and trigger asynchronous rasterization & indexing.
  - `POST /api/v1/query`: Execute ColPali retrieval + VLM generation + visual citation.
  - `GET /api/v1/documents/{doc_id}/pages/{page_idx}`: Stream raw or annotated page image.
- [ ] Implement `src/ui/app.py` in Streamlit:
  - Drag-and-drop PDF upload area.
  - Interactive query prompt.
  - Split-screen UI: Left side shows conversational answer with citations; right side shows original PDF page with dynamic red bounding-box highlight.

---

### Phase 6: ViDoRe Evaluation & Performance Audit (Days 12–13)
- [ ] Run benchmark against a subset of the **ViDoRe (Visual Document Retrieval)** financial and technical test split.
- [ ] Measure NDCG@5, Recall@5, Mean Reciprocal Rank (MRR), and latency vs traditional Text-RAG (LangChain + Chunking + BGE-M3).
- [ ] Compile comprehensive audit report in `reports/VIDORE_BENCHMARK_REPORT.md`.
