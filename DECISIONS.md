# Architectural Decision Records (ADR)
## DocuVision-ColPali

---

### ADR-001: Selection of ColPali (Late Interaction) over Standard VLM Dense Bi-Encoder
- **Date:** 2026-11-02
- **Context:** Standard dense multimodal embeddings (e.g., CLIP, SigLIP) condense an entire page into a single 768d vector, losing tabular cell boundaries and footnote details. Cross-encoders are too slow for real-time document search.
- **Decision:** Adopt ColPali multi-vector late interaction. Each page is represented by 1024 patch vectors, evaluated via the MaxSim operator.
- **Consequences:** Higher disk/RAM index footprint ($1024 \times 128$ bytes per page), but achieves state-of-the-art retrieval accuracy and enables native bounding box extraction from patch heatmaps.

### ADR-002: Byaldi / Faiss as Primary Multi-Vector Index Engine
- **Date:** 2026-11-02
- **Context:** Managing millions of patch vectors across thousands of pages requires an accelerated multi-vector index.
- **Decision:** Utilize `byaldi`, an open-source library built on top of RAGatouille and Faiss designed specifically for ColPali indexing.
- **Consequences:** Simplifies deployment with native Python bindings while providing sub-200ms retrieval.

### ADR-003: PyMuPDF (fitz) for PDF Rendering
- **Date:** 2026-11-02
- **Context:** Python PDF rasterization often relies on Poppler binaries which complicate cross-platform Windows/Linux deployments.
- **Decision:** Use PyMuPDF (`fitz`), which packages MuPDF directly in compiled C binaries with no external system library dependencies.
- **Consequences:** 4x faster page rasterization speed compared to pdf2image and zero external Poppler installation requirements.
