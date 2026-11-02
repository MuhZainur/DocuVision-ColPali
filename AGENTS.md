# AGENTS.md - Agent Roles & Operating Guidelines

## Agent Personas & Task Scopes

### 1. Document Ingestion Specialist (`@ingestion`)
- Scope: PDF parsing, rendering optimization, multi-resolution caching.
- Tooling: PyMuPDF, Pillow, pdf2image.
- Rules: Enforce strict memory bounds when rendering 100+ page documents to prevent host memory exhaustion.

### 2. Multi-Vector Retrieval Engineer (`@retrieval`)
- Scope: ColPali model initialization, Byaldi HNSW index creation, MaxSim scoring operator.
- Tooling: PyTorch, Byaldi, Transformers, Accelerate.
- Rules: Implement automatic fallback to CPU quantized inference when GPU VRAM is constrained.

### 3. Spatial Grounding & UI Engineer (`@grounding-ui`)
- Scope: Patch attention heatmaps, Otsu thresholding, contour detection, Streamlit split-screen interface.
- Tooling: OpenCV, NumPy, Streamlit, Matplotlib.
- Rules: Ensure bounding box coordinates accurately correspond to the user's display viewport.
