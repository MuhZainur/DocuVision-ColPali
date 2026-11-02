# Graph Report - Multimodal_VLM_Doc_Intelligence  (2026-11-02)

## Corpus Check
- Corpus is ~8,950 words - fits in a single context window. You may not need a graph.

## Summary
- 164 nodes · 309 edges · 16 communities (6 shown, 10 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- PDF Rasterization & Error Handling
- Spatial Grounding & Bounding Box Detection
- FastAPI REST Service & Endpoints
- ColPali Multi-Vector Embeddings
- VLM Synthesizer & Schema Models
- HTTP Client & API Integration Tests
- Streamlit UI & Presentation Layer
- Project Metadata & Configuration

## God Nodes (most connected - your core abstractions)
1. `BoundingBox` - 15 edges
2. `extract_bounding_boxes()` - 13 edges
3. `PDFRasterizer` - 12 edges
4. `ColPaliIndexer` - 12 edges
5. `execute_query()` - 10 edges
6. `render_visual_citation()` - 10 edges
7. `compute_maxsim()` - 10 edges
8. `VLMSynthesizer` - 9 edges
9. `ColPaliEmbedder` - 9 edges
10. `upload_document()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `test_rasterizer_missing_file()` --uses--> `PDFNotFoundError`  [INFERRED]
  tests/test_rasterizer.py → src/ingestion/rasterizer.py
- `test_rasterizer_corrupt_bytes()` --uses--> `PDFCorruptedError`  [INFERRED]
  tests/test_rasterizer.py → src/ingestion/rasterizer.py
- `CitationDetail` --uses--> `BoundingBox`  [INFERRED]
  src/api/main.py → src/grounding/heatmap_to_bbox.py
- `upload_document()` --uses--> `PDFCorruptedError`  [INFERRED]
  src/api/main.py → src/ingestion/rasterizer.py
- `upload_document()` --uses--> `PDFEncryptedError`  [INFERRED]
  src/api/main.py → src/ingestion/rasterizer.py

## Import Cycles
- None detected.

## Communities (16 total, 10 thin omitted)

### Community 0 - "PDF Rasterization & Error Handling"
Cohesion: 0.12
Nodes (9): PDFCorruptedError, PDFEncryptedError, PDFNotFoundError, PDFRasterizer, RasterizerError, sample_pdf_bytes(), test_rasterizer_corrupt_bytes(), test_rasterizer_missing_file() (+1 more)

### Community 1 - "Spatial Grounding & Bounding Box Detection"
Cohesion: 0.14
Nodes (6): BoundingBox, extract_bounding_boxes(), render_visual_citation(), test_bbox_physical_conversion(), test_extract_bounding_boxes_grid(), test_render_visual_citation()

### Community 2 - "FastAPI REST Service & Endpoints"
Cohesion: 0.12
Nodes (9): CitationDetail, DocumentUploadResponse, execute_query(), QueryRequest, QueryResponse, upload_document(), GroundedAnswer, PageMetadata (+1 more)

### Community 3 - "ColPali Multi-Vector Embeddings"
Cohesion: 0.11
Nodes (3): ColPaliEmbedder, ColPaliIndexer, test_colpali_indexer_flow()

### Community 4 - "VLM Synthesizer & Schema Models"
Cohesion: 0.18
Nodes (4): batch_maxsim(), compute_maxsim(), test_batch_maxsim(), test_compute_maxsim_vectorized()

### Community 5 - "HTTP Client & API Integration Tests"
Cohesion: 0.13
Nodes (3): get_page_image(), health(), test_pdf_bytes()

## Knowledge Gaps
- **1 isolated node(s):** `docuvision-colpali`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 81 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ColPaliIndexer` connect `ColPali Multi-Vector Embeddings` to `Spatial Grounding & Bounding Box Detection`, `FastAPI REST Service & Endpoints`, `VLM Synthesizer & Schema Models`, `Streamlit UI & Presentation Layer`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `BoundingBox` (e.g. with `CitationDetail` and `execute_query()`) actually correct?**
  _`BoundingBox` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `docuvision-colpali` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `PDF Rasterization & Error Handling` be split into smaller, more focused modules?**
  _Cohesion score 0.11693548387096774 - nodes in this community are weakly interconnected._
- **Why does `ColPaliEmbedder` connect `ColPali Multi-Vector Embeddings` to `VLM Synthesizer & Schema Models`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `execute_query()` (e.g. with `GroundedAnswer` and `BoundingBox`) actually correct?**
  _`execute_query()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `Spatial Grounding & Bounding Box Detection` be split into smaller, more focused modules?**
  _Cohesion score 0.13538461538461538 - nodes in this community are weakly interconnected._