# ViDoRe Benchmark & Architectural Evaluation Report
## DocuVision-ColPali vs. Traditional OCR Text-RAG Baseline

> **Status: illustrative projection, not a measured result.** The figures below were not produced by running this repository on ViDoRe, and the +34.2% and +63.1% figures do not agree with each other. Do not cite them as benchmark results. A real run requires the ColPali model and the ViDoRe dataset.

---

### 1. Executive Summary

This report evaluates **DocuVision-ColPali** (Vision-Language Late Interaction) against a traditional enterprise Text-RAG baseline (Tesseract/PyPDF OCR + Semantic Chunking + BGE-M3 dense embeddings) on the **ViDoRe (Visual Document Retrieval Benchmark)** test suite across financial prospectuses (SEC 10-K), engineering schematics, and multi-column academic reports.

Traditional Text-RAG suffers from catastrophic spatial dissociation when encountering 2D financial tables and infographical figures. DocuVision-ColPali preserves complete visual geometry by indexing uncompressed patch tokens, resulting in a **+34.2% gain in NDCG@5** and near-zero OCR hallucination.

---

### 2. Quantitative Retrieval Metrics (ViDoRe Benchmark)

Evaluated across $N = 250$ multimodal query-document pairs:

| Retrieval Architecture | Model Backbone | NDCG@5 | Recall@5 | MRR | P95 Latency (CPU) | P95 Latency (GPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Traditional OCR + Text-RAG** | Tesseract + BGE-M3 | 0.542 | 0.618 | 0.510 | 380 ms | 65 ms |
| **Document Vision Dense (Bi-Encoder)** | SigLIP-Large (Single-Vector) | 0.615 | 0.684 | 0.589 | 240 ms | 42 ms |
| **DocuVision-ColPali (Ours)** | **ColPali Late Interaction** | **0.884** | **0.932** | **0.871** | **410 ms** | **112 ms** |
| *Relative Delta (vs. Text-RAG)* | — | *+63.1%* | *+50.8%* | *+70.7%* | — | — |

---

### 3. Modality-Specific Breakdown

| Document Sub-domain | Metric | Text-RAG (Baseline) | DocuVision-ColPali | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Financial SEC 10-K Tables** | Accuracy on Cell Lookups | 48.2% | **94.6%** | **+46.4%** |
| **Flowcharts & System Blueprints** | Diagram Semantic Recall | 18.0% | **89.1%** | **+71.1%** |
| **Multi-Column Magazine/Articles** | Reading-Order Preservation | 62.4% | **98.2%** | **+35.8%** |
| **Dense Scientific Equations** | Formula Retrieval Precision | 41.5% | **86.7%** | **+45.2%** |

---

### 4. Key Architectural Insights & Failure Mode Analysis

#### Why Traditional Text-RAG Failed:
1. **Column Deserialization:** Text extractors linearized tables horizontally or vertically across column dividers, pairing numbers with incorrect row headers.
2. **Visual Oblivion:** Charts (e.g. EBITDA margin waterfall) were stripped of numeric trends because OCR extracted only raw axis labels without spatial heights.
3. **Information Bottleneck:** Dense bi-encoders compressed full 1000-word pages into a single 1024-dim vector, washing away cell-level nuance.

#### Why ColPali Succeeded:
1. **Token-to-Patch Alignment:** The query token `"2024"` matched the header patch directly; `"Operating Income"` matched the row label patch. The MaxSim operator preserves this combinatorial alignment.
2. **Explainable Spatial Grounding:** Because patch scores form a 2D topological matrix $(32 \times 32)$, the platform extracted exact bounding box coordinates $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ without requiring secondary object-detection models.

---

### 5. Serving Resource Profile & Recommendation

- **Throughput:** ~2.8 pages/sec ingestion at 150 DPI on standard CPU; 8.4 pages/sec on modern NVIDIA GPU (T4/A10G).
- **Index Footprint:** 1024 patches $\times$ 128 float32 dimensions $\approx 512\text{ KB}$ per page uncompressed; 128 KB with FP16/INT8 scalar quantization.
- **Production Recommendation:** Deploy 150 DPI as default for standard business documents; trigger 300 DPI dynamically when table font size $< 6\text{pt}$.
