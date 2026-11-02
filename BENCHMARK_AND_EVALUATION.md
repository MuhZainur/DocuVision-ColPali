# Benchmark & Evaluation Methodology
## DocuVision-ColPali: Performance & Accuracy Evaluation

---

### 1. Benchmark Suites & Metrics
The system is evaluated across three primary evaluation dimensions:

1. **Retrieval Accuracy (ViDoRe Metrics):**
   - **NDCG@5 (Normalized Discounted Cumulative Gain):** Measures whether the most relevant visual pages are ranked at position 1.
   - **Recall@5:** Percentage of relevant pages present in the Top-5 retrieved candidates.
   - **MRR (Mean Reciprocal Rank):** Speed at which the first relevant document page appears.
2. **Spatial Grounding Accuracy:**
   - **IoU (Intersection-over-Union):** Overlap between generated bounding box and ground-truth table/chart region.
   - **Hit Rate @ 0.5 IoU:** Percentage of visual citations with $\text{IoU} \ge 0.5$.
3. **Generation Faithfulness & Hallucination Rate:**
   - Evaluated via Ragas Vision Triad:
     - *Visual Faithfulness:* Answer claims verified against page pixel evidence.
     - *Visual Answer Relevance:* Direct relevance of the answer to the visual query.

---

### 2. Baseline Comparison Protocol
We benchmark **DocuVision-ColPali** directly against the industry standard text baseline:
- **Baseline A (Standard Text RAG):** `PyPDF` $\to$ RecursiveCharacterTextSplitter (chunk size 500, overlap 50) $\to$ `BAAI/bge-m3` $\to$ Qdrant $\to$ GPT-4o-mini.
- **Baseline B (OCR-Enhanced RAG):** `Tesseract OCR / AWS Textract` $\to$ Table Markdown Parser $\to$ `text-embedding-3-large` $\to$ Claude 3.5 Sonnet.
- **DocuVision-ColPali (Our Pipeline):** 150 DPI Render $\to$ `ColPali-v1.2` Byaldi Index $\to$ MaxSim $\to$ Spatial Bounding-Box Overlay $\to$ Qwen2-VL-7B-Instruct.

Target Acceptance Criteria:
$$\text{NDCG@5}_{\text{ColPali}} \ge \text{NDCG@5}_{\text{Baseline A}} + 18.0\%$$
