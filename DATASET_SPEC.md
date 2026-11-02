# Dataset Specification & Ingestion Pipeline
## DocuVision-ColPali: Multimodal Visual Document RAG

---

### 1. Document Taxonomy & Ingestion Corpora
The platform is designed to handle three distinct categories of enterprise document layouts:

| Category | Typical Layout Features | Failure Mode in Text RAG | Evaluation Target |
| :--- | :--- | :--- | :--- |
| **Financial SEC 10-K & Annual Reports** | Multi-column nested tables, consolidated balance sheets, segment earnings footnotes. | Column header alignment lost; numbers disassociated from fiscal years. | Exact numeric cell extraction ($\ge 98\%$). |
| **Industrial Equipment & Engineering Blueprints** | Circuit diagrams, flowcharts, exploded mechanical views, component callouts. | Discarded as empty images or non-text blocks. | Flow connection & pinout retrieval ($\ge 92\%$). |
| **Scientific & Medical Publications** | Two-column academic formatting, ROC curves, bar charts with error bars, LaTeX equations. | Equation mangling; visual charts stripped completely. | Chart trend interpretation ($\ge 90\%$). |

---

### 2. Standard Benchmark Datasets Included
1. **ViDoRe Benchmark Split (Visual Document Retrieval):**
   - Source: Hugging Face `vidore/vidore-benchmark`
   - Splits: `syntheticDocQA_energy`, `shiftproject_reports`, `docvqa_test`
   - Ground truth: Query-to-Page-Index relevance pairs and bounding boxes.
2. **SEC 10-K Multi-Table Test Pack:**
   - 10-K filings from Apple (AAPL), Microsoft (MSFT), and Tesla (TSLA).
   - Manually annotated queries targeting footnotes, revenue bridges, and capital expenditures.

---

### 3. Image Preprocessing & Normalization Standards
- **Rasterization DPI:** Default 150 DPI ($1240 \times 1754$ pixels for A4). Optional 300 DPI ($2480 \times 3508$ pixels) for dense tabular sheets.
- **Color Space:** 24-bit RGB (sRGB profile).
- **Interpolation:** Lanczos resampler for downscaling or patch mapping.
- **Aspect Ratio:** Preserved with white padding if fed into fixed-dimension ViT architectures.
