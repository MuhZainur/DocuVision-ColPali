# CLAUDE.md - Developer Guidelines for DocuVision-ColPali

## Project Overview
DocuVision-ColPali is an enterprise multimodal visual document intelligence platform utilizing ColPali (Vision-Language Late Interaction) for end-to-end PDF retrieval and visual bounding-box citation without traditional OCR text chunking.

## Development Commands
- Install dependencies: `uv pip install -e ".[dev]"`
- Run tests: `pytest tests/ -v`
- Run linting: `ruff check src/ tests/`
- Format code: `ruff format src/ tests/`
- Type checking: `mypy src/`
- Start API server: `uvicorn src.api.main:app --reload --port 8000`
- Launch Streamlit UI: `streamlit run src/ui/app.py`

## Code Style & Architecture Invariants
- Python 3.11+ with strict type annotations on all functions.
- Every retrieved document page must preserve original DPI metadata and uncompressed image tensor representation.
- MaxSim operations must be fully vectorized (avoid nested Python for-loops over patch indices).
- Bounding-box coordinates must always be normalized to $[0.0, 1.0]$ before being translated to physical PDF pixel dimensions.
- Never strip error handling around PDF rasterization: corrupt or password-protected PDF files must fail gracefully with structured error codes.
