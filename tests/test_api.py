"""Integration tests for FastAPI endpoints."""

import io
from pathlib import Path
import fitz
from fastapi.testclient import TestClient
import pytest

from src.api.main import app

client = TestClient(app)


@pytest.fixture
def test_pdf_bytes() -> bytes:
    doc = fitz.open()
    p = doc.new_page(width=300, height=400)
    p.insert_text((40, 50), "Test PDF for DocuVision API", fontsize=14)
    p.insert_text((40, 100), "Quarterly Financial Overview", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_health_endpoint() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "indexed_pages" in data


def test_upload_and_query_flow(test_pdf_bytes: bytes) -> None:
    # 1. Upload
    files = {"file": ("report.pdf", test_pdf_bytes, "application/pdf")}
    upload_resp = client.post("/api/v1/documents/upload?dpi=100", files=files)
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    doc_id = upload_data["doc_id"]
    assert upload_data["num_pages"] == 1

    # 2. Query
    query_payload = {
        "query": "financial overview",
        "doc_id": doc_id,
        "top_k": 1,
    }
    q_resp = client.post("/api/v1/query", json=query_payload)
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert "grounded_answer" in q_data
    assert len(q_data["citations"]) >= 1

    # 3. Fetch Page Image
    img_resp = client.get(f"/api/v1/documents/{doc_id}/pages/0?annotated=true")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/jpeg"
