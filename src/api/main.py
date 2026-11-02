"""FastAPI production service for DocuVision-ColPali multimodal document intelligence."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from src.generation.vlm_synthesizer import GroundedAnswer, VLMSynthesizer
from src.grounding.heatmap_to_bbox import (
    BoundingBox,
    extract_bounding_boxes,
    render_visual_citation,
)
from src.ingestion.rasterizer import (
    PageMetadata,
    PDFCorruptedError,
    PDFEncryptedError,
    PDFNotFoundError,
    PDFRasterizer,
)
from src.retrieval.colpali_indexer import ColPaliIndexer, SearchResult

app = FastAPI(
    title="DocuVision-ColPali API",
    version="0.1.0",
    description="Multimodal Visual Document Intelligence using ColPali Late Interaction RAG",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
rasterizer = PDFRasterizer(cache_dir="data/cache/pages")
indexer = ColPaliIndexer(index_dir="data/cache/index")
synthesizer = VLMSynthesizer()


class DocumentUploadResponse(BaseModel):
    doc_id: str
    num_pages: int
    dpi: int
    pages: list[PageMetadata]


class QueryRequest(BaseModel):
    query: str
    doc_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


class CitationDetail(BaseModel):
    page_idx: int
    score: float
    image_url: str
    annotated_url: str
    bounding_boxes: list[BoundingBox]


class QueryResponse(BaseModel):
    query: str
    grounded_answer: GroundedAnswer
    citations: list[CitationDetail]


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "DocuVision-ColPali",
        "indexed_pages": len(indexer.pages),
    }


@app.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def upload_document(
    file: UploadFile = File(...),
    dpi: int = Query(default=150, ge=72, le=300),
) -> DocumentUploadResponse:
    """Upload and index a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported.",
        )

    content = await file.read()
    try:
        doc_id, pages_meta, images = rasterizer.rasterize(content, dpi=dpi)
    except PDFEncryptedError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except (PDFCorruptedError, PDFNotFoundError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rasterization error: {e}",
        )

    # Index into ColPali multi-vector store
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

    return DocumentUploadResponse(
        doc_id=doc_id,
        num_pages=len(pages_meta),
        dpi=dpi,
        pages=pages_meta,
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Retrieval & Reasoning"])
def execute_query(req: QueryRequest) -> QueryResponse:
    """Execute Late Interaction retrieval, visual grounding, and VLM synthesis."""
    hits: list[SearchResult] = indexer.search(
        query=req.query,
        top_k=req.top_k,
        doc_id_filter=req.doc_id,
    )

    if not hits:
        refusal = GroundedAnswer(
            answer="NOT_FOUND_IN_DOCUMENT",
            confidence=0.0,
            supporting_pages=[],
            bounding_boxes=[],
            refusal=True,
            evidence_summary="No matching documents found in index.",
        )
        return QueryResponse(query=req.query, grounded_answer=refusal, citations=[])

    retrieved_images: list[Image.Image] = []
    retrieved_pages: list[int] = []
    bboxes_per_page: list[list[BoundingBox]] = []
    citations: list[CitationDetail] = []

    for hit in hits:
        p = Path(hit.image_path)
        if p.exists():
            img = Image.open(p)
            retrieved_images.append(img)
            retrieved_pages.append(hit.page_idx)
            boxes = extract_bounding_boxes(hit.patch_heatmap)
            bboxes_per_page.append(boxes)

            citations.append(
                CitationDetail(
                    page_idx=hit.page_idx,
                    score=hit.score,
                    image_url=f"/api/v1/documents/{hit.doc_id}/pages/{hit.page_idx}?annotated=false",
                    annotated_url=f"/api/v1/documents/{hit.doc_id}/pages/{hit.page_idx}?annotated=true",
                    bounding_boxes=boxes,
                )
            )

    top_score = hits[0].score if hits else 0.0
    grounded_ans = synthesizer.synthesize(
        query=req.query,
        retrieved_images=retrieved_images,
        retrieved_pages=retrieved_pages,
        bboxes_per_page=bboxes_per_page,
        top_retrieval_score=top_score,
    )

    return QueryResponse(
        query=req.query,
        grounded_answer=grounded_ans,
        citations=citations,
    )


@app.get(
    "/api/v1/documents/{doc_id}/pages/{page_idx}",
    response_class=Response,
    tags=["Inspection"],
)
def get_page_image(
    doc_id: str,
    page_idx: int,
    annotated: bool = Query(default=False),
) -> Response:
    """Retrieve raw or citation-annotated document page image."""
    page_file = Path("data/cache/pages") / doc_id / f"page_{page_idx}.png"
    if not page_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page_idx} for document {doc_id} not found.",
        )

    img = Image.open(page_file)

    if annotated:
        # Search page in index to get heatmap
        for meta, emb in zip(indexer.pages, indexer.embeddings):
            if meta["doc_id"] == doc_id and meta["page_idx"] == page_idx:
                q_dummy = indexer.embedder.embed_query("document content table chart")
                from src.retrieval.maxsim_engine import compute_maxsim
                _, patch_heat = compute_maxsim(q_dummy, emb)
                boxes = extract_bounding_boxes(patch_heat.cpu().tolist())
                img = render_visual_citation(img, boxes)
                break

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
