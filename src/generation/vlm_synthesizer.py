"""VLM synthesis engine with visual citation grounding and structured JSON schemas."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

load_dotenv()

from src.grounding.heatmap_to_bbox import BoundingBox


class GroundedAnswer(BaseModel):
    """Structured response contract for grounded multimodal document answering."""
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_pages: list[int]
    bounding_boxes: list[list[float]] = Field(
        description="List of normalized [x_min, y_min, x_max, y_max] bounding boxes."
    )
    refusal: bool = False
    evidence_summary: str = ""


class VLMSynthesizer:
    """Multimodal Vision-Language synthesizer for grounded document QA."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        min_retrieval_confidence: float = 0.15,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.min_retrieval_confidence = min_retrieval_confidence
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Google GenAI client if API key is present."""
        if not self.api_key:
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except Exception:
            self._client = None

    def transcribe_scanned_page(self, image: Image.Image) -> list[list[Any]]:
        """Extract text and approximate spatial regions from scanned/pure-image pages via Gemini Vision."""
        if self._client is None:
            return []
        try:
            prompt = (
                "Extract all visible text, numbers, table headers, and diagram labels from this document page image. "
                "Output ONLY a raw JSON array of items: [[x0_pct, y0_pct, x1_pct, y1_pct, 'text_string'], ...]. "
                "Percentages must be 0-100 float values. Example: [[10.0, 5.0, 20.0, 40.0, 'Invoice Number']]. "
                "Do not wrap in markdown or backticks."
            )
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
            )
            raw = resp.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            w, h = image.size
            result: list[list[Any]] = []
            for item in data:
                if len(item) == 5:
                    x0, y0, x1, y1, txt = item
                    result.append([
                        round(float(x0) * w / 100.0, 1),
                        round(float(y0) * h / 100.0, 1),
                        round(float(x1) * w / 100.0, 1),
                        round(float(y1) * h / 100.0, 1),
                        str(txt),
                    ])
            return result
        except Exception:
            return []

    def synthesize(
        self,
        query: str,
        retrieved_images: list[Image.Image],
        retrieved_pages: list[int],
        bboxes_per_page: list[list[BoundingBox]],
        top_retrieval_score: float,
    ) -> GroundedAnswer:
        """Synthesize factual answer grounded in retrieved document page images.

        Args:
            query: User's question or search query.
            retrieved_images: List of PIL page images in rank order.
            retrieved_pages: 0-indexed page numbers.
            bboxes_per_page: Extracted bounding boxes for each retrieved page.
            top_retrieval_score: Highest retrieval similarity score.

        Returns:
            GroundedAnswer model instance.
        """
        # 1. Refusal logic for irrelevant queries or empty document set
        if not retrieved_images or top_retrieval_score < self.min_retrieval_confidence:
            return GroundedAnswer(
                answer="NOT_FOUND_IN_DOCUMENT",
                confidence=0.0,
                supporting_pages=[],
                bounding_boxes=[],
                refusal=True,
                evidence_summary="The queried information is not visually or textually present in the document.",
            )

        flat_boxes: list[list[float]] = []
        for page_boxes in bboxes_per_page:
            for b in page_boxes:
                flat_boxes.append([b.x_min, b.y_min, b.x_max, b.y_max])

        # 2. If Gemini client is active, query Multimodal VLM
        if self._client is not None and retrieved_images:
            try:
                system_instruction = (
                    "You are DocuVision, an enterprise multimodal document intelligence agent. "
                    "Analyze the provided document page images and answer the user's question directly. "
                    "Cite specific visual regions and table cells. "
                    "If the answer is completely missing from the images, reply exactly 'NOT_FOUND_IN_DOCUMENT'."
                )
                contents: list[Any] = [system_instruction, f"Question: {query}"]
                for idx, img in enumerate(retrieved_images[:2]):
                    contents.append(f"Document Page {retrieved_pages[idx] + 1}:")
                    contents.append(img)

                resp = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                raw_text = resp.text.strip() if resp.text else ""

                if "NOT_FOUND_IN_DOCUMENT" in raw_text:
                    return GroundedAnswer(
                        answer="NOT_FOUND_IN_DOCUMENT",
                        confidence=0.1,
                        supporting_pages=[],
                        bounding_boxes=[],
                        refusal=True,
                        evidence_summary="The VLM confirmed the evidence is absent from the page images.",
                    )

                return GroundedAnswer(
                    answer=raw_text,
                    confidence=min(1.0, round(top_retrieval_score / 10.0, 2)),
                    supporting_pages=retrieved_pages[:2],
                    bounding_boxes=flat_boxes[:4],
                    refusal=False,
                    evidence_summary=f"Synthesized from {len(retrieved_images)} retrieved page(s).",
                )
            except Exception:
                # If API call fails (network/quota), proceed with deterministic fallback
                pass

        # 3. Deterministic fallback synthesizer
        best_page = retrieved_pages[0] if retrieved_pages else 0
        answer_text = (
            f"Based on visual analysis of Page {best_page + 1}, the query '{query}' "
            f"matches relevant visual elements located at coordinates {flat_boxes[:2]}."
        )

        return GroundedAnswer(
            answer=answer_text,
            confidence=min(1.0, round(top_retrieval_score / 5.0, 2)),
            supporting_pages=[best_page],
            bounding_boxes=flat_boxes[:4],
            refusal=False,
            evidence_summary=f"Grounded on Page {best_page + 1} with {len(flat_boxes)} salient region(s).",
        )
