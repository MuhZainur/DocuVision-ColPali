"""Spatial grounding: transforms patch similarity heatmaps into normalized and physical bounding boxes."""

from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates in [0.0, 1.0] and physical pixel dimensions."""
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    def to_physical_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized [0.0, 1.0] coordinates to physical pixel dimensions."""
        px_min = int(round(self.x_min * width))
        py_min = int(round(self.y_min * height))
        px_max = int(round(self.x_max * width))
        py_max = int(round(self.y_max * height))
        # Ensure within image bounds
        return (
            max(0, min(width - 1, px_min)),
            max(0, min(height - 1, py_min)),
            max(0, min(width, px_max)),
            max(0, min(height, py_max)),
        )


def extract_bounding_boxes(
    patch_scores: list[float] | np.ndarray,
    grid_size: int = 32,
    percentile_threshold: float = 85.0,
    min_area_ratio: float = 0.005,
    max_boxes: int = 4,
) -> list[BoundingBox]:
    """Convert 1D patch similarity scores to normalized spatial bounding boxes.

    Args:
        patch_scores: 1D array/list of length grid_size * grid_size (e.g. 1024).
        grid_size: Spatial dimension of the patch grid (default 32 for 32x32).
        percentile_threshold: Attention percentile cut-off (default 85.0).
        min_area_ratio: Filter out tiny noise contours smaller than ratio of total area.
        max_boxes: Maximum number of salient bounding boxes to return.

    Returns:
        List of BoundingBox objects sorted by attention confidence descending.
    """
    scores = np.asarray(patch_scores, dtype=np.float32)
    expected_len = grid_size * grid_size
    if len(scores) != expected_len:
        # If size differs, interpolate or infer closest square
        s = int(math.isqrt(len(scores)))
        if s * s == len(scores):
            grid_size = s
        else:
            scores = np.resize(scores, (expected_len,))

    # 1. Reshape to 2D topological matrix (H, W)
    heatmap_2d = scores.reshape((grid_size, grid_size))

    # Normalize to [0.0, 1.0] range
    h_min, h_max = float(heatmap_2d.min()), float(heatmap_2d.max())
    if h_max > h_min:
        norm_map = (heatmap_2d - h_min) / (h_max - h_min)
    else:
        norm_map = np.zeros_like(heatmap_2d)

    # 2. Gaussian smoothing
    blurred = cv2.GaussianBlur(norm_map, (3, 3), sigmaX=1.0)

    # 3. Otsu thresholding on normalized attention map
    blurred_u8 = (blurred * 255.0).astype(np.uint8)
    if blurred_u8.max() > blurred_u8.min():
        _, binary_mask = cv2.threshold(blurred_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        threshold_val = float(np.percentile(blurred, percentile_threshold))
        binary_mask = (blurred >= max(threshold_val, 0.5)).astype(np.uint8) * 255

    # Morphological closing to bridge adjacent patches
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Detect outer contours
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[BoundingBox] = []
    total_grid_area = grid_size * grid_size

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        box_area = w * h
        if (box_area / total_grid_area) < min_area_ratio:
            continue

        # Normalized coordinates [0.0, 1.0]
        x_min = float(x / grid_size)
        y_min = float(y / grid_size)
        x_max = float(min(1.0, (x + w) / grid_size))
        y_max = float(min(1.0, (y + h) / grid_size))

        # Confidence is average attention value inside the box region
        roi = norm_map[y : y + h, x : x + w]
        confidence = float(roi.mean()) if roi.size > 0 else 0.5

        boxes.append(
            BoundingBox(
                x_min=round(x_min, 4),
                y_min=round(y_min, 4),
                x_max=round(x_max, 4),
                y_max=round(y_max, 4),
                confidence=round(confidence, 4),
            )
        )

    # If no contours met criteria, fall back to top attention bounding box
    if not boxes and h_max > h_min:
        y_idx, x_idx = np.unravel_index(np.argmax(norm_map), norm_map.shape)
        half_w = max(1, grid_size // 8)
        x0 = max(0, x_idx - half_w)
        y0 = max(0, y_idx - half_w)
        x1 = min(grid_size, x_idx + half_w + 1)
        y1 = min(grid_size, y_idx + half_w + 1)
        boxes.append(
            BoundingBox(
                x_min=round(x0 / grid_size, 4),
                y_min=round(y0 / grid_size, 4),
                x_max=round(x1 / grid_size, 4),
                y_max=round(y1 / grid_size, 4),
                confidence=0.85,
            )
        )

    boxes.sort(key=lambda b: b.confidence, reverse=True)
    return boxes[:max_boxes]


def render_visual_citation(
    image: Image.Image,
    bboxes: list[BoundingBox],
    outline_color: tuple[int, int, int] = (230, 40, 40),
    fill_alpha: int = 60,
    save_path: str | Path | None = None,
) -> Image.Image:
    """Render semi-transparent bounding box overlays on source document page image.

    Args:
        image: Original PIL page image.
        bboxes: List of normalized BoundingBox objects.
        outline_color: RGB tuple for bounding box outline (default red).
        fill_alpha: Transparency alpha for fill highlight (0-255).
        save_path: Optional file path to persist rendered image.

    Returns:
        New PIL Image with visual citations rendered.
    """
    base_img = image.convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = base_img.size

    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = box.to_physical_pixels(w, h)
        fill_color = (*outline_color, fill_alpha)
        # Draw translucent filled rectangle
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=(*outline_color, 240), width=3)
        # Small corner tag
        tag = f"Citation #{idx+1} ({int(box.confidence * 100)}%)"
        draw.text((x1 + 4, max(0, y1 - 16)), tag, fill=(*outline_color, 255))

    combined = Image.alpha_composite(base_img, overlay).convert("RGB")

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        combined.save(p, format="JPEG", quality=92)

    return combined
