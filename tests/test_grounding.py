"""Unit tests for heatmap to bounding-box detector and visual citation rendering."""

from PIL import Image
import numpy as np
import pytest

from src.grounding.heatmap_to_bbox import (
    BoundingBox,
    extract_bounding_boxes,
    render_visual_citation,
)


def test_extract_bounding_boxes_grid() -> None:
    # 32x32 = 1024 patches
    grid = np.zeros((32, 32), dtype=np.float32)
    # Inject a high attention cluster in rows 10-18, cols 10-22
    grid[10:18, 10:22] = 1.0

    boxes = extract_bounding_boxes(grid.flatten().tolist(), grid_size=32)

    assert len(boxes) >= 1
    top_box = boxes[0]
    assert 0.0 <= top_box.x_min < top_box.x_max <= 1.0
    assert 0.0 <= top_box.y_min < top_box.y_max <= 1.0
    assert top_box.confidence > 0.5


def test_bbox_physical_conversion() -> None:
    box = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9, confidence=0.95)
    px1, py1, px2, py2 = box.to_physical_pixels(width=1000, height=2000)

    assert px1 == 200
    assert py1 == 600
    assert px2 == 800
    assert py2 == 1800


def test_render_visual_citation(tmp_path) -> None:
    img = Image.new("RGB", (400, 600), color=(240, 240, 240))
    boxes = [
        BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.4, confidence=0.9),
    ]
    out_file = tmp_path / "citation_test.jpg"
    annotated = render_visual_citation(img, boxes, save_path=out_file)

    assert annotated.size == (400, 600)
    assert out_file.exists()
