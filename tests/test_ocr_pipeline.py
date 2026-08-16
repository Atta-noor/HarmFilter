"""
Unit tests for MemeOCRPipeline — PaddleOCR v4 backbone.

Tests (per PRD §5.4):
  1. preprocess() returns dict with exactly 3 keys: normal, inverted, dilated
  2. NMS deduplication removes boxes with center distance dx<50 AND dy<20
  3. DBSCAN clusters top text (y<200) and bottom text (y>400) into separate groups
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_pipeline import MemeOCRPipeline, TextRegion


@pytest.fixture
def pipeline():
    """Create a pipeline instance (no GPU)."""
    return MemeOCRPipeline(use_gpu=False)


# ── Test 1: preprocess returns 3 image versions ─────────────────────────

def test_preprocess_returns_three_versions(pipeline):
    """preprocess() must return a dict with keys: normal, inverted, dilated."""
    # Create a synthetic 640×480 BGR image (below 800px to also test upscaling)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipeline.preprocess(dummy_image)

    assert isinstance(result, dict), "preprocess must return a dict"
    expected_keys = {"normal", "inverted", "dilated"}
    assert set(result.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(result.keys())}"
    )

    # Each value should be a numpy array
    for key in expected_keys:
        assert isinstance(result[key], np.ndarray), f"'{key}' must be a numpy array"
        assert result[key].ndim == 2, f"'{key}' must be a 2D (grayscale) array"


# ── Test 2: NMS deduplication ────────────────────────────────────────────

def test_nms_removes_near_duplicates(pipeline):
    """Two boxes with |dx|<50 AND |dy|<20 → keep only higher confidence."""
    regions = [
        TextRegion(text="hello", confidence=0.9,
                   x_center=100, y_center=100),
        TextRegion(text="hello", confidence=0.7,
                   x_center=130, y_center=110),  # dx=30<50, dy=10<20 → duplicate
        TextRegion(text="world", confidence=0.85,
                   x_center=500, y_center=400),   # far away → kept
    ]

    deduped = MemeOCRPipeline._nms_dedup(regions)

    assert len(deduped) == 2, f"Expected 2 regions after NMS, got {len(deduped)}"

    # The higher-confidence "hello" (0.9) should survive
    texts = [(r.text, r.confidence) for r in deduped]
    assert ("hello", 0.9) in texts, "Higher-confidence 'hello' must survive NMS"
    assert ("hello", 0.7) not in texts, "Lower-confidence duplicate must be removed"
    assert ("world", 0.85) in texts, "Distant 'world' region must survive"


def test_nms_keeps_distant_boxes(pipeline):
    """Boxes with |dx|>=50 OR |dy|>=20 should NOT be removed."""
    regions = [
        TextRegion(text="top", confidence=0.8,
                   x_center=100, y_center=50),
        TextRegion(text="bottom", confidence=0.8,
                   x_center=100, y_center=450),  # dy=400 >> 20
    ]

    deduped = MemeOCRPipeline._nms_dedup(regions)
    assert len(deduped) == 2, "Distant regions must both survive NMS"


# ── Test 3: DBSCAN clusters top/bottom separately ───────────────────────

def test_dbscan_separates_top_bottom(pipeline):
    """Regions at y<200 and y>400 must cluster into separate top/bottom groups."""
    regions = [
        TextRegion(text="When", confidence=0.95,
                   x_center=100, y_center=50),
        TextRegion(text="you", confidence=0.90,
                   x_center=200, y_center=60),
        TextRegion(text="deploy", confidence=0.88,
                   x_center=300, y_center=55),
        # --- gap ---
        TextRegion(text="And", confidence=0.92,
                   x_center=100, y_center=450),
        TextRegion(text="nothing", confidence=0.87,
                   x_center=200, y_center=455),
        TextRegion(text="breaks", confidence=0.91,
                   x_center=300, y_center=460),
    ]

    result = pipeline.cluster_by_position(regions)

    assert "top_text" in result
    assert "bottom_text" in result
    assert "all_text" in result
    assert "num_regions" in result
    assert "raw_regions" in result

    # Top cluster should contain words from y~50-60
    assert "When" in result["top_text"]
    assert "you" in result["top_text"]

    # Bottom cluster should contain words from y~450-460
    assert "breaks" in result["bottom_text"]
    assert "nothing" in result["bottom_text"]

    # Top and bottom should be different
    assert result["top_text"] != result["bottom_text"]

    # all_text should contain everything
    assert "When" in result["all_text"]
    assert "breaks" in result["all_text"]

    # num_regions
    assert result["num_regions"] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
