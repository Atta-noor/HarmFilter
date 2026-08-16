"""Standalone test runner — no pytest needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ocr_pipeline import MemeOCRPipeline, TextRegion

pipeline = MemeOCRPipeline(use_gpu=False)

# Test 1: preprocess returns 3 versions
print("Test 1: preprocess returns 3 versions...")
dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
result = pipeline.preprocess(dummy)
assert isinstance(result, dict), f"Expected dict, got {type(result)}"
assert set(result.keys()) == {"original", "inverted", "enhanced_gray"}, f"Keys: {result.keys()}"
for k in result:
    assert isinstance(result[k], np.ndarray), f"{k} not ndarray"
    assert result[k].ndim == 3, f"{k} should be 3-channel BGR, got ndim={result[k].ndim}"
print("  PASSED")

# Test 2: NMS deduplication
print("Test 2: NMS deduplication...")
regions = [
    TextRegion(text="hello", confidence=0.9, x_center=100, y_center=100),
    TextRegion(text="hello", confidence=0.7, x_center=130, y_center=110),  # dup
    TextRegion(text="world", confidence=0.85, x_center=500, y_center=400),
]
deduped = MemeOCRPipeline._nms_dedup(regions)
assert len(deduped) == 2, f"Expected 2, got {len(deduped)}"
texts = [(r.text, r.confidence) for r in deduped]
assert ("hello", 0.9) in texts
assert ("hello", 0.7) not in texts
assert ("world", 0.85) in texts
print("  PASSED")

# Test 3: DBSCAN separates top/bottom
print("Test 3: DBSCAN separates top/bottom...")
regions2 = [
    TextRegion(text="When", confidence=0.95, x_center=100, y_center=50),
    TextRegion(text="you", confidence=0.90, x_center=200, y_center=60),
    TextRegion(text="deploy", confidence=0.88, x_center=300, y_center=55),
    TextRegion(text="And", confidence=0.92, x_center=100, y_center=450),
    TextRegion(text="nothing", confidence=0.87, x_center=200, y_center=455),
    TextRegion(text="breaks", confidence=0.91, x_center=300, y_center=460),
]
res = pipeline.cluster_by_position(regions2)
top = res["top_text"]
bottom = res["bottom_text"]
assert "When" in top, f"top_text missing 'When': {top}"
assert "breaks" in bottom, f"bottom_text missing 'breaks': {bottom}"
assert top != bottom
assert res["num_regions"] == 6
assert "When" in res["all_text"] and "breaks" in res["all_text"]
print("  PASSED")

print()
print("ALL 3 TESTS PASSED")
