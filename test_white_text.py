"""Test white text (Impact font style) on dark background."""
import os
os.environ["FLAGS_use_mkldnn"] = "0"

from ocr_pipeline import MemeOCRPipeline
from PIL import Image
import numpy as np
import cv2

pipeline = MemeOCRPipeline(use_gpu=False)

# Simulate a meme: dark background with WHITE text
img = np.zeros((600, 800, 3), dtype=np.uint8)
# Dark gradient background
img[:] = (30, 30, 50)

# White text with black outline (Impact font style)
font = cv2.FONT_HERSHEY_SIMPLEX
text_top = "WHEN YOU FIX ONE BUG"
text_bot = "AND CREATE THREE MORE"

# Draw black outline then white fill (simulates Impact font stroke)
for dx in [-2, -1, 0, 1, 2]:
    for dy in [-2, -1, 0, 1, 2]:
        cv2.putText(img, text_top, (50+dx, 80+dy), font, 1.2, (0, 0, 0), 3)
        cv2.putText(img, text_bot, (40+dx, 550+dy), font, 1.2, (0, 0, 0), 3)
cv2.putText(img, text_top, (50, 80), font, 1.2, (255, 255, 255), 2)
cv2.putText(img, text_bot, (40, 550), font, 1.2, (255, 255, 255), 2)

pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

result = pipeline.extract_from_pil(pil_img)
print("=== White Text on Dark Background ===")
print(f"Top:    {result['top_text']}")
print(f"Bottom: {result['bottom_text']}")
print(f"All:    {result['all_text']}")
print(f"Regions: {result['num_regions']}")
for r in result["raw_regions"]:
    print(f"  - '{r['text']}' (conf={r['confidence']:.2f})")
