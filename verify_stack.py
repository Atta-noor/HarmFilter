"""Verify the full stack: TF/BiLSTM + PaddleOCR + protobuf."""
import os
os.environ["FLAGS_use_mkldnn"] = "0"

# 1. Verify TF + BiLSTM (model lives in models/, resolve it like app.py/api.py do)
from lstm_inference import load_lstm_predictor
_model_path = next(
    (p for p in ["bilstm_model.h5", os.path.join("models", "bilstm_model.h5")] if os.path.exists(p)),
    "bilstm_model.h5",
)
predictor = load_lstm_predictor(_model_path)
label, conf = predictor.predict("hello world")
print(f"BiLSTM: {label} ({conf:.1f}%)")

# 2. Verify OCR pipeline
from ocr_pipeline import MemeOCRPipeline
from PIL import Image
import numpy as np, cv2

img_np = np.ones((600, 800, 3), dtype=np.uint8) * 200
cv2.putText(img_np, "TOP TEXT", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
cv2.putText(img_np, "BOTTOM", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
img = Image.fromarray(img_np)

pipeline = MemeOCRPipeline(use_gpu=False)
result = pipeline.extract_from_pil(img)
print("OCR top:", result["top_text"])
print("OCR bottom:", result["bottom_text"])
print("OCR all:", result["all_text"])

# 3. Summary
import paddle, paddleocr, google.protobuf, tensorflow as tf
print()
print(f"paddle={paddle.__version__}  paddleocr={paddleocr.__version__}  protobuf={google.protobuf.__version__}  tf={tf.__version__}")
print("ALL WORKING")
