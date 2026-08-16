"""
Hate Speech Integration Wrapper
================================
Chains the PaddleOCR v4 meme pipeline with the existing BiLSTM classifier.

Usage:
    from src.hate_speech import analyze_meme
    result = analyze_meme("meme.jpg")
    # result contains OCR fields + hate_speech_label + hate_speech_score + is_hate_speech
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ocr_pipeline import MemeOCRPipeline


def analyze_meme(
    image_path: str,
    predictor=None,
    use_gpu: bool = False,
    language: str = "english",
) -> dict:
    """End-to-end meme hate speech detection.

    1. Extract text from the meme image using PaddleOCR v4 pipeline
    2. Classify the extracted text using the BiLSTM model

    Args:
        image_path: Path to the meme image.
        predictor: An LSTMPredictor instance. If None, one is loaded automatically.
        use_gpu: Whether to use GPU for PaddleOCR.
        language: Language for the BiLSTM model ('english' or 'roman_urdu').

    Returns:
        dict with keys:
            - top_text, bottom_text, all_text, num_regions, raw_regions  (from OCR)
            - hate_speech_label: str  ('Hate Speech', 'Offensive Language', 'Neither')
            - hate_speech_score: float  (confidence percentage)
            - hate_speech_probs: list[float]  (per-class probabilities)
            - is_hate_speech: bool
    """
    # --- Stage 0-2: OCR ---
    pipeline = MemeOCRPipeline(use_gpu=use_gpu)
    ocr_result = pipeline.extract(image_path)

    all_text = ocr_result.get("all_text", "")

    # --- Stage 3: BiLSTM Classification ---
    if predictor is None:
        try:
            from lstm_inference import load_lstm_predictor
            predictor = load_lstm_predictor(language=language)
        except Exception as exc:
            logger.error("Failed to load BiLSTM predictor: %s", exc)
            ocr_result.update({
                "hate_speech_label": "Error",
                "hate_speech_score": 0.0,
                "hate_speech_probs": [],
                "is_hate_speech": False,
            })
            return ocr_result

    if not all_text.strip():
        ocr_result.update({
            "hate_speech_label": "Neither",
            "hate_speech_score": 0.0,
            "hate_speech_probs": [0.0, 0.0, 1.0],
            "is_hate_speech": False,
        })
        return ocr_result

    try:
        label, confidence, probs = predictor.predict(all_text, return_probabilities=True)
        probs_list = probs.tolist() if hasattr(probs, "tolist") else list(probs)
    except Exception as exc:
        logger.error("BiLSTM prediction failed: %s", exc)
        label, confidence, probs_list = "Error", 0.0, []

    ocr_result.update({
        "hate_speech_label": label,
        "hate_speech_score": confidence,
        "hate_speech_probs": probs_list,
        "is_hate_speech": label == "Hate Speech",
    })

    return ocr_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.hate_speech <image_path> [language]")
        sys.exit(1)

    img_path = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "english"

    result = analyze_meme(img_path, language=lang)
    print(json.dumps(result, indent=2, ensure_ascii=False))
