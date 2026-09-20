"""
REST API for text hate-speech detection — the bridge between the BiLSTM
model and any external frontend (e.g. a Flutter app via ngrok).

This wraps the SAME lstm_inference.LSTMPredictor that app.py (Streamlit) uses,
so predictions are identical to the web UI.

Run:
    python api.py                 # serves on http://0.0.0.0:5000
Then expose it publicly:
    ngrok http 5000               # gives you an https://xxxx.ngrok-free.app URL

Endpoints:
    GET  /health                  -> {"status": "ok", "model_loaded": true|false}
    POST /predict                 -> classify text
        request  JSON: {"text": "..."}
        response JSON: {
            "label": "Hate Speech" | "Offensive Language" | "Neither",
            "confidence": 87.4,                      # percent for `label`
            "probabilities": {                       # percents, sum ~100
                "hate_speech": 87.4,
                "offensive_language": 9.1,
                "neither": 3.5
            }
        }
"""
import os
# Disable oneDNN for PaddlePaddle 3.0 on Windows CPU (kept consistent with app.py)
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from flask import Flask, request, jsonify
from flask_cors import CORS

from lstm_inference import load_lstm_predictor

app = Flask(__name__)
CORS(app)  # allow calls from Flutter web / any origin

# Cache the predictor so the model loads once, not per request.
_predictor = None


def _resolve_model_path() -> str:
    """Mirror app.py's lookup: root or models/, .h5 or .keras."""
    base = "bilstm_model"
    candidates = [
        f"{base}.h5",
        f"{base}.keras",
        os.path.join("models", f"{base}.h5"),
        os.path.join("models", f"{base}.keras"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No BiLSTM model found. Tried: {', '.join(candidates)}"
    )


def get_predictor():
    global _predictor
    if _predictor is None:
        model_path = _resolve_model_path()
        _predictor = load_lstm_predictor(model_path)
    return _predictor


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _predictor is not None})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400

    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        # Model artifacts (tokenizer.pkl / lstm_config.pkl / .h5) not present — see CLAUDE.md
        return jsonify({"error": str(e)}), 503

    try:
        label, confidence, probs = predictor.predict(text, return_probabilities=True)
        probs = probs.tolist() if hasattr(probs, "tolist") else list(probs)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify({
        "label": label,
        "confidence": round(float(confidence), 2),
        "probabilities": {
            "hate_speech": round(float(probs[0]) * 100, 2),
            "offensive_language": round(float(probs[1]) * 100, 2),
            "neither": round(float(probs[2]) * 100, 2),
        },
    })


if __name__ == "__main__":
    # host=0.0.0.0 so ngrok (and other devices) can reach it; debug off for stability.
    app.run(host="0.0.0.0", port=5000, debug=False)
