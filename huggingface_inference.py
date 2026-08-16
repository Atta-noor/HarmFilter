"""
Small wrapper around Hugging Face zero-shot pipeline to provide
an interface compatible with the existing app.

Uses `valhalla/distilbart-mnli-12-1` by default (smaller MNLI model).
"""
from transformers import pipeline


class HFZeroShotPredictor:
    def __init__(self, model_name: str = "valhalla/distilbart-mnli-12-1"):
        # candidate labels expected by the app
        self.model_name = model_name
        self.pipeline = pipeline("zero-shot-classification", model=self.model_name)

    def predict(self, text: str, return_probabilities: bool = False):
        """Classify text into Hate/Offensive/Neither labels.

        Returns: (label, confidence_percent, probabilities_list)
        probabilities_list corresponds to [Hate Speech, Offensive Language, Neither]
        """
        if not text or not text.strip():
            return "Neither", 0.0, [0.0, 0.0, 1.0]

        labels = ["Hate Speech", "Offensive Language", "Neither"]
        result = self.pipeline(text, candidate_labels=labels, hypothesis_template="This text is {}.")

        # pipeline returns scores aligned with labels
        scores = result.get("scores", [])
        # Map scores to labels order above (pipeline already returns in that order if passed)
        # Ensure we return list in same order [Hate, Offensive, Neither]
        probs = [0.0, 0.0, 0.0]
        labels_out = result.get("labels", [])
        for lbl, sc in zip(labels_out, scores):
            if lbl == "Hate Speech":
                probs[0] = sc
            elif lbl == "Offensive Language":
                probs[1] = sc
            elif lbl == "Neither":
                probs[2] = sc

        # If any zero (missing), try to fill by direct mapping
        for i, lbl in enumerate(labels):
            if probs[i] == 0.0 and lbl in labels_out:
                probs[i] = scores[labels_out.index(lbl)]

        # Normalize in case
        total = sum(probs) if sum(probs) > 0 else 1.0
        probs = [p / total for p in probs]

        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        best_label = labels[best_idx]
        confidence = probs[best_idx] * 100.0

        if return_probabilities:
            return best_label, confidence, probs
        return best_label, confidence


def load_hf_predictor(model_name: str = None):
    model_name = model_name or "valhalla/distilbart-mnli-12-1"
    return HFZeroShotPredictor(model_name=model_name)
