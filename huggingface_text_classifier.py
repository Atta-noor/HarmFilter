"""
Wrapper for Hugging Face text-classification pipeline tuned for hate/offensive detection.

This wrapper attempts to map model-specific labels into the project's three classes:
`Hate Speech`, `Offensive Language`, `Neither`.

It is intentionally conservative: when labels are unclear it returns `Neither`.
"""
from typing import Tuple, List


class HFTextClassifier:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-hate"):
        try:
            from transformers import pipeline
        except Exception as e:
            raise RuntimeError("transformers is required for HF text classifier: pip install transformers[torch]")

        # return_all_scores gives full list of label scores
        self.pipeline = pipeline("text-classification", model=model_name, return_all_scores=True)
        self.model_name = model_name

    def _map_label(self, label: str) -> str:
        """Map model label to one of the three project labels."""
        l = label.lower()
        if any(k in l for k in ["hate", "hateful", "racist", "hatespeech"]):
            return "Hate Speech"
        if any(k in l for k in ["offensive", "abusive", "insult", "offence", "toxic"]):
            return "Offensive Language"
        # Some models return 'normal', 'neutral' etc.
        if any(k in l for k in ["none", "neutral", "normal", "non-hate"]):
            return "Neither"
        # Fallback conservative mapping
        return "Neither"

    def predict(self, text: str, return_probabilities: bool = False) -> Tuple[str, float, List[float]]:
        """Predict and return (label, confidence_percent, [p_hate,p_offensive,p_neither])."""
        if not text or not text.strip():
            return "Neither", 0.0, [0.0, 0.0, 1.0]

        results = self.pipeline(text)
        # results is list of dicts: [{label:score}, ...] per model's labels
        # Flatten into a dict label->score
        scores_map = {}
        for entry in results[0]:
            lbl = entry.get('label')
            sc = entry.get('score', 0.0)
            scores_map[lbl] = sc

        # Map each model label into our three classes and accumulate
        hate_score = 0.0
        off_score = 0.0
        neither_score = 0.0
        for lbl, sc in scores_map.items():
            mapped = self._map_label(lbl)
            if mapped == "Hate Speech":
                hate_score += sc
            elif mapped == "Offensive Language":
                off_score += sc
            else:
                neither_score += sc

        total = hate_score + off_score + neither_score
        if total <= 0:
            probs = [0.0, 0.0, 1.0]
        else:
            probs = [hate_score / total, off_score / total, neither_score / total]

        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        labels = ["Hate Speech", "Offensive Language", "Neither"]
        best_label = labels[best_idx]
        confidence = probs[best_idx] * 100.0

        if return_probabilities:
            return best_label, confidence, probs
        return best_label, confidence


def load_hf_text_predictor(model_name: str = None):
    model_name = model_name or "cardiffnlp/twitter-roberta-base-hate"
    return HFTextClassifier(model_name=model_name)
