# Product Requirements Document
## Meme Hate Speech Detection: OCR Pipeline Upgrade to PaddleOCR v4

| Field | Value |
|---|---|
| **Version** | 1.0 |
| **Status** | Ready for Implementation |
| **Target Model** | Claude Opus |
| **Project** | Meme Hate Speech Classifier |

---

## 1. Overview

This PRD defines the requirements to upgrade the existing meme OCR pipeline from a CRAFT + EasyOCR architecture to a PaddleOCR v4 backbone. The existing Bi-LSTM hate speech classifier is preserved unchanged; only the text extraction layer is being replaced and enhanced.

> **Objective:** Replace the fragile single-path OCR pipeline with a multi-path, multi-version detection system using PaddleOCR v4 as the primary backbone, DBSCAN-based spatial clustering for top/bottom caption separation, and structured JSON output feeding directly into the existing Bi-LSTM classifier.

---

## 2. Problem Statement

### 2.1 Current Failures

| Failure Mode | Root Cause | Impact |
|---|---|---|
| White Impact-font text missed | Single threshold path fails on dark backgrounds | 🔴 HIGH — most memes use this format |
| Top & bottom captions merged | EasyOCR `paragraph=True` ignores semantic separation | 🔴 HIGH — BiLSTM receives corrupted input |
| Hallucinated text in faces/logos | No confidence gate post-recognition | 🟡 MEDIUM — false positives in classifier |
| Tiny meme text undetected | No upscaling pre-processing | 🟡 MEDIUM — sub-800px images fail silently |
| Broken character chains | No morphological ops post-threshold | 🟢 LOW — partial words degrade accuracy |

### 2.2 Why PaddleOCR v4

- Handles curved, rotated, and stylized meme fonts natively via angle classification
- PP-OCRv4 det+rec models outperform CRAFT+EasyOCR on non-standard fonts by ~18% accuracy
- 2–3x faster inference than CRAFT on CPU
- Built-in angle classifier (`use_angle_cls=True`) catches rotated/flipped meme text
- Single install replaces two separate OCR libraries

---

## 3. Architecture: Enhanced Pipeline

### Stage 0 — Multi-Scale Preprocessing `NEW`

- Upscale to 800px minimum (LANCZOS4 interpolation) for small meme text
- CLAHE contrast enhancement before filtering (`clipLimit=3.0`, `tileGridSize=8x8`)
- Bilateral filter to smooth noise while preserving text edges
- Generate **3 image versions** in parallel:
  - `normal` — standard adaptive threshold (dark text on light bg)
  - `inverted` — bitwise NOT of normal (white Impact font on dark bg)
  - `dilated` — morphological dilation (kernel 2x2, 1 iter) to connect broken chars

### Stage 1 — Multi-Path Detection `REPLACES CRAFT`

- Run PaddleOCR v4 independently on all 3 image versions
- Collect all bounding boxes with text and confidence scores
- NMS deduplication: if two boxes have `|dx| < 50px` AND `|dy| < 20px` → keep higher confidence
- Confidence gate: discard any region with confidence `< 0.6`
- Fallback: if paddle confidence < 0.6 on a region, optionally run EasyOCR on that crop

### Stage 2 — Spatial Clustering `REPLACES paragraph=True`

- DBSCAN clustering on y-coordinates only (`eps=50`, `min_samples=1`)
- Sort clusters by mean y-position: lowest y = top caption
- Within each cluster, sort regions left-to-right by `x_center`
- Join words in each cluster with spaces
- Output structured dict:

```json
{
  "top_text":    "When you deploy on Friday",
  "bottom_text": "And nothing breaks",
  "all_text":    "When you deploy on Friday And nothing breaks",
  "num_regions": 4,
  "raw_regions": [{"text": "...", "confidence": 0.94}]
}
```

### Stage 3 — Bi-LSTM Integration `UNCHANGED`

- Feed `result['all_text']` to existing `tokenizer.texts_to_sequences()`
- Pad sequences to existing `maxlen` (typically 100)
- Return merged output dict: OCR fields + `hate_speech_score` + `is_hate_speech` flag

---

## 4. End-to-End Flow

```
INPUT IMAGE
     │
     ▼
┌─────────────────────────────┐
│  STAGE 0: Multi-Scale Prep  │  ← Upscale, CLAHE, Bilateral,
│  3 image versions           │    Normal / Inverted / Dilated
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  STAGE 1: PaddleOCR v4     │  ← Run on all 3 versions,
│  Multi-Path Detection       │    NMS dedup, conf ≥ 0.6
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  STAGE 2: DBSCAN Cluster   │  ← Separate top/bottom captions,
│  Spatial Separation         │    sort L→R within clusters
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  STAGE 3: Bi-LSTM (EXISTING)│  ← Tokenize all_text,
│  Hate Speech Classification │    predict, threshold 0.5
└─────────────────────────────┘
     │
     ▼
  OUTPUT JSON
```

---

## 5. Technical Requirements

### 5.1 Files to Create / Modify

| File | Action | Description |
|---|---|---|
| `src/ocr_pipeline.py` | CREATE | `MemeOCRPipeline` class with all 3 stages |
| `src/hate_speech.py` | MODIFY | Integration wrapper: OCR → BiLSTM |
| `tests/test_ocr_pipeline.py` | CREATE | 3 unit tests covering core logic |
| `requirements.txt` | UPDATE | Add paddleocr, paddlepaddle, scikit-learn |
| `src/craft_pipeline.py` | DEPRECATE | Keep old pipeline for rollback |

### 5.2 Class Interface

```python
class MemeOCRPipeline:
    def __init__(self, use_gpu: bool = False)

    def preprocess(self, image: np.ndarray) -> dict
        # Returns: {'original', 'normal', 'inverted', 'dilated'}

    def detect_text_regions(self, image_versions: dict) -> List[TextRegion]
        # Returns: deduplicated, confidence-filtered regions

    def cluster_by_position(self, regions: List[TextRegion]) -> dict
        # Returns: {top_text, bottom_text, all_text, num_regions, raw_regions}

    def extract(self, image_path: str) -> dict
        # Main entrypoint: path in, structured dict out
```

### 5.3 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `paddleocr` | ≥2.7.0 | Primary OCR backbone (PP-OCRv4) |
| `paddlepaddle` | ≥2.5.0 | PaddleOCR backend framework |
| `scikit-learn` | ≥1.3.0 | DBSCAN spatial clustering |
| `opencv-python` | ≥4.8.0 | Image preprocessing |
| `easyocr` | existing | Optional fallback ensemble |
| `numpy` | existing | Array operations |

### 5.4 Unit Tests Required

1. `preprocess()` returns dict with exactly 3 keys: `normal`, `inverted`, `dilated`
2. NMS deduplication removes boxes with center distance `dx<50` AND `dy<20`
3. DBSCAN clusters top text (`y<200`) and bottom text (`y>400`) into separate groups

> ✅ **Acceptance Criteria:** Pipeline must correctly extract top and bottom text separately on a standard Impact-font meme. Confidence gate must drop OCR regions below 0.6. All 3 unit tests must pass. `extract()` must return a dict with keys: `top_text`, `bottom_text`, `all_text`, `num_regions`, `raw_regions`.

---

## 6. Claude Opus Implementation Prompt

> Open Claude.ai → Select **Claude Opus** → Paste the prompt below → Attach your existing project files if available.

---

```
You are a senior ML/AI engineer. Refactor my existing meme hate-speech
detection project to use the enhanced OCR pipeline described below.

## CONTEXT
My project already has:
  - A trained Bi-LSTM model for hate speech classification
  - An old OCR pipeline (CRAFT + EasyOCR, basic preprocessing)
  - Standard project structure: src/, models/, utils/, main.py

## TASK: Replace the OCR pipeline with PaddleOCR v4 backbone

### STAGE 0 - Multi-Scale Preprocessing (NEW)
  - Upscale images < 800px (LANCZOS4)
  - CLAHE contrast enhancement (clipLimit=3.0, tileGridSize=(8,8))
  - Generate 3 image versions: original, normal-threshold, inverted-threshold
  - Bilateral filter BEFORE adaptive threshold
  - Morphological dilation to connect broken chars (kernel 2x2, iter=1)

### STAGE 1 - Multi-Path Detection (REPLACE CRAFT)
  - Primary: PaddleOCR v4 (use_angle_cls=True, det+rec PP-OCRv4 models)
  - Run detection on ALL 3 image versions independently
  - NMS deduplication: center-distance based (dx<50px AND dy<20px = duplicate)
  - Keep highest-confidence box when deduplicating
  - Confidence gate: drop any region below 0.6
  - Fallback: if paddle confidence < 0.6 on a region, optionally run EasyOCR

### STAGE 2 - Spatial Clustering (REPLACE paragraph=True)
  - REMOVE EasyOCR paragraph=True flag
  - Use DBSCAN (eps=50, min_samples=1) on y-coordinates of bboxes
  - Sort clusters by mean y-position (top cluster = top caption)
  - Within each cluster, sort regions left-to-right by x_center
  - Output structured dict: {top_text, bottom_text, all_text, num_regions}

### STAGE 3 - BiLSTM Integration (KEEP EXISTING)
  - Feed result['all_text'] to existing tokenizer + Bi-LSTM
  - Return combined output: ocr_result merged with hate speech score

### REQUIREMENTS
  - Class: MemeOCRPipeline(use_gpu=False)
  - Main method: pipeline.extract(image_path) -> dict
  - Full try/except with logging.warning on per-version failures
  - requirements.txt: paddleocr>=2.7, paddlepaddle, scikit-learn, opencv-python
  - Keep EasyOCR as optional ensemble fallback (only if paddle conf < 0.6)
  - Add unit test file: tests/test_ocr_pipeline.py with 3 test cases:
    (1) preprocess returns 3 image versions
    (2) NMS removes near-duplicate boxes
    (3) DBSCAN correctly separates top/bottom clusters

Output ONLY the refactored code files. No explanations outside code comments.
Files to produce:
  src/ocr_pipeline.py       (MemeOCRPipeline class)
  src/hate_speech.py        (integration wrapper)
  tests/test_ocr_pipeline.py
  requirements.txt
```

---

## 7. Migration & Rollback

### 7.1 Migration Steps

1. Keep `src/craft_pipeline.py` intact — do not delete (rollback target)
2. Install new dependencies: `pip install paddleocr paddlepaddle scikit-learn`
3. First PaddleOCR run downloads PP-OCRv4 models (~50MB, one-time only)
4. Run unit tests: `pytest tests/test_ocr_pipeline.py`
5. Smoke test on 5 known memes, verify `top_text` / `bottom_text` separation
6. Update import in `main.py`: `MemeOCRPipeline` from `src.ocr_pipeline`

### 7.2 Rollback

- If PaddleOCR fails: revert import to `src/craft_pipeline.py`
- No model retraining required — Bi-LSTM is unchanged
- Environment variable `MEME_OCR_BACKEND=craft` forces legacy mode

> ⚠️ **Risk — First-Run Model Download:** PaddleOCR downloads PP-OCRv4 models on first run (~50MB). In air-gapped or CI environments, pre-download models and set `model_dir` paths explicitly in the `PaddleOCR()` constructor.

---

*Meme Hate Speech Detection — OCR Pipeline PRD v1.0*