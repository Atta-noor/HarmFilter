<h1 align="center">🛡️ HarmFilter</h1>
<h3 align="center">Hate Speech & Offensive Language Detection — Text + Memes</h3>

<p align="center">
  A deep-learning NLP system that classifies English text as
  <b>Hate Speech</b>, <b>Offensive Language</b>, or <b>Neither</b> —
  from raw text <i>or</i> from images of memes via OCR.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/PaddleOCR-0062B0?style=flat-square&logo=baidu&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
</p>

---

## ✨ Overview

HarmFilter is an end-to-end content-moderation prototype. Its primary classifier is a
**Bidirectional LSTM** (Keras) trained to distinguish three classes, and it goes beyond
plain text: drop in a **meme image** and the app runs a multi-stage **OCR pipeline** to
extract the caption, then classifies it. It also lets you swap in two **HuggingFace**
transformer backends for comparison.

| Class | Label | Meaning |
|:---:|:---|:---|
| `0` | **Hate Speech** | Content promoting hatred or violence toward a group |
| `1` | **Offensive Language** | Offensive/profane but not hate speech |
| `2` | **Neither** | Neutral content |

---

## 🚀 Features

- **🧠 BiLSTM deep-learning classifier** (Keras) — the primary, locally-run model.
- **🖼️ Image / meme analysis** — a 4-stage PaddleOCR v4 pipeline (multi-view preprocessing →
  detection + NMS de-duplication → DBSCAN top/bottom caption clustering → text assembly),
  with an EasyOCR fallback for low-confidence regions.
- **🔄 Pluggable backends** — switch between the local BiLSTM and two HuggingFace models
  (zero-shot `distilbart-mnli` and `twitter-roberta-base-hate`) from the sidebar.
- **📊 Explainable output** — every prediction shows a confidence score and the full
  probability distribution across all three classes.
- **🖥️ Clean Streamlit UI** — pick analysis mode, language, and model with three toggles.

---

## 🏗️ How It Works

```
                    ┌──────────────────────────┐
   Text input ─────►│                          │
                    │   Preprocess (clean)     │──►  BiLSTM  ──►  [ Hate / Offensive / Neither ]
   Image input ─┐   │   tokenize + pad          │      or HuggingFace backend
                │   └──────────────────────────┘
                ▼
      ┌─────────────────────┐
      │  PaddleOCR pipeline │  preprocess (3 views) → detect + NMS → cluster (DBSCAN) → text
      └─────────────────────┘
```

The text-classification path lives in `lstm_inference.py` (`LSTMPredictor`); the OCR
backbone is `ocr_pipeline.py` (`MemeOCRPipeline`); `src/hate_speech.py` wires them together
for a programmatic OCR→classify entry point.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> First run also auto-downloads NLTK stopwords and PaddleOCR models (~50 MB).

### Train the models

Model artifacts (`*.h5`, `tokenizer.pkl`, `lstm_config.pkl`) and datasets are gitignored,
so train them locally first:

```bash
python train_lstm_model.py              # English BiLSTM
```

### Run the app

```bash
streamlit run app.py                    # http://localhost:8501
```

---

## 🖥️ Usage

**In the app:** choose an **analysis mode** (Text / Image) and a **model backend**, then
enter text or upload a meme and view the prediction with its confidence and probability
breakdown.

**From the command line:**

```bash
python ocr_pipeline.py <image>          # OCR only  -> JSON
python -m src.hate_speech <image>       # OCR + classify -> JSON with label
```

---

## 🧰 Tech Stack

**ML / NLP:** TensorFlow · Keras · BiLSTM · NLTK · scikit-learn · HuggingFace Transformers
**OCR / CV:** PaddleOCR v4 · OpenCV · EasyOCR · DBSCAN clustering
**App:** Streamlit · Python

---

## 📁 Project Structure

```
app.py                        # Streamlit web application (UI + backend selection)
lstm_inference.py             # LSTMPredictor — primary BiLSTM inference
ocr_pipeline.py               # MemeOCRPipeline — PaddleOCR v4 meme OCR
huggingface_inference.py      # Zero-shot transformer backend
huggingface_text_classifier.py# RoBERTa hate-speech backend
src/hate_speech.py            # End-to-end OCR -> classify wrapper (CLI)
train_lstm_model.py           # Training script (English BiLSTM)
requirements.txt
```

---

## 📌 Notes

- The BiLSTM preprocessing is kept **byte-for-byte consistent** with the training scripts —
  tokenizer indices depend on it.
- Built and tested on Windows / CPU; PaddlePaddle and NumPy versions are pinned for compatibility.

---

## 📄 License

Educational / research use.

<p align="center"><i>Built by <a href="https://github.com/Atta-noor">Atta Noor</a> — AI/ML · Full-Stack · Flutter</i></p>
