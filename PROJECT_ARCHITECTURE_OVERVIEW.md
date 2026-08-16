# Project Architecture & Image Text Extraction Analysis

## 📋 Project Overview

This is a **Hate Speech Detection System** with dual capabilities:
- **Text Analysis**: Direct text input for hate speech classification
- **Image Analysis**: Extract text from images (memes) and then analyze them

**Tech Stack**:
- **Frontend**: Streamlit (web interface)
- **Backend**: Python with TensorFlow/Keras
- **Models**: BiLSTM (deep learning) for text classification
- **OCR**: EasyOCR for image text extraction
- **Support**: English & Roman Urdu languages

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT WEB APP (app.py)                  │
└─────────────────────────────────────────────────────────────────┘
                              |
                              v
                    ┌─────────────────┐
                    │  Input Mode     │
                    ├─────────────────┤
                    │ Text Analysis   │
                    │ Image Analysis  │
                    └─────────────────┘
                              |
          ┌───────────────────┴───────────────────┐
          |                                        |
    TEXT INPUT                            IMAGE INPUT
          |                                        |
          v                                        v
   ┌────────────┐                        ┌─────────────────┐
   │ LSTM Model │◄────────────────────►  │   OCR Engine    │
   │ (Inference)│                        │  (EasyOCR)      │
   └────────────┘                        └─────────────────┘
          |                                      |
          v                                      v
   ┌────────────┐                        ┌──────────────────┐
   │ Prediction │                        │ Text Extraction  │
   │ & Probs    │◄───────────────────────│ & Correction     │
   └────────────┘                        └──────────────────┘
          |
          v
   ┌────────────┐
   │ Results    │
   │ Display    │
   └────────────┘
```

---

## 📁 File Structure & Components

### Core Application Files
| File | Purpose |
|------|---------|
| **app.py** | Main Streamlit web interface with text & image analysis modes |
| **lstm_inference.py** | LSTM model loading and prediction logic |
| **preprocessing.py** | Text preprocessing pipeline |
| **ocr_correction.py** | Post-OCR spell correction and character fixing |

### Training Files
| File | Purpose |
|------|---------|
| **train_lstm_model.py** | Train English BiLSTM model |
| **train_lstm_model_roman_urdu.py** | Train Roman Urdu BiLSTM model |
| **train_and_save_models.py** | Train traditional ML models (RF, DT, AdaBoost) |

### Model Files (Generated)
| File | Purpose |
|------|---------|
| **bilstm_model.h5** | Trained English BiLSTM model |
| **bilstm_model_roman_urdu.h5** | Trained Roman Urdu BiLSTM model |
| **tokenizer.pkl** | English text tokenizer |
| **tokenizer_roman_urdu.pkl** | Roman Urdu text tokenizer |
| **lstm_config.pkl** | English model configuration |
| **lstm_config_roman_urdu.pkl** | Roman Urdu model configuration |

### Datasets
| File | Purpose |
|------|---------|
| **labeled_data.csv** | English hate speech dataset |
| **Hate Speech Roman Urdu (HS-RU-20).csv** | Roman Urdu dataset |
| **MultiLanguageTrainDataset.csv** | Multi-language dataset |
| **Rurdu_test.csv** | Roman Urdu test data |

### Testing Files
| File | Purpose |
|------|---------|
| **test_surya.py** | Alternative OCR test using Surya recognition model |
| **test_fast.py** | Quick OCR testing |
| **test_keras.py** | Model loading test |
| **test_prediction.py** | Prediction pipeline test |

---

## 🖼️ Image Text Extraction Pipeline

### Current Implementation (app.py, lines 114-250)

```
Image Upload
    ↓
[EasyOCR Initialization]
    ↓
[Image Preprocessing]
    ├─→ Convert to RGB
    ├─→ Upscale if <1000px width
    └─→ Generate variants:
        • Original (padded)
        • Inverted (high contrast)
    ↓
[Run EasyOCR]
    ├─→ Paragraph mode enabled
    ├─→ Beam search decoder
    └─→ Select variant with most text
    ↓
[Spell Correction] (ocr_correction.py)
    ├─→ Fix OCR character confusion
    ├─→ Auto-correct words
    └─→ Filter garbage tokens
    ↓
[Extract & Edit]
    ├─→ Show raw output
    ├─→ Allow user edits
    └─→ Display extraction
    ↓
[Feed to LSTM]
    └─→ Hate Speech Classification
```

### Key Parameters
- **EasyOCR Settings**:
  - `paragraph=True` → Group text into blocks
  - `decoder='beamsearch'` → High accuracy, slower
  - `mag_ratio=1.5` → Magnification for better text detection
  - `width_ths=0.4` → Word separation threshold
  - `text_threshold=0.5` → Confidence threshold
  - `low_text=0.3` → Detection threshold

- **Image Preprocessing**:
  - Auto-upscale if width < 1000px
  - Add 40px white padding
  - Generate inverted grayscale variant

- **Spell Correction** (ocr_correction.py):
  - Character confusion map (0→o, 1→l, 4→i, etc.)
  - Auto-correct with `autocorrect` library
  - Filter single-character garbage
  - Preserve vocab words (don't over-correct)

---

## ⚠️ Image Text Extraction Issues & Troubleshooting

### **Issue #1: OCR Engine Not Initializing**

**Symptoms**:
- "OCR Error: No module named 'easyocr'"
- App crashes on "Extract Text" button
- GPU/CUDA compatibility errors

**Root Causes**:
1. EasyOCR not installed or corrupted
2. Missing dependencies (paddlepaddle, paddleocr)
3. CUDA/GPU driver conflicts
4. Model files not downloaded

**Solutions**:
```bash
# Fresh install
pip uninstall easyocr paddlepaddle paddleocr -y
pip install easyocr>=1.7.0 --no-cache-dir
pip install paddlepaddle paddleocr>=2.7.3

# Or force CPU-only (faster setup)
pip install easyocr[cpu]>=1.7.0
```

---

### **Issue #2: No Text Extracted from Image**

**Symptoms**:
- Extraction completes but shows "No text could be extracted"
- Only partial text extracted from clear image
- Special fonts (Impact, meme fonts) not recognized

**Root Causes**:
1. Image too small or blurry
2. Font not compatible with EasyOCR
3. Text too close together
4. Language not specified (defaulting to single language)

**Solutions**:
```python
# In app.py (modify run_fast_ocr function):

# 1. Increase magnification
mag_ratio=2.0  # was 1.5, increase for tiny text

# 2. Lower detection threshold (catch more text)
low_text=0.2  # was 0.3

# 3. Enable language detection
results = reader.readtext(
    img_array,
    languages=['en', 'roman_urdu'],  # Add detected languages
    paragraph=True,
    mag_ratio=2.0,
)

# 4. Check image quality before processing
if img.size[0] < 300 or img.size[1] < 300:
    st.warning("Image is very small. Upscaling...")
```

---

### **Issue #3: Poor Character Recognition (OCR mistakes)**

**Symptoms**:
- "0" recognized as "O", "1" as "l"
- "S" as "5", "I" as "!"
- Misspelled output even with correction

**Root Causes**:
1. OCR confusion map incomplete in ocr_correction.py
2. Post-correction logic too aggressive
3. Low confidence words not caught

**Solutions**:
```python
# In ocr_correction.py, expand OCR_CHAR_FIXES:
OCR_CHAR_FIXES = {
    "0": "o",
    "1": "l",
    "4": "i",
    "5": "s",
    "8": "b",
    "|": "l",
    "!": "i",
    "@": "a",
    "$": "s",
    "[": "n",
    "]": "j",
    "{": "t",
    "}": "j",
    # ADD MORE:
    "7": "T",
    "3": "e",
    "6": "b",
    "9": "g",
    "(": "c",
    ")": ")",
}
```

---

### **Issue #4: Spell Correction Not Working**

**Symptoms**:
- Intentional misspellings corrected incorrectly
- Slang words destroyed
- Roman Urdu words not recognized

**Root Causes**:
1. `autocorrect` library aggressive on non-English
2. Confidence threshold too high/low
3. No vocabulary preservation for slang/casual language

**Solutions**:
```python
# In ocr_correction.py, filter-before-correct:

# Preserve common slang/meme vocabulary
PRESERVE_WORDS = {
    'ur', 'r', 'u', 'lol', 'lmao', 'smh', 'ngl',
    'tbh', 'imho', 'aight', 'gonna', 'wanna', 'gotta'
}

def correct_ocr_text(text, confidence_threshold=0.80):
    words = text.split()
    corrected = []
    for word in words:
        if word.lower() in PRESERVE_WORDS:
            corrected.append(word)
        else:
            corrected.append(speller(word))
    return ' '.join(corrected)
```

---

### **Issue #5: Memory/Performance Problems**

**Symptoms**:
- App hangs during OCR extraction
- "Memory Error" or "CUDA out of memory"
- Very slow extraction (>30 seconds)

**Root Causes**:
1. EasyOCR GPU memory leak
2. Large image not downscaled
3. Variant preprocessing creating duplicates
4. Model cache not cleared

**Solutions**:
```bash
# In terminal before running app:
export CUDA_VISIBLE_DEVICES=""  # Force CPU-only

# Or modify app initialization:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TensorFlow CPU only
```

```python
# In app.py, add cleanup:
@st.cache_resource
def get_ocr_reader():
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return reader

# After extraction, clear cache
if st.button("Clear Cache"):
    st.cache_resource.clear()
```

---

### **Issue #6: Alternative OCR (Surya) Not Integrated**

**Symptom**: `test_surya.py` exists but unused (orphaned code)

**Solution**: Consider integrating as fallback:
```python
# In app.py, add option:
ocr_engine = st.sidebar.radio(
    "OCR Engine:",
    ["EasyOCR (Recommended)", "Surya (Backup)", "Tesseract (Legacy)"]
)

if ocr_engine == "Surya (Backup)":
    from test_surya import test_surya
    extracted_text = test_surya(uploaded_image)
```

Status: **NOT YET INTEGRATED** - Only available as standalone test script.

---

## 🔧 Quick Diagnostic Checklist

```bash
# 1. Check all required files exist
ls -la bilstm_model.h5 tokenizer.pkl lstm_config.pkl

# 2. Test model loading
python -c "
from lstm_inference import LSTMPredictor
p = LSTMPredictor('bilstm_model.h5', 'tokenizer.pkl', 'lstm_config.pkl')
print('✓ Model loads successfully')
"

# 3. Test OCR engine
python -c "
import easyocr
reader = easyocr.Reader(['en'], gpu=False)
print('✓ EasyOCR loads successfully')
"

# 4. Test spell correction
python -c "
from ocr_correction import correct_ocr_text
result = correct_ocr_text('th1s is a t3st')
print(f'✓ Correction works: {result}')
"

# 5. Run Streamlit diagnostics
streamlit run app.py --logger.level=debug 2>&1 | tee app_debug.log
```

---

## 📊 Model Information

### English BiLSTM Model
- **Architecture**: Bidirectional LSTM with embedding layer
- **Training Data**: `labeled_data.csv`
- **Classes**: 0=Hate Speech, 1=Offensive, 2=Neither
- **Performance**: ~93% accuracy on test set
- **Tokenizer**: Vocab size 5,000+ tokens

### Roman Urdu BiLSTM Model
- **Architecture**: Same as English model
- **Training Data**: `Hate Speech Roman Urdu (HS-RU-20).csv`
- **Classes**: 0=Hate Speech, 1=Offensive, 2=Neither
- **Performance**: ~85-90% accuracy
- **Tokenizer**: Specialized for Roman Urdu romanization

---

## 🚀 Recommendations

1. **Better Error Handling**: Add try-catch with specific error messages
2. **Timeout Protection**: Add 30-second timeout to OCR extraction
3. **Alternative OCR**: Integrate Surya OCR as fallback
4. **Batch Processing**: Process multiple images at once
5. **Caching**: Cache extracted text to avoid re-extraction
6. **Model Versioning**: Maintain multiple model checkpoints
7. **Logging**: Add comprehensive logging for debugging

---

## 📝 Configuration Files Missing (Optional)

These can improve performance if added:
- `lstm_config_roman_urdu.pkl` ✓ (exists)
- `lstm_config.pkl` ✓ (exists)
- Need: Configuration for model hyperparameters documentation

