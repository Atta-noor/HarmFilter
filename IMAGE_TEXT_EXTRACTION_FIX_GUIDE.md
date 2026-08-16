# 🔧 Image Text Extraction - Troubleshooting & Fix Guide

## Quick Diagnosis

**What to check first:**

### Issue: "OCR Error" or blank text extraction
```bash
# 1. Check if EasyOCR is properly installed
python -c "import easyocr; print(easyocr.__version__)"

# Expected output: 1.7.0 or higher

# 2. If import fails, reinstall
pip uninstall easyocr -y
pip install easyocr>=1.7.0 --no-cache-dir
```

### Issue: Text extraction works but quality is poor
This is the most common problem. The OCR finds text but recognition accuracy is low.

---

## Solution 1: Fix EasyOCR Configuration

**File to modify**: `app.py` lines 165-183

**Current configuration** (in `run_fast_ocr` function):
```python
results = reader.readtext(
    img_array,
    paragraph=True,
    low_text=0.3,
    text_threshold=0.5,
    width_ths=0.4,
    mag_ratio=1.5,
    decoder='beamsearch'
)
```

**Improved configuration**:
```python
# For better meme/special font detection:
results = reader.readtext(
    img_array,
    paragraph=True,
    low_text=0.2,           # ↓ Lower = detect more text (was 0.3)
    text_threshold=0.4,     # ↓ Lower = less strict (was 0.5)
    width_ths=0.3,          # ↓ Better word separation (was 0.4)
    mag_ratio=2.0,          # ↑ Better tiny text (was 1.5)
    decoder='beamsearch',   # Keep (most accurate)
    contrast_ths=0.1        # ← ADD: Better contrast handling
)
```

**When to use each setting:**

| Setting | Small Text | Blurry | Poor Lighting | Crowded |
|---------|-----------|--------|---------------|---------|
| `mag_ratio` | 2.0-2.5 | 1.8 | 1.5 | 1.5 |
| `low_text` | 0.1 | 0.25 | 0.2 | 0.3 |
| `text_threshold` | 0.3 | 0.4 | 0.3 | 0.5 |

---

## Solution 2: Enhance Image Preprocessing

**File to modify**: `app.py` lines 144-162 (in `preprocess_image_variants` function)

**Current preprocessing**:
```python
def preprocess_image_variants(img):
    variants = []
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    pad = 40
    padded = Image.new('RGB', (img.width + 2*pad, img.height + 2*pad), (255, 255, 255))
    padded.paste(img, (pad, pad))
    
    img_np = np.array(padded)
    variants.append(('original', img_np))
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    variants.append(('inverted', inverted))
    
    return variants
```

**Improved preprocessing** (add contrast enhancement):
```python
def preprocess_image_variants(img):
    variants = []
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    
    # Optional: Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Increase contrast by 50%

    pad = 40
    padded = Image.new('RGB', (img.width + 2*pad, img.height + 2*pad), (255, 255, 255))
    padded.paste(img, (pad, pad))
    
    img_np = np.array(padded)
    variants.append(('original', img_np))
    
    # Variant 2: Inverted
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    inverted = cv2.bitwise_not(gray)
    variants.append(('inverted', inverted))
    
    # NEW VARIANT 3: Adaptive Histogram Equalization (for high-contrast detection)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    variants.append(('equalized', equalized))
    
    return variants
```

---

## Solution 3: Improve Spell Correction

**File to modify**: `ocr_correction.py` lines 1-40

**Add these character mappings**:
```python
OCR_CHAR_FIXES = {
    "0": "o",   # zero → letter o
    "1": "l",   # one → letter l
    "4": "i",   # four → letter i
    "5": "s",   # five → letter s
    "8": "b",   # eight → letter b
    "|": "l",   # pipe → letter l
    "!": "i",   # bang → letter i
    "@": "a",   # at → letter a
    "$": "s",   # dollar → letter s
    "[": "n",   # bracket → letter n
    "]": "j",   # bracket → letter j
    "{": "t",   # brace → letter t
    "}": "j",   # brace → letter j
    
    # ADDITIONAL MAPPINGS FOR MEME FONTS:
    "7": "T",   # seven → letter T
    "3": "e",   # three → letter e
    "6": "b",   # six → letter b
    "9": "g",   # nine → letter g
    "(": "c",   # paren → letter c
    ")": "d",   # paren → letter d
    "_": "-",   # underscore → hyphen
    "~": "-",   # tilde → hyphen
}

# NEW: Preserve common slang/internet words
PRESERVE_WORDS = {
    # Internet slang
    'ur', 'r', 'u', 'lol', 'lmao', 'smh', 'ngl', 'tbh', 
    'imho', 'aight', 'gonna', 'wanna', 'gotta', 'dunno',
    'dontcare', 'idc', 'bruh', 'ew', 'ok', 'yo', 'aye',
    # Common meme words
    'meme', 'derp', 'noob', 'pwn', 'leet', 'w00t',
}
```

**Update correction function**:
```python
def correct_ocr_text(text, word_confidences=None, confidence_threshold=0.80):
    """
    Post-OCR text correction
    1. Fix confused characters
    2. Preserve slang/vocab
    3. Apply spelling correction
    """
    speller = _build_speller()
    
    # Step 1: Replace confused characters
    fixed_text = text
    for ocr_char, correct_char in OCR_CHAR_FIXES.items():
        fixed_text = fixed_text.replace(ocr_char, correct_char)
    
    # Step 2: Split into words and correct with vocab preservation
    words = fixed_text.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        if not word:
            corrected_words.append(word)
            continue
            
        # Preserve known slang/vocab
        if word.lower() in PRESERVE_WORDS:
            corrected_words.append(word)
            continue
            
        # Skip if high confidence (only correct low-confidence words)
        if word_confidences and i < len(word_confidences):
            if word_confidences[i] > confidence_threshold:
                corrected_words.append(word)
                continue
        
        # Filter out garbage (single char/symbols)
        if len(word) <= 1 and not word.isalpha():
            continue
            
        # Apply spelling correction if available
        if speller:
            corrected_words.append(speller(word))
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)
```

---

## Solution 4: Add Multi-Engine OCR Fallback

**Create new file**: `ocr_engines.py`

```python
"""
Multiple OCR engines with automatic fallback
"""
import io
import sys
from PIL import Image
import numpy as np

class OCREngine:
    """Base OCR class"""
    def extract_text(self, img_array):
        raise NotImplementedError
    
    def name(self):
        raise NotImplementedError

class EasyOCREngine(OCREngine):
    """EasyOCR implementation"""
    def __init__(self, languages=['en'], gpu=False):
        import easyocr
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        
    def extract_text(self, img_array):
        results = self.reader.readtext(
            img_array,
            paragraph=True,
            mag_ratio=2.0,
            decoder='beamsearch'
        )
        
        if not results:
            return "", 0
        
        text_blocks = [r[1] for r in results]
        text = " \n ".join(text_blocks)
        char_count = sum(len(t.strip()) for t in text_blocks)
        
        return text, char_count
    
    def name(self):
        return "EasyOCR"

class PaddleOCREngine(OCREngine):
    """PaddleOCR as fallback (if installed)"""
    def __init__(self, lang='en'):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        
    def extract_text(self, img_array):
        # Convert numpy array to PIL for PaddleOCR
        if isinstance(img_array, np.ndarray):
            img = Image.fromarray(img_array)
        else:
            img = img_array
            
        results = self.ocr.ocr(np.array(img), cls=True)
        
        text_lines = []
        for line in results:
            if line:
                for word_info in line:
                    text_lines.append(word_info[1][0])
        
        text = " ".join(text_lines)
        return text, len(text)
    
    def name(self):
        return "PaddleOCR"

def extract_with_fallback(img_array, primary="easyocr", verbose=True):
    """
    Try primary OCR engine, fallback to secondary if it fails
    """
    engines = []
    
    try:
        engine = EasyOCREngine()
        engines.append(engine)
    except:
        if verbose:
            print("⚠️  EasyOCR not available")
    
    try:
        engine = PaddleOCREngine()
        engines.append(engine)
    except:
        if verbose:
            print("ℹ️  PaddleOCR not installed (optional)")
    
    if not engines:
        raise Exception("No OCR engines available!")
    
    best_text = ""
    best_chars = 0
    
    for engine in engines:
        try:
            if verbose:
                print(f"Trying {engine.name()}...")
            text, char_count = engine.extract_text(img_array)
            
            if char_count > best_chars:
                best_text = text
                best_chars = char_count
                if verbose:
                    print(f"✓ {engine.name()}: {char_count} characters extracted")
        except Exception as e:
            if verbose:
                print(f"✗ {engine.name()} failed: {str(e)}")
            continue
    
    return best_text, best_chars
```

---

## Solution 5: Add Debug/Logging

**Modify app.py** - add verbose logging:

```python
if extract_btn:
    with st.spinner("Extracting text from image using OCR..."):
        try:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            
            debug_info = []
            
            # Check image properties
            debug_info.append(f"Image size: {image.size}")
            debug_info.append(f"Image format: {image.format}")
            debug_info.append(f"Image mode: {image.mode}")
            
            with st.expander("Debug Information"):
                st.write("\n".join(debug_info))
            
            # ... rest of OCR code with added st.write() calls
            
            with st.expander("OCR Processing Steps"):
                st.write(f"1. Original image: {image.size}")
                st.write(f"2. Preprocessing variants: 3 (original, inverted, equalized)")
                st.write(f"3. Best variant: {label}")
                st.write(f"4. Characters extracted: {best_chars}")
                st.write(f"5. Spell correction applied")
                
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
```

---

## Testing Your Fixes

**Step 1**: Create a test image with text
```python
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Create test meme
img = Image.new('RGB', (400, 300), color='white')
draw = ImageDraw.Draw(img)

# Try to use a system font
try:
    font = ImageFont.truetype("arial.ttf", 40)
except:
    font = ImageFont.load_default()

draw.text((50, 50), "THIS IS A TEST", fill='black', font=font)
draw.text((50, 150), "HATE SPEECH DETECTION", fill='black', font=font)

img.save('test_meme.jpg')
print("✓ Test image created: test_meme.jpg")
```

**Step 2**: Test with diagnostic
```bash
python run_diagnostics.py
```

**Step 3**: Test extraction directly
```python
from app import get_ocr_reader, preprocess_image_variants, run_fast_ocr

reader = get_ocr_reader()
image = Image.open('test_meme.jpg')

variants = preprocess_image_variants(image)
for label, img_arr in variants:
    text, chars, _ = run_fast_ocr(reader, img_arr)
    print(f"{label}: {chars} chars - {text[:50]}...")
```

**Step 4**: Run the full app
```bash
streamlit run app.py
```

---

## Performance Tips

| Optimization | Speed Gain | Quality Impact |
|--------------|-----------|----------------|
| Reduce `mag_ratio` (1.5→1.0) | 30% faster | -5% accuracy |
| Reduce image upscaling | 20% faster | -3% accuracy |
| Use `decoder='greedy'` instead of 'beamsearch' | 40% faster | -10% accuracy |
| Cache OCR reader | 0% (next extract) | No impact |
| Parallel variant processing | 50% faster* | Same |

*Requires threading/async implementation

---

## Common Error Messages & Fixes

### Error: "ModuleNotFoundError: No module named 'easyocr'"
```bash
pip install easyocr paddlepaddle paddleocr
```

### Error: "CUDA out of memory"
```python
# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# OR manually specify in EasyOCR
reader = easyocr.Reader(['en'], gpu=False)
```

### Error: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Error: "Timeout waiting for OCR"
Add timeout wrapper:
```python
from functools import wraps
import signal

def timeout(seconds=30):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"OCR took longer than {seconds}s")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

@timeout(30)
def extract_text_timeout(reader, img_array):
    return reader.readtext(img_array)
```

---

## When to Use Each Solution

| Issue | Solution |
|-------|----------|
| No text at all | Solution 1 (OCR config) + Solution 2 (preprocessing) |
| Poor recognition quality | Solution 1 + Solution 3 (spell correction) |
| Slow extraction | Solution 2 + Solution 4 (parallel engines) |
| Crashes/errors | Solution 4 (fallback engines) + Solution 5 (debug logging) |
| Internet slang destroyed | Solution 3 (preserve words) |

---

## Next Steps

1. ✓ Read this guide
2. ✓ Run `python run_diagnostics.py`
3. ✓ Apply Solution 1 (OCR config)
4. ✓ Apply Solution 2 (preprocessing)
5. ✓ Test with `test_meme.jpg`
6. ✓ Apply Solution 3 if results still poor
7. ✓ Deploy and monitor

**Need help?** Check `PROJECT_ARCHITECTURE_OVERVIEW.md` for full system details.

