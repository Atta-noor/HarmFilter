"""
Local OCR pipeline: image preprocessing (grayscale, contrast, threshold, denoise),
run EasyOCR and Tesseract, merge results, and perform AI-style cleaning
to normalize and correct OCR text while preserving tone and offensive words.

Usage:
    from ocr_pipeline import extract_and_clean_text
    cleaned = extract_and_clean_text(pil_image)
"""
from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
import io

try:
    from autocorrect import Speller
except Exception:
    Speller = None
"""
Local OCR pipeline: image preprocessing (grayscale, contrast, threshold, denoise),
run EasyOCR and Tesseract, merge results, and perform AI-style cleaning
to normalize and correct OCR text while preserving tone and offensive words.

Usage:
    from ocr_pipeline import extract_and_clean_text
    cleaned = extract_and_clean_text(pil_image)
"""
from typing import Tuple, List
import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
import io

try:
    from autocorrect import Speller
except Exception:
    Speller = None

# OCR character confusion map
OCR_CHAR_MAP = {
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "|": "l",
}

# Words to preserve (slang, intentional misspellings). Expand if needed.
PRESERVE_WORDS = {
    'lol', 'lmao', 'bruh', 'wtf', 'idk', 'u', 'ur', 'teh'
}


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def enhance_contrast_and_grayscale(pil_img: Image.Image, contrast: float = 1.8, outline: bool = False, scale: float = 1.0, return_array: bool = False):
    """Conservative enhancer: upscale (optional), increase contrast, convert to grayscale.

    Parameters:
    - contrast: multiplier passed to PIL ImageEnhance.Contrast
    - outline: if True, add a light stroke-outline around edges to reinforce text
    - scale: upscaling factor (1.0 = no upscale). Small upscale (1.2-1.5) helps OCR.
    - return_array: if True, return a numpy uint8 array (grayscale), otherwise a PIL Image in 'L' mode.

    This function is intentionally conservative compared with earlier heavy preprocessing
    so it improves readability without destroying subtle glyph features.
    """
    try:
        from PIL import ImageEnhance

        # Ensure RGB for consistent contrast enhancement
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Optional lightweight upscaling
        if scale and scale != 1.0:
            w, h = pil_img.size
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(float(contrast))

        # Convert to grayscale numpy for optional OpenCV ops
        gray = np.array(enhanced.convert('L'))

        # Apply CLAHE for local contrast enhancement when available
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

        if outline:
            # Light edge detection and dilation to reinforce thin strokes
            edges = cv2.Canny(gray, 60, 150)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            edges = cv2.dilate(edges, k, iterations=1)
            mask = (edges > 0).astype('uint8') * 255
            # Blend mask with gray image (subtle)
            gray = cv2.addWeighted(gray, 0.9, mask, 0.4, 0)

        gray = np.clip(gray, 0, 255).astype('uint8')

        if return_array:
            return gray
        return Image.fromarray(gray).convert('L')
    except Exception:
        try:
            # best-effort fallback
            out = pil_img.convert('L')
            if return_array:
                return np.array(out)
            return out
        except Exception:
            return pil_img


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def preprocess_image_cv2(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale, increase contrast (CLAHE), threshold, and denoise."""
    # ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Binarization (Otsu)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise (fastNlMeans)
    denoised = cv2.fastNlMeansDenoising(thresh, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    return opened


_EASYOCR_READER = None


def _get_easyocr_reader(lang_list=None):
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        langs = lang_list or ['en']
        # suppress verbose stdout during initialization
        _EASYOCR_READER = easyocr.Reader(langs, gpu=False, verbose=False)
    return _EASYOCR_READER


def run_easyocr_on_image(img: np.ndarray, paragraph: bool = True) -> (str, int):
    """Run EasyOCR on the image and return (text, char_count).

    Uses tuned parameters for meme-style images and returns the extracted
    text along with a simple character count used to choose the best variant.
    """
    try:
        reader = _get_easyocr_reader()
        results = reader.readtext(
            img,
            paragraph=paragraph,
            low_text=0.2,
            text_threshold=0.4,
            width_ths=0.3,
            mag_ratio=2.0,
            decoder='beamsearch',
            contrast_ths=0.1,
        )
        if not results:
            return "", 0
        if paragraph:
            # results may be nested blocks when paragraph=True
            text_blocks = [r[1] for r in results if isinstance(r, (list, tuple)) and len(r) > 1]
            text = " \n ".join(text_blocks)
        else:
            text = " ".join([r[1] for r in results])
        char_count = sum(len(t.strip()) for t in text.splitlines())
        return text, char_count
    except Exception:
        return "", 0


# Debug info exported for callers
OCR_LAST_DEBUG = {}


def run_tesseract_on_image(img_pil: Image.Image) -> str:
    # Use Tesseract's default English
    try:
        text = pytesseract.image_to_string(img_pil, lang='eng')
    except Exception:
        text = ""
    return text


def merge_ocr_outputs(easy_text: str, tesseract_text: str) -> str:
    """Merge outputs conservatively: prefer easy_text segments, append tesseract where missing."""
    # Simple merge: if easy_text is not empty, use it; otherwise use tesseract.
    if easy_text and easy_text.strip():
        # If tesseract contains extra lines not in easy_text, append them
        easy_set = set([s.strip().lower() for s in easy_text.splitlines() if s.strip()])
        lines = [l for l in easy_text.splitlines() if l.strip()]
        for l in tesseract_text.splitlines():
            if l.strip() and l.strip().lower() not in easy_set:
                lines.append(l.strip())
        return " \n ".join(lines)
    return tesseract_text


def fix_ocr_chars_word(word: str) -> str:
    out = []
    for ch in word:
        if ch in OCR_CHAR_MAP:
            out.append(OCR_CHAR_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


def normalize_word(word: str) -> str:
    w = word
    # Remove surrounding punctuation except internal apostrophes
    w = w.strip("\"\'`~<>[]{}()*:;,.!?/\\|@#$%^&*_+=")

    # Fix common OCR char swaps
    w = fix_ocr_chars_word(w)

    # Reduce repeated characters but keep emotive length (max 2 repeats)
    import re
    w = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", w)

    # If it's in preserve set, keep as is
    if w.lower() in PRESERVE_WORDS:
        return w

    # Spell correction (use autocorrect if available)
    if Speller is not None and len(w) > 1 and any(c.isalpha() for c in w):
        try:
            sp = Speller(lang='en', fast=True)
            corrected = sp(w)
            return corrected
        except Exception:
            return w

    return w


def clean_text(raw: str) -> str:
    """Apply token-level normalization and rejoin into cleaned text."""
    if not raw:
        return ""
    # Split on whitespace and punctuation preserved
    import re
    tokens = re.split(r"(\s+)", raw)
    cleaned_tokens = []
    for t in tokens:
        if t.isspace():
            cleaned_tokens.append(t)
            continue
        cleaned_tokens.append(normalize_word(t))

    cleaned = "".join(cleaned_tokens)

    # Final cleanup: collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_and_clean_text(pil_image: Image.Image) -> Tuple[str, str]:
    """Full pipeline: preprocess, OCR (EasyOCR + Tesseract), merge, clean.

    Returns (cleaned_text, raw_merged_text)
    """
    # Create multiple preprocessing variants to handle meme-style text (white text + outline)
    variants = []

    # Base: conservative enhancement (grayscale + CLAHE)
    base = enhance_contrast_and_grayscale(pil_image, contrast=1.6, outline=False, scale=1.2, return_array=True)
    variants.append(('base', base))

    # Inverted grayscale (white text -> black)
    try:
        inv = cv2.bitwise_not(base)
        variants.append(('inverted', inv))
    except Exception:
        pass

    # Equalized (CLAHE already applied in enhancer) but try a raw equalize
    try:
        gray_raw = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray_raw)
        variants.append(('equalized', equalized))
    except Exception:
        pass

    # Outline-enhanced variant (stronger stroke emphasis)
    try:
        outline_img = enhance_contrast_and_grayscale(pil_image, contrast=1.6, outline=True, scale=1.2, return_array=True)
        variants.append(('outline', outline_img))
    except Exception:
        pass

    best_text = ""
    best_count = 0
    best_variant_name = None

    for name, var in variants:
        try:
            # Ensure image passed to EasyOCR is in uint8 grayscale or RGB
            img_for_ocr = var
            if len(img_for_ocr.shape) == 2:
                pass
            elif img_for_ocr.shape[2] == 3:
                img_for_ocr = cv2.cvtColor(img_for_ocr, cv2.COLOR_BGR2GRAY)

            text, cnt = run_easyocr_on_image(img_for_ocr, paragraph=True)
            # If empty, also try paragraph=False (word-level)
            if not text:
                text, cnt = run_easyocr_on_image(img_for_ocr, paragraph=False)

            if cnt > best_count:
                best_count = cnt
                best_text = text
                best_variant_name = name
        except Exception:
            continue

    # Fallback to Tesseract on the best variant (or base) if EasyOCR didn't find text
    if not best_text or best_count == 0:
        try:
            chosen = variants[0][1] if variants else np.array(pil_image.convert('L'))
            pil_for_tess = cv2_to_pil(cv2.cvtColor(chosen, cv2.COLOR_GRAY2BGR))
            tess_text = run_tesseract_on_image(pil_for_tess)
            merged = merge_ocr_outputs(best_text, tess_text)
        except Exception:
            merged = best_text
    else:
        merged = best_text

    cleaned = clean_text(merged)
    # Populate debug info for callers
    try:
        OCR_LAST_DEBUG.clear()
        OCR_LAST_DEBUG['variants_tested'] = []
        for name, var in variants:
            try:
                # quick char count estimate by running easyocr in word mode (cheap)
                txt, cnt = run_easyocr_on_image(var, paragraph=False)
            except Exception:
                txt, cnt = "", 0
            OCR_LAST_DEBUG['variants_tested'].append((name, int(cnt)))
        OCR_LAST_DEBUG['best_variant'] = best_variant_name
        OCR_LAST_DEBUG['best_count'] = int(best_count)
    except Exception:
        pass

    return cleaned, merged


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    img = Image.open(path).convert('RGB')
    cleaned, raw = extract_and_clean_text(img)
    # Only output cleaned text as requested
    