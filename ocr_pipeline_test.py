"""
Quick CLI test for the OCR pipeline.
Prints only the cleaned text (one-line) for easy piping.
"""
from PIL import Image
from ocr_pipeline import extract_and_clean_text
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ocr_pipeline_test.py <image>')
        sys.exit(1)
    img = Image.open(sys.argv[1]).convert('RGB')
    cleaned, raw = extract_and_clean_text(img)
    print(cleaned)
