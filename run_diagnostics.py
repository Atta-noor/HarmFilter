#!/usr/bin/env python3
"""
🔍 Hate Speech Detection System - Diagnostic Script
Identifies issues with image text extraction and model loading
"""

import os
import sys
from pathlib import Path

print("🔍 Hate Speech Detection System Diagnostics")
print("=" * 60)

# ============================================================================
# SECTION 1: Check Required Files
# ============================================================================
print("\n📋 SECTION 1: Checking Required Files")
print("-" * 60)

required_files = {
    "Models": [
        "bilstm_model.h5",
    ],
    "Tokenizers": [
        "tokenizer.pkl",
        "lstm_config.pkl",
    ],
    "Datasets": [
        "labeled_data.csv",
    ],
    "Python Scripts": [
        "app.py",
        "lstm_inference.py",
        "ocr_correction.py",
        "preprocessing.py",
    ]
}

all_files_ok = True
for category, files in required_files.items():
    print(f"\n{category}:")
    for f in files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        status = "✓" if exists else "✗"
        print(f"  {status} {f:<40} ({size:>10,} bytes)")
        if not exists:
            all_files_ok = False

# ============================================================================
# SECTION 2: Check Python Dependencies
# ============================================================================
print("\n\n📦 SECTION 2: Checking Python Dependencies")
print("-" * 60)

dependencies = {
    "Core ML": ["numpy", "pandas", "scikit-learn"],
    "Deep Learning": ["tensorflow", "keras"],
    "Image/OCR": ["PIL", "cv2", "easyocr"],
    "Text Processing": ["nltk", "autocorrect"],
    "Web Framework": ["streamlit"],
    "Serialization": ["joblib", "pickle"],
}

missing_deps = []
for category, modules in dependencies.items():
    print(f"\n{category}:")
    for module in modules:
        try:
            __import__(module)
            version = ""
            try:
                mod = sys.modules[module]
                if hasattr(mod, '__version__'):
                    version = f" (v{mod.__version__})"
            except:
                pass
            print(f"  ✓ {module:<20}{version}")
        except ImportError as e:
            print(f"  ✗ {module:<20} NOT INSTALLED")
            missing_deps.append(module)

# ============================================================================
# SECTION 3: Test Model Loading
# ============================================================================
print("\n\n🧠 SECTION 3: Testing Model Loading")
print("-" * 60)

try:
    print("\nEnglish BiLSTM Model:")
    from lstm_inference import LSTMPredictor
    predictor_en = LSTMPredictor('bilstm_model.h5', 'tokenizer.pkl', 'lstm_config.pkl', 'english')
    print("  ✓ Model loaded successfully")
    print(f"  ✓ Language: {predictor_en.language}")
    
    # Test prediction
    test_text = "This is a test sentence"
    label, conf, probs = predictor_en.predict(test_text, return_probabilities=True)
    print(f"  ✓ Test prediction successful: {label} ({conf:.1f}%)")
    
except Exception as e:
    print(f"  ✗ Error: {str(e)}")

# ============================================================================
# SECTION 4: Test OCR Engine
# ============================================================================
print("\n\n📸 SECTION 4: Testing OCR Engine")
print("-" * 60)

print("\nChecking EasyOCR...")
try:
    import easyocr
    print("  ✓ EasyOCR imported successfully")
    
    print("  ℹ Initializing EasyOCR reader (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("  ✓ EasyOCR reader initialized successfully")
    
    # Test with a simple image if available
    test_image_path = None
    if os.path.exists('test_image.jpg'):
        test_image_path = 'test_image.jpg'
    elif os.path.exists('test_image.png'):
        test_image_path = 'test_image.png'
    
    if test_image_path:
        print(f"  ℹ Testing with {test_image_path}...")
        results = reader.readtext(test_image_path, paragraph=True)
        print(f"  ✓ OCR extraction successful ({len(results)} text blocks found)")
    else:
        print("  ℹ No test image found (test_image.jpg/png)")
        
except ImportError:
    print("  ✗ EasyOCR not installed")
except Exception as e:
    print(f"  ✗ Error: {str(e)}")

print("\nChecking Tesseract (optional)...")
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    print("  ✓ Tesseract available")
except:
    print("  ℹ Tesseract not available (optional)")

# ============================================================================
# SECTION 5: Test Spell Correction
# ============================================================================
print("\n\n✏️  SECTION 5: Testing Spell Correction")
print("-" * 60)

try:
    from ocr_correction import correct_ocr_text
    
    test_cases = [
        "th1s is a t3st",
        "hello w0rld",
        "teh quick brown fox",
    ]
    
    print("\nSpelling correction tests:")
    for test in test_cases:
        corrected = correct_ocr_text(test)
        print(f"  {test:<30} → {corrected}")
    print("  ✓ Spell correction working")
    
except Exception as e:
    print(f"  ✗ Error: {str(e)}")

# ============================================================================
# SECTION 6: Check Image Preprocessing
# ============================================================================
print("\n\n🖼️  SECTION 6: Testing Image Preprocessing")
print("-" * 60)

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    import cv2
    
    print("  ✓ PIL imported successfully")
    print("  ✓ OpenCV imported successfully")
    
    # Create a test image
    test_img = Image.new('RGB', (500, 300), color='white')
    print("  ✓ Can create test images")
    print(f"  ✓ Test image size: {test_img.size}")
    
except Exception as e:
    print(f"  ✗ Error: {str(e)}")

# ============================================================================
# SECTION 7: Check Streamlit
# ============================================================================
print("\n\n🌐 SECTION 7: Testing Streamlit")
print("-" * 60)

try:
    import streamlit as st
    print(f"  ✓ Streamlit installed (v{st.__version__})")
except ImportError:
    print("  ✗ Streamlit not installed or corrupted")

# ============================================================================
# SECTION 8: Summary & Recommendations
# ============================================================================
print("\n\n" + "=" * 60)
print("📊 SUMMARY")
print("=" * 60)

if not all_files_ok:
    print("\n⚠️  Issue: Some required files are missing!")
    print("\nTo fix:")
    print("  1. Run: python train_lstm_model.py")

if missing_deps:
    print(f"\n⚠️  Issue: {len(missing_deps)} Python dependencies missing:")
    for dep in missing_deps:
        print(f"    - {dep}")
    print("\nTo fix:")
    print("  pip install -r requirements.txt --no-cache-dir")

print("\n" + "=" * 60)
print("✅ Diagnostic complete!")
print("=" * 60)

# ============================================================================
# SECTION 9: Quick Start Commands
# ============================================================================
print("\n\n🚀 QUICK START COMMANDS")
print("=" * 60)

print("\n1. If models are missing:")
print("   python train_lstm_model.py")

print("\n2. If dependencies are missing:")
print("   pip install -r requirements.txt --no-cache-dir")

print("\n3. To run the app:")
print("   streamlit run app.py")

print("\n4. For debugging:")
print("   streamlit run app.py --logger.level=debug")

print("\n5. To use different port (if 8501 is busy):")
print("   streamlit run app.py --server.port 8502")

print("\n" + "=" * 60)
