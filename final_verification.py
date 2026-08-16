#!/usr/bin/env python3
"""Final verification that all fixes are working"""

import os
import sys

print('🔍 FINAL VERIFICATION')
print('=' * 60)

# Check files
files_ok = all(os.path.exists(f) for f in [
    'app.py', 'lstm_inference.py', 'ocr_correction.py',
    'bilstm_model.h5', 'bilstm_model_roman_urdu.h5',
    'tokenizer.pkl', 'tokenizer_roman_urdu.pkl'
])
print(f'\n01. All files present: {"✅" if files_ok else "❌"}')

# Check dependencies
deps_ok = True
missing = []
print('\n02. Checking dependencies:')
for mod in ['streamlit', 'tensorflow', 'keras', 'easyocr', 'sklearn', 'nltk', 'numpy', 'pandas', 'cv2', 'PIL']:
    try:
        __import__(mod)
        print(f'    ✅ {mod}')
    except ImportError:
        print(f'    ❌ {mod}')
        deps_ok = False

# Check models load
models_ok = True
print('\n03. Model loading:')
try:
    from lstm_inference import LSTMPredictor
    p = LSTMPredictor('bilstm_model.h5', 'tokenizer.pkl', 'lstm_config.pkl', 'english')
    print('    ✅ English BiLSTM model')
except Exception as e:
    models_ok = False
    print(f'    ❌ English BiLSTM: {str(e)[:50]}')

try:
    p2 = LSTMPredictor('bilstm_model_roman_urdu.h5', 'tokenizer_roman_urdu.pkl', 'lstm_config_roman_urdu.pkl', 'roman_urdu')
    print('    ✅ Roman Urdu BiLSTM model')
except Exception as e:
    models_ok = False
    print(f'    ❌ Roman Urdu BiLSTM: {str(e)[:50]}')

# Check OCR Correction
ocr_ok = True
print('\n04. OCR correction:')
try:
    from ocr_correction import PRESERVE_WORDS, OCR_CHAR_FIXES, correct_ocr_text
    print(f'    ✅ Character fixes: {len(OCR_CHAR_FIXES)} mappings')
    print(f'    ✅ Preserve words: {len(PRESERVE_WORDS)} words')
    
    test = correct_ocr_text('th1s is a t3st')
    if test == 'this is a test':
        print('    ✅ Spell correction working')
    else:
        print(f'    ❌ Spell correction: expected "this is a test", got "{test}"')
        ocr_ok = False
except Exception as e:
    ocr_ok = False
    print(f'    ❌ OCR error: {str(e)[:50]}')

# Check EasyOCR
print('\n05. EasyOCR engine:')
try:
    import easyocr
    print('    ✅ EasyOCR imported')
except Exception as e:
    print(f'    ❌ EasyOCR: {str(e)[:50]}')

print('\n' + '=' * 60)
if all([files_ok, deps_ok, models_ok, ocr_ok]):
    print('✅ ALL SYSTEMS READY - Successfully fixed!')
    print('Ready to deploy the application.')
    sys.exit(0)
else:
    print('⚠️  Some issues remain')
    sys.exit(1)
print('=' * 60)
