import keras_ocr
from PIL import Image
import sys
import numpy as np

def test_keras(img_path):
    print("Loading pipeline...")
    pipeline = keras_ocr.pipeline.Pipeline()
    print("Pipeline loaded.")
    
    img = Image.open(img_path).convert("RGB")
    
    # ── Upscale small images ──
    w, h = img.size
    if w < 1000:
        scale = 1000 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # ── Add padding around edges ──
    pad = 40
    padded = Image.new('RGB', (img.width + 2*pad, img.height + 2*pad), (255, 255, 255))
    padded.paste(img, (pad, pad))
    
    img_np = np.array(padded)
    
    import time
    start = time.time()
    results = pipeline.recognize([img_np])[0]
    print(f"Took {time.time() - start:.2f}s")
    
    print("Results:")
    for text, box in results:
        print(text)

try:
    test_keras(sys.argv[1])
except Exception as e:
    print(e)
