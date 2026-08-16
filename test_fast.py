import cv2
import easyocr
import time

reader = easyocr.Reader(['en'], gpu=False)

def test_img(img_path):
    print("Testing:", img_path)
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    methods = [
        ("original", gray),
    ]
    
    # 1. High contrast inverted (white text -> dark text, black outline -> bright outline)
    v1 = cv2.bitwise_not(gray)
    methods.append(("inverted", v1))
    
    # 2. White isolation (what we used before)
    _, v2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    v2 = cv2.bitwise_not(v2)
    methods.append(("white_iso", v2))
    
    # 3. Edge detection (Canny) to find the black outlines
    edges = cv2.Canny(gray, 100, 200)
    methods.append(("canny", cv2.bitwise_not(edges)))
    
    for name, processed in methods:
        start = time.time()
        res = reader.readtext(processed)
        el = time.time() - start
        texts = [r[1] for r in res if r[2] > 0.15]
        print(f"[{name:<12}] {el:.1f}s -> {' | '.join(texts)}")

test_img(r"C:\Users\DELL\.gemini\antigravity\brain\8c5f1f9b-127e-45af-a20e-b1b95be9eb1c\media__1777047917465.png")
test_img(r"C:\Users\DELL\.gemini\antigravity\brain\8c5f1f9b-127e-45af-a20e-b1b95be9eb1c\media__1777047931795.png")
