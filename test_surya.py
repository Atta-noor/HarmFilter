from PIL import Image
import sys
from surya.recognition import RecognitionPredictor
from surya.detection import batch_text_detection
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor

def test_surya(img_path):
    img = Image.open(img_path).convert("RGB")
    
    det_processor, det_model = load_det_processor(), load_det_model()
    # det_results is a list of DetectionResult, we have 1 image so [0]
    det_results = batch_text_detection([img], det_model, det_processor)[0]
    
    bboxes = [line.bbox for line in det_results.bboxes]
    print(f"Found {len(bboxes)} bounding boxes.")
    
    recognizer = RecognitionPredictor()
    # Run recognition: [img], [langs], [bboxes]
    langs = ["en"]
    rec_results = recognizer([img], [langs], [bboxes])[0]
    
    print("\nResults:")
    for line in rec_results.text_lines:
        print(line.text)

try:
    test_surya(sys.argv[1])
except Exception as e:
    import traceback
    traceback.print_exc()
