"""
Simple test script to verify the prediction pipeline works
"""
import joblib
from preprocessing import predict_text

def test_prediction():
    """Test the prediction with sample texts"""
    
    # Load model and vectorizer
    try:
        model = joblib.load('rf.joblib')
        vectorizer = joblib.load('count_vectorizer.joblib')
        print("✓ Models loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error loading models: {e}")
        print("Please run 'train_and_save_models.py' first")
        return
    
    # Test texts
    test_texts = [
        "I hate all people from that country",
        "That's so stupid and annoying",
        "The weather is nice today",
        "I love learning about machine learning"
    ]
    
    print("\n" + "="*60)
    print("Testing Predictions")
    print("="*60 + "\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: {text[:50]}...")
        try:
            predicted_label, confidence, probabilities = predict_text(text, model, vectorizer)
            print(f"  Prediction: {predicted_label}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Probabilities: Hate={probabilities[0]*100:.1f}%, "
                  f"Offensive={probabilities[1]*100:.1f}%, "
                  f"Neither={probabilities[2]*100:.1f}%")
            print()
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    print("="*60)
    print("Testing complete!")

if __name__ == "__main__":
    test_prediction()

