"""
LSTM/BiLSTM Inference Utilities for Hate Speech Detection
"""
import numpy as np
import pickle
import re
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class LSTMPredictor:
    """LSTM Model Predictor for Hate Speech Detection"""
    
    def __init__(self, model_path='bilstm_model.h5', tokenizer_path='tokenizer.pkl', config_path='lstm_config.pkl', language='english'):
        """
        Initialize the LSTM predictor
        
        Args:
            model_path: Path to saved Keras model
            tokenizer_path: Path to saved tokenizer
            config_path: Path to model configuration
            language: Kept for backward compatibility (English is the only supported language)
        """
        self.model = None
        self.tokenizer = None
        self.config = None
        self.language = 'english'
        self.stop_words = set(stopwords.words('english'))
        
        try:
            
            print(f"Loading model from {model_path}...")
            self.model = load_model(model_path)
            print("[OK] Model loaded successfully")
            
           
            print(f"Loading tokenizer from {tokenizer_path}...")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("[OK] Tokenizer loaded successfully")
            
            
            print(f"Loading configuration from {config_path}...")
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print("[OK] Configuration loaded successfully")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required file not found: {e}. Please train the model first.")
        except Exception as e:
            raise Exception(f"Error loading model files: {e}")
    
    def preprocess_text(self, text):
        """
        Preprocess text to match training pipeline
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        if pd.isna(text) or text == '':
            return ''
        
        
        text = str(text).lower()
        
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        
        text = re.sub(r'@\w+|#', '', text)
        
        
        punctuation_signs = list("?:!.,;")
        for punct_sign in punctuation_signs:
            text = text.replace(punct_sign, '')
        
        
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('    ', ' ')
        text = text.replace('"', '')
        
        
        if self.language == 'english':
            
            text = text.replace("'s", "")
            
            words = text.split()
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
        else:
            
            text = text.replace("'", '')
            
        
        
        text = ' '.join(text.split())
        
        return text
    
    def predict(self, text, return_probabilities=False):
        """
        Predict the class of input text
        
        Args:
            text: Input text string
            return_probabilities: If True, return probability distribution
            
        Returns:
            If return_probabilities=False: (predicted_label, confidence)
            If return_probabilities=True: (predicted_label, confidence, probabilities)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded. Please initialize the predictor first.")
        
        
        processed_text = self.preprocess_text(text)
        
        if not processed_text or len(processed_text.strip()) == 0:
            
            probabilities = np.array([0.33, 0.33, 0.34])
            predicted_class = 2  
            confidence = 0.34
        else:
            
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            
            
            max_len = self.config.get('max_sequence_length', 100)
            padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
            
           
            probabilities = self.model.predict(padded_sequence, verbose=0)[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        
        class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        predicted_label = class_labels[predicted_class]
        confidence_percent = confidence * 100
        
        if return_probabilities:
            return predicted_label, confidence_percent, probabilities
        else:
            return predicted_label, confidence_percent
    
    def predict_batch(self, texts):
        """
        Predict classes for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of (predicted_label, confidence, probabilities) tuples
        """
        results = []
        for text in texts:
            result = self.predict(text, return_probabilities=True)
            results.append(result)
        return results


def load_lstm_predictor(model_path='bilstm_model.h5', language='english'):
    """
    Convenience function to load the English LSTM predictor.

    Args:
        model_path: Path to model file
        language: Kept for backward compatibility (English is the only supported language)

    Returns:
        LSTMPredictor instance
    """
    return LSTMPredictor(model_path, 'tokenizer.pkl', 'lstm_config.pkl', language='english')



if __name__ == "__main__":
    print("Testing LSTM Predictor...")
    print("="*60)
    
    try:
        
        predictor = load_lstm_predictor()
        
        
        test_texts = [
            "I hate all people from that country",
            "That's so stupid and annoying",
            "The weather is nice today",
            "I love learning about machine learning"
        ]
        
        print("\nPredictions:")
        print("-"*60)
        
        for i, text in enumerate(test_texts, 1):
            label, confidence, probs = predictor.predict(text, return_probabilities=True)
            print(f"\nTest {i}: {text[:50]}...")
            print(f"  Prediction: {label}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Probabilities:")
            print(f"    - Hate Speech: {probs[0]*100:.2f}%")
            print(f"    - Offensive Language: {probs[1]*100:.2f}%")
            print(f"    - Neither: {probs[2]*100:.2f}%")
        
        print("\n" + "="*60)
        print("Testing complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using 'train_lstm_model.py'")
    except Exception as e:
        print(f"Error: {e}")

