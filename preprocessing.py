import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import joblib


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess a single text following the same pipeline as training
    """
    if pd.isna(text) or text == '':
        return ''
    
    
    text = str(text).lower()
    
    
    punctuation_signs = list("?:!.,;")
    for punct_sign in punctuation_signs:
        text = text.replace(punct_sign, '')
    
    
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('    ', ' ')
    text = text.replace('"', '')
    text = text.replace("'s", "")
    
    
    stop_words = list(stopwords.words('english'))
    for stop_word in stop_words:
        regex_stopword = r"\b" + re.escape(stop_word) + r"\b"
        text = re.sub(regex_stopword, '', text)
    
    
    text = ' '.join(text.split())
    
    return text

def load_vectorizer(vectorizer_path='count_vectorizer.joblib'):
    """
    Load the CountVectorizer used during training
    """
    try:
        return joblib.load(vectorizer_path)
    except FileNotFoundError:
       
        print(f"Warning: {vectorizer_path} not found. Creating new vectorizer.")
        return None

def preprocess_and_vectorize(text, vectorizer=None):
    """
    Preprocess text and convert to feature vector
    """
    
    processed_text = preprocess_text(text)
    
    if vectorizer is None:
        raise ValueError("Vectorizer is required. Please train and save the vectorizer first.")
    
    
    text_vector = vectorizer.transform([processed_text]).toarray()
    
    return text_vector

def predict_text(text, model, vectorizer):
    """
    Predict the class of a text
    Returns: (predicted_class, confidence)
    """
    
    text_vector = preprocess_and_vectorize(text, vectorizer)
    
   
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    
   
    class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    predicted_label = class_labels[prediction]
    confidence = probabilities[prediction] * 100
    
    return predicted_label, confidence, probabilities

