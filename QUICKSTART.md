# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Models
Make sure `labeled_data.csv` is in the current directory, then run:
```bash
python train_and_save_models.py
```

This will take a few minutes and create:
- `rf.joblib` - Random Forest model
- `decision.joblib` - Decision Tree model  
- `ada.joblib` - AdaBoost model
- `count_vectorizer.joblib` - Text vectorizer

### Step 3: Launch the Web Interface
```bash
streamlit run app.py
```

The app will open automatically in your browser!

## 📝 Usage

1. **Select a Model**: Choose from Random Forest, Decision Tree, or AdaBoost in the sidebar
2. **Enter Text**: Type or paste the text you want to analyze
3. **Get Results**: Click "Analyze Text" to see:
   - Main prediction (Hate Speech / Offensive Language / Neither)
   - Confidence percentage
   - Detailed probability breakdown

## 🧪 Test the System

You can test the prediction pipeline with:
```bash
python test_prediction.py
```

## ⚠️ Troubleshooting

**"Model file not found"**
- Run `train_and_save_models.py` first

**"labeled_data.csv not found"**
- Make sure the CSV file is in the same directory as the scripts

**NLTK stopwords error**
- The script will download them automatically, but if issues persist:
  ```python
  import nltk
  nltk.download('stopwords')
  ```

## 📊 Model Performance

- Random Forest: ~84.1% accuracy
- Decision Tree: ~82.2% accuracy  
- AdaBoost: ~84.7% accuracy

Enjoy using the Hate Speech Detection System! 🎉

