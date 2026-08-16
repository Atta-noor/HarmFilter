"""
Advanced LSTM/BiLSTM Training Script for Roman Urdu Hate Speech Detection
Uses proper tokenization and sequence-based approach
"""
import os
import numpy as np
import pandas as pd
import nltk
import re
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("="*60)
print("Advanced LSTM/BiLSTM Model Training - Roman Urdu")
print("="*60)

# ==================== Configuration ====================
MAX_WORDS = 10000  # Maximum vocabulary size
MAX_SEQUENCE_LENGTH = 100  # Maximum length of input sequences
EMBEDDING_DIM = 128  # Embedding dimension
LSTM_UNITS = 64  # Number of LSTM units
BATCH_SIZE = 64
EPOCHS = 20
USE_BILSTM = True  # Set to False for regular LSTM
MODEL_NAME = 'bilstm_model_roman_urdu.h5' if USE_BILSTM else 'lstm_model_roman_urdu.h5'

# ==================== Load and Preprocess Data ====================
print("\n[1/6] Loading dataset...")
df = pd.read_csv('MultiLanguageTrainDataset.csv')

# Filter for Roman Urdu (language code 8)
df = df[df['language'] == 8].copy()

# Map labels: 0.0 (Hate Speech) -> 0 (Hate Speech), 1.0 (Not Hate/Neutral) -> 2 (Neither)
# We'll use 3 classes to match the English model structure
df['class'] = df['label'].map({0.0: 0, 1.0: 2})

# Remove any rows with invalid labels
df = df[df['class'].notna()]

# Rename text column for consistency
df.rename(columns={'text': 'sentence'}, inplace=True)

print(f"Dataset loaded: {len(df)} samples (Roman Urdu only)")
print(f"Class distribution:\n{df['class'].value_counts().sort_index()}")
print(f"Label mapping: 0.0 (Hate Speech) -> 0 (Hate Speech), 1.0 (Neutral) -> 2 (Neither)")

# ==================== Text Preprocessing ====================
print("\n[2/6] Preprocessing text...")

def preprocess_text_roman_urdu(text):
    """Clean and preprocess Roman Urdu text"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags (keep the text after #)
    text = re.sub(r'@\w+|#', '', text)
    
    # Remove punctuation (keeping Roman Urdu specific characters)
    punctuation_signs = list("?:!.,;")
    for punct_sign in punctuation_signs:
        text = text.replace(punct_sign, '')
    
    # Remove \n, \t, extra spaces, quotes
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('    ', ' ')
    text = text.replace('"', '')
    text = text.replace("'", '')
    
    # Note: We don't remove stopwords for Roman Urdu as it may affect meaning
    # Roman Urdu doesn't have standard stopword lists
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text

# Apply preprocessing
df['sentence_processed'] = df['sentence'].apply(preprocess_text_roman_urdu)

# Remove empty texts
df = df[df['sentence_processed'].str.len() > 0]
print(f"After preprocessing: {len(df)} samples")

# ==================== Tokenization ====================
print("\n[3/6] Tokenizing text...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['sentence_processed'])

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(df['sentence_processed'])

# Pad sequences to same length
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
y = df['class'].values

print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")

# ==================== Split Data ====================
print("\n[4/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y, 
    random_state=42
)

# Convert labels to categorical (3 classes to match English model)
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ==================== Build Model ====================
print("\n[5/6] Building model...")

model = Sequential()

# Embedding layer
model.add(Embedding(
    input_dim=min(MAX_WORDS, len(tokenizer.word_index) + 1),
    output_dim=EMBEDDING_DIM,
    mask_zero=True
))

# Spatial dropout for regularization
model.add(SpatialDropout1D(0.2))

# LSTM layers
if USE_BILSTM:
    # Bidirectional LSTM (reads text both ways - better for context)
    model.add(Bidirectional(LSTM(
        LSTM_UNITS,
        dropout=0.2,
        recurrent_dropout=0.2,
        return_sequences=False
    )))
    print("Using Bidirectional LSTM")
else:
    # Regular LSTM
    model.add(LSTM(
        LSTM_UNITS,
        dropout=0.2,
        recurrent_dropout=0.2,
        return_sequences=False
    ))
    print("Using Regular LSTM")

# Dense layers
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# ==================== Callbacks ====================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_NAME,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
]

# ==================== Train Model ====================
print("\n[6/6] Training model...")
print(f"Training for up to {EPOCHS} epochs...")
print(f"Batch size: {BATCH_SIZE}")
print("-"*60)

history = model.fit(
    X_train, y_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    verbose=1
)

# ==================== Evaluate Model ====================
print("\n" + "="*60)
print("Model Evaluation")
print("="*60)

# Load best model
model.load_weights(MODEL_NAME)

# Predictions
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report - only include classes that exist in the data
unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
target_names = [class_labels[cls] for cls in unique_classes]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          labels=unique_classes,
                          target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
print("\nConfusion Matrix:")
print(f"Classes: {unique_classes}")
print(cm)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('lstm_training_history_roman_urdu.png', dpi=300, bbox_inches='tight')
print("\nTraining history saved to 'lstm_training_history_roman_urdu.png'")

# ==================== Save Tokenizer ====================
print("\n" + "="*60)
print("Saving Model and Tokenizer")
print("="*60)

# Save tokenizer
with open('tokenizer_roman_urdu.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved to 'tokenizer_roman_urdu.pkl'")

# Save model (already saved by ModelCheckpoint, but save again to be sure)
model.save(MODEL_NAME)
print(f"✓ Model saved to '{MODEL_NAME}'")

# Save model configuration
config = {
    'max_words': MAX_WORDS,
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'embedding_dim': EMBEDDING_DIM,
    'lstm_units': LSTM_UNITS,
    'use_bilstm': USE_BILSTM,
    'vocab_size': len(tokenizer.word_index) + 1,
    'language': 'roman_urdu'
}

with open('lstm_config_roman_urdu.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✓ Model configuration saved to 'lstm_config_roman_urdu.pkl'")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print(f"\nFiles created:")
print(f"  - {MODEL_NAME} (model weights)")
print(f"  - tokenizer_roman_urdu.pkl (text tokenizer)")
print(f"  - lstm_config_roman_urdu.pkl (model configuration)")
print(f"  - lstm_training_history_roman_urdu.png (training plots)")
print(f"\nYou can now use this model in the Streamlit UI!")

