"""
Script to train models and save them along with the vectorizer
This ensures we have all necessary files for prediction
"""
import os
import numpy as np 
import pandas as pd 
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import joblib

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Read dataset
print("Loading dataset...")
df = pd.read_csv('labeled_data.csv')
text = df['tweet'].tolist()
clas = df['class'].tolist()
df = pd.DataFrame({'tweet': text, 'class': clas})

print("Preprocessing data...")
# Convert to lowercase
df['tweet'] = df['tweet'].apply(lambda x: str(x).lower() if pd.notna(x) else '')

# Remove punctuation
punctuation_signs = list("?:!.,;")
for punct_sign in punctuation_signs:
    df['tweet'] = df['tweet'].str.replace(punct_sign, '', regex=False)

# Remove \n, \t, extra spaces, quotes, and 's
df['tweet'] = df['tweet'].apply(lambda x: str(x).replace('\n', ' '))
df['tweet'] = df['tweet'].apply(lambda x: str(x).replace('\t', ' '))
df['tweet'] = df['tweet'].str.replace('    ', ' ', regex=False)
df['tweet'] = df['tweet'].str.replace('"', '', regex=False)
df['tweet'] = df['tweet'].str.replace("'s", "", regex=False)

# Remove stopwords
stop_words = list(stopwords.words('english'))
for stop_word in stop_words:
    regex_stopword = r"\b" + re.escape(stop_word) + r"\b"
    df['tweet'] = df['tweet'].str.replace(regex_stopword, '', regex=True)

# Bag of Words
print("Creating feature vectors...")
cv = CountVectorizer(max_features=75)
X = cv.fit_transform(df['tweet']).toarray()
y = df['class']

# Save the vectorizer
print("Saving vectorizer...")
joblib.dump(cv, 'count_vectorizer.joblib')
print("Vectorizer saved to count_vectorizer.joblib")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train Random Forest
print("Training Random Forest...")
clf_rf = RandomForestClassifier(n_estimators=10)
clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")
joblib.dump(clf_rf, 'rf.joblib')
print("Random Forest saved to rf.joblib")

# Train Decision Tree
print("Training Decision Tree...")
clf_dt = tree.DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)
y_pred = clf_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
joblib.dump(clf_dt, 'decision.joblib')
print("Decision Tree saved to decision.joblib")

# Train AdaBoost
print("Training AdaBoost...")
clf_ada = AdaBoostClassifier(n_estimators=100)
clf_ada.fit(X_train, y_train)
y_pred = clf_ada.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Accuracy: {accuracy:.4f}")
joblib.dump(clf_ada, 'ada.joblib')
print("AdaBoost saved to ada.joblib")

print("\nAll models and vectorizer saved successfully!")
print("You can now use the UI to test predictions.")

