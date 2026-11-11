"""
Model Training Script.
Trains a TF-IDF + Logistic Regression classification model on labeled resumes.
Saves model artifacts, evaluates performance, and outputs honest metrics.
"""

import os
import sys
import json
import pickle
import datetime

# Ensure project root in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from modules.preprocessor import clean_text
from modules.data_generator import generate_resumes_csv

DATA_PATH = os.path.join(BASE_DIR, "data", "resumes_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "resume_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def train(data_path: str = DATA_PATH) -> dict:
    """Train, evaluate, and persist the resume classification model."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load or generate data
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Generating benchmark dataset...")
        df = generate_resumes_csv(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset: {len(df)} resumes across {df['category'].nunique()} categories.")
    print("Class distribution:\n", df['category'].value_counts())

    # 2. Text preprocessing
    print("\n[Step 1/4] Preprocessing resume texts...")
    cleaned_texts = [clean_text(text) for text in df["resume_text"].tolist()]
    labels = df["category"].tolist()

    # 3. Train-Test Split (stratified)
    print("[Step 2/4] Splitting dataset (75% train, 25% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")


    # 4. TF-IDF Vectorization
    print("[Step 3/4] Fitting TF-IDF Vectorizer (ngram=(1, 2), max_features=1500)...")
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Model Training (Logistic Regression)
    print("[Step 4/4] Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        C=1.5,
        solver="lbfgs"
    )
    classifier.fit(X_train_vec, y_train)

    # 6. Evaluation
    y_pred = classifier.predict(X_test_vec)
    accuracy = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    categories = sorted(list(classifier.classes_))
    conf_matrix = confusion_matrix(y_test, y_pred, labels=categories).tolist()

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (Test Split)")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print("\nDetailed Classification Report:")
    print(report_text)
    print("=" * 60)
    print("NOTE ON REAL-WORLD PERFORMANCE:")
    print("The above metrics reflect performance on this controlled benchmark dataset.")
    print("Real-world resumes vary widely in layout, domain jargon, and quality;")
    print("production deployment requires continuous training on diverse industry corpora.")
    print("=" * 60 + "\n")

    # 7. Persist Artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(classifier, f)
    print(f"Saved classifier model to: {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved TF-IDF vectorizer to: {VECTORIZER_PATH}")

    metadata = {
        "model_name": "Logistic Regression + TF-IDF",
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(accuracy, 4),
        "categories": categories,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "classification_report": report_dict,
        "confusion_matrix": conf_matrix,
        "disclaimer": "Benchmark educational dataset. Does not represent production accuracy across uncurated resumes."
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    train()

