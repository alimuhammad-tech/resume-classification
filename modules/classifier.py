"""
Classifier Wrapper Module.
Loads the trained TF-IDF vectorizer and Logistic Regression model
to predict categories and confidence distributions for resumes.
"""

from typing import Dict, Any, List, Tuple
import os
import pickle
import json
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "resume_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


class ResumeClassifier:
    """Wrapper for Logistic Regression inference and confidence estimation."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        vectorizer_path: str = VECTORIZER_PATH,
        metadata_path: str = METADATA_PATH
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.metadata_path = metadata_path
        self.model = None
        self.vectorizer = None
        self.metadata = {}
        self.is_loaded = False
        self.load()

    def load(self) -> bool:
        """Load trained model, vectorizer, and metadata if available."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            self.is_loaded = True
            return True
        return False

    def predict(self, cleaned_text: str) -> Dict[str, Any]:
        """
        Predict category and confidence score for a preprocessed resume string.
        
        Returns:
            dict: {
                "category": str,
                "confidence": float (0-1),
                "probabilities": dict {category: prob},
                "sorted_rankings": list of (category, prob)
            }
        """
        if not self.is_loaded or self.model is None or self.vectorizer is None:
            raise RuntimeError("Model is not loaded. Train the model first via train_model.py.")

        # Transform text via TF-IDF
        features = self.vectorizer.transform([cleaned_text])

        # Predict class and probability
        probabilities = self.model.predict_proba(features)[0]
        classes = self.model.classes_

        class_prob_pairs = [
            (str(cls_name), float(prob))
            for cls_name, prob in zip(classes, probabilities)
        ]
        # Sort descending by probability
        class_prob_pairs.sort(key=lambda x: x[1], reverse=True)

        top_category, top_confidence = class_prob_pairs[0]

        return {
            "category": top_category,
            "confidence": round(top_confidence, 4),
            "probabilities": {cls: round(prob, 4) for cls, prob in class_prob_pairs},
            "sorted_rankings": class_prob_pairs
        }

