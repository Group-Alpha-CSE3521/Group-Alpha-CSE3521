"""Inference adapter"""

import json
import re

import joblib
import numpy as np
from scipy.sparse import diags


LABELS = {0: "Negative", 1: "Positive"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_tfidf(path):
    bundle = joblib.load(path)
    if bundle["feature_method"] not in {"sklearn", "lecture"}:
        raise ValueError("Unsupported TF-IDF feature method.")
    if bundle["feature_method"] == "lecture" and len(bundle["idf"]) != len(
        bundle["vectorizer"].get_feature_names_out()
    ):
        raise ValueError("IDF weights do not match the fitted vocabulary.")
    return bundle


def predict_tfidf(bundle, text):
    features = bundle["vectorizer"].transform([text])
    if bundle["feature_method"] == "lecture":
        lengths = np.asarray(features.sum(axis=1)).ravel()
        frequencies = diags(1 / np.maximum(lengths, 1)) @ features
        features = frequencies.multiply(bundle["idf"]).tocsr()

    model = bundle["model"]
    label = model.predict(features)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        class_index = list(model.classes_).index(label)
        confidence = float(model.predict_proba(features)[0][class_index])
    return LABELS[label], confidence


def load_lstm(model_path, vocabulary_path):
    import torch

    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    if vocabulary["<PAD>"] != 0 or vocabulary["<UNK>"] != 1:
        raise ValueError("LSTM vocabulary must use PAD=0 and UNK=1.")
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    return model, vocabulary


def predict_lstm(bundle, text):
    import torch

    model, vocabulary = bundle
    tokens = text.split(maxsplit=150)[:150]
    sequence = [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens]
    sequence.extend([vocabulary["<PAD>"]] * (150 - len(sequence)))

    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.tensor([sequence], dtype=torch.long))).item()
    positive = probability >= 0.5
    return LABELS[int(positive)], probability if positive else 1 - probability
