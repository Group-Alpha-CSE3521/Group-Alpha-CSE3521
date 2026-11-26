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


def load_tfidf(path, vectorizer_path=None):
    model = joblib.load(path)
    if vectorizer_path is not None:
        return {
            "model": model,
            "vectorizer": joblib.load(vectorizer_path),
            "feature_method": "sklearn",
        }
    bundle = model
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


def load_lstm(model_path, vocabulary_path, config_path=None):
    import torch

    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    if vocabulary["<PAD>"] != 0 or vocabulary["<UNK>"] != 1:
        raise ValueError("LSTM vocabulary must use PAD=0 and UNK=1.")
    if config_path is None:
        return torch.jit.load(str(model_path), map_location="cpu").eval(), vocabulary, 150

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if len(vocabulary) != config["vocab_size"]:
        raise ValueError("LSTM vocabulary size does not match the model config.")

    from torch import nn

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_dim = config["hidden_dim"]
            layer_dim = config["layer_dim"]
            dropout = config["dropout"]
            self.embedding = nn.Embedding(config["vocab_size"], config["embed_dim"], padding_idx=0)
            self.emb_dropout = nn.Dropout(dropout)
            self.lstm = nn.LSTM(
                input_size=config["embed_dim"],
                hidden_size=hidden_dim,
                num_layers=layer_dim,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if layer_dim > 1 else 0.0,
            )
            self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 4, 1))

        def forward(self, tokens):
            embeddings = self.emb_dropout(self.embedding(tokens))
            output, _ = self.lstm(embeddings)
            pooled = torch.cat([output.max(dim=1).values, output.mean(dim=1)], dim=1)
            return self.fc(pooled)

    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return model.eval(), vocabulary, config["seq_length"]


def predict_lstm(bundle, text):
    import torch

    model, vocabulary, seq_length = bundle
    tokens = text.split(maxsplit=seq_length)[:seq_length]
    sequence = [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens]
    sequence.extend([vocabulary["<PAD>"]] * (seq_length - len(sequence)))

    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.tensor([sequence], dtype=torch.long))).item()
    positive = probability >= 0.5
    return LABELS[int(positive)], probability if positive else 1 - probability
