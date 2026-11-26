# CSE 3521 - Group Alpha
# Milestone 2/3 data preparation: cleaning, splitting, TF-IDF, and Word2Vec

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import diags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 5000


def clean_text(text):
    """Apply the Milestone 2 text-cleaning rules to one review."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_reviews(df):
    """Clean the sample and create the sentiment label without changing the sample."""
    cleaned = df.dropna(subset=["Text", "Score"]).copy()
    cleaned = cleaned.drop_duplicates(subset=["Text"]).copy()
    cleaned["Clean_Text"] = cleaned["Text"].apply(clean_text)
    cleaned = cleaned[cleaned["Clean_Text"] != ""].copy()
    cleaned["Sentiment"] = (cleaned["Score"] >= 4).astype(int)
    return cleaned


def load_and_clean(input_path):
    """Load the original 50,000-review sample and reproduce the Milestone 1/2 data."""
    df = pd.read_csv(input_path)
    required = {"Text", "Score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    original_count = len(df)
    original_stats = {
        "original_reviews": original_count,
        "positive_reviews": int((df["Score"] >= 4).sum()),
        "negative_reviews": int((df["Score"] <= 3).sum()),
        "average_review_length_words": float(
            df["Text"].dropna().apply(lambda text: len(str(text).split())).mean()
        ),
        "duplicate_review_texts": int(df["Text"].duplicated().sum()),
        "missing_review_summaries": int(df["Summary"].isna().sum()) if "Summary" in df else None,
        "five_star_reviews": int((df["Score"] == 5).sum()),
        "rating_counts": df["Score"].value_counts().sort_index().to_dict(),
    }

    cleaned = clean_reviews(df)
    train_df, test_df = train_test_split(
        cleaned,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=cleaned["Sentiment"],
    )
    return original_stats, cleaned, train_df, test_df


def fit_lecture_tfidf(train_texts):
    """Fit a 5,000-word CountVectorizer and Lesson 6 TF-IDF weights."""
    vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    train_bow = vectorizer.fit_transform(train_texts)

    # TF uses the sum of counts retained in the 5,000-word vocabulary.
    # It approximates full document length because out-of-vocabulary words are excluded.
    train_lengths = np.asarray(train_bow.sum(axis=1)).ravel()
    train_lengths = np.maximum(train_lengths, 1)
    train_tf = diags(1 / train_lengths) @ train_bow

    document_count = train_bow.shape[0]
    document_frequency = np.asarray((train_bow > 0).sum(axis=0)).ravel()
    idf = np.log(document_count / document_frequency)
    train_tfidf = train_tf.multiply(idf).tocsr()
    return vectorizer, idf, train_tfidf


def transform_tfidf(texts, pipeline):
    """Transform new or existing text using an already-fitted pipeline."""
    vectorizer = pipeline["vectorizer"]
    idf = pipeline["idf"]
    bow = vectorizer.transform(texts)
    lengths = np.asarray(bow.sum(axis=1)).ravel()
    lengths = np.maximum(lengths, 1)
    tf = diags(1 / lengths) @ bow
    return tf.multiply(idf).tocsr()


def clean_texts(texts):
    """Clean text before a reusable TF-IDF or app-side transformation."""
    return [clean_text(text) for text in texts]


def train_word2vec(clean_texts):
    """Train Word2Vec using only the cleaned training reviews."""
    tokenized_reviews = [text.split() for text in clean_texts]
    model = Word2Vec(
        sentences=tokenized_reviews,
        vector_size=100,
        window=5,
        min_count=2,
        workers=2,
        seed=RANDOM_STATE,
        epochs=5,
    )
    return model


def save_rating_distribution(rating_counts, output_path):
    ratings = sorted(rating_counts)
    counts = [rating_counts[rating] for rating in ratings]
    plt.figure(figsize=(8, 5))
    plt.bar([str(rating) for rating in ratings], counts, color="steelblue")
    plt.title("Group Alpha: Rating Distribution")
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_tfidf_features(feature_names, difference, output_path):
    top_positive = np.argsort(difference)[-10:][::-1]
    top_negative = np.argsort(difference)[:10]
    indices = np.concatenate([top_negative[::-1], top_positive])

    words = feature_names[indices]
    scores = difference[indices]
    colors = ["tomato" if score < 0 else "seagreen" for score in scores]

    plt.figure(figsize=(12, 7))
    plt.barh(words, scores, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Group Alpha: TF-IDF Sentiment Features")
    plt.xlabel("Difference in Average TF-IDF (Positive - Negative)")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def verify_existing_test(existing_path, new_test_df):
    """Compare review IDs with an existing test file when it contains an Id column."""
    if not existing_path.exists():
        return None
    try:
        existing = pd.read_csv(existing_path)
    except Exception:
        return None
    if "Id" not in existing or "Id" not in new_test_df:
        return None
    return set(existing["Id"].dropna()) == set(new_test_df["Id"].dropna())


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Group Alpha CSE 3521 data engineering pipeline"
    )
    parser.add_argument(
        "--input",
        default=str(project_dir / "group_sample_seed42.csv"),
        help="Input 50,000-review CSV",
    )
    parser.add_argument(
        "--output",
        default=str(project_dir / "alpha_outputs"),
        help="Directory for generated output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    original_stats, cleaned, train_df, test_df = load_and_clean(input_path)

    train_path = output_dir / "alpha_train.csv"
    test_path = output_dir / "alpha_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    vectorizer, idf, train_tfidf = fit_lecture_tfidf(
        train_df["Clean_Text"]
    )
    test_tfidf = transform_tfidf(
        test_df["Clean_Text"],
        {"vectorizer": vectorizer, "idf": idf},
    )

    pipeline_path = output_dir / "alpha_tfidf_pipeline.joblib"
    joblib.dump(
        {"vectorizer": vectorizer, "idf": idf},
        pipeline_path,
    )

    word2vec = train_word2vec(train_df["Clean_Text"])
    word2vec_path = output_dir / "alpha_word2vec.model"
    word2vec.save(str(word2vec_path))

    feature_names = vectorizer.get_feature_names_out()
    positive_mask = train_df["Sentiment"].to_numpy() == 1
    negative_mask = train_df["Sentiment"].to_numpy() == 0
    positive_avg = np.asarray(
        train_tfidf[positive_mask].mean(axis=0)
    ).ravel()
    negative_avg = np.asarray(
        train_tfidf[negative_mask].mean(axis=0)
    ).ravel()
    difference = positive_avg - negative_avg

    tfidf_plot_path = output_dir / "alpha_tfidf_features.png"
    save_tfidf_features(feature_names, difference, tfidf_plot_path)
    rating_plot_path = output_dir / "alpha_rating_distribution.png"
    save_rating_distribution(original_stats["rating_counts"], rating_plot_path)

    top_positive = np.argsort(difference)[-10:][::-1]
    top_negative = np.argsort(difference)[:10]

    stats = {
        "milestone_1": original_stats,
        "cleaned_reviews": len(cleaned),
        "training_reviews": len(train_df),
        "testing_reviews": len(test_df),
        "training_sentiment_counts": train_df["Sentiment"].value_counts().sort_index().to_dict(),
        "testing_sentiment_counts": test_df["Sentiment"].value_counts().sort_index().to_dict(),
        "tfidf_vocabulary_size": len(feature_names),
        "training_tfidf_shape": list(train_tfidf.shape),
        "testing_tfidf_shape": list(test_tfidf.shape),
        "word2vec_embedding_dimensions": word2vec.vector_size,
        "word2vec_vocabulary_size": len(word2vec.wv),
        "top_positive_features": [
            {"word": feature_names[index], "difference": float(difference[index])}
            for index in top_positive
        ],
        "top_negative_features": [
            {"word": feature_names[index], "difference": float(difference[index])}
            for index in top_negative
        ],
    }
    with open(output_dir / "alpha_dataset_stats.json", "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    existing_test_path = input_path.parent / "alpha_test.csv"
    existing_test_match = verify_existing_test(existing_test_path, test_df)

    print("\n===== GROUP ALPHA DATA ENGINEERING SUMMARY =====")
    print(f"Original reviews: {original_stats['original_reviews']}")
    print(f"Cleaned reviews: {len(cleaned)}")
    print(f"Training reviews: {len(train_df)}")
    print(f"Testing reviews: {len(test_df)}")
    print("Training sentiment counts:", train_df["Sentiment"].value_counts().sort_index().to_dict())
    print("Testing sentiment counts:", test_df["Sentiment"].value_counts().sort_index().to_dict())
    print(f"TF-IDF vocabulary size: {len(feature_names)}")
    print(f"Training matrix: {train_tfidf.shape}")
    print(f"Testing matrix: {test_tfidf.shape}")
    print(f"Word2Vec dimensions: {word2vec.vector_size}")
    print(f"Word2Vec vocabulary size: {len(word2vec.wv)}")
    if existing_test_match is not None:
        status = "MATCH" if existing_test_match else "DIFFERENT"
        print(f"Existing alpha_test.csv review IDs: {status}")
    print("\nTop positive TF-IDF features:")
    for item in stats["top_positive_features"]:
        print(f"  {item['word']}: {item['difference']:.4f}")
    print("\nTop negative TF-IDF features:")
    for item in stats["top_negative_features"]:
        print(f"  {item['word']}: {item['difference']:.4f}")
    print("\nSaved output files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()
