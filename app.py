import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import LABELS, clean_text, load_lstm, load_tfidf, predict_lstm, predict_tfidf


ROOT = Path(__file__).resolve().parent
MODELS = {
    "Naive Bayes": ("naive_bayes.joblib",),
    "Logistic Regression": ("logistic_regression.joblib",),
    "BiLSTM": ("bilstm.pt", "bilstm_vocab.json"),
}
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_model(paths, fingerprints):
    # Fingerprints invalidate the cache when an export is replaced.
    if len(paths) == 2:
        return load_lstm(*paths)
    return load_tfidf(paths[0])


def analyze_review(review):
    predictions = {}
    errors = {}
    for name, filenames in MODELS.items():
        paths = tuple(ROOT / "models" / filename for filename in filenames)
        try:
            missing = [path.name for path in paths if not path.is_file()]
            if missing:
                errors[name] = f"Missing trained files in models/: {', '.join(missing)}."
                continue

            with st.spinner(f"Loading and running {name}…", show_time=True):
                stats = [path.stat() for path in paths]
                fingerprints = tuple((stat.st_mtime_ns, stat.st_size) for stat in stats)
                bundle = load_model(paths, fingerprints)
                predict = predict_lstm if len(paths) == 2 else predict_tfidf
                predictions[name] = predict(bundle, review)
        except ImportError as exc:
            LOGGER.exception("Cannot import a dependency for %s", name)
            errors[name] = (
                f"Model load or prediction could not import a module ({exc}). "
                "Check the export and its runtime versions. See the server log for details."
            )
        except OSError as exc:
            LOGGER.exception("Cannot read %s artifacts", name)
            errors[name] = f"Could not read the trained files ({type(exc).__name__}). See the server log."
        except Exception as exc:
            LOGGER.exception("Inference failed for %s", name)
            errors[name] = (
                f"Model load or prediction failed ({type(exc).__name__}). "
                "Check that the model and its fitted preprocessing were exported together "
                "and that the dependency versions match. See the server log for details."
            )
    return predictions, errors


def show_predictions(predictions, errors):
    if len(predictions) < 2:
        st.warning(
            f"Produced {len(predictions)} of the 2 predictions needed for a comparison. "
            "See the model cards below for details."
        )
    for column, name in zip(st.columns(len(MODELS)), MODELS):
        with column, st.container(border=True):
            st.subheader(name)
            if name in errors:
                st.error(errors[name])
                continue

            label, confidence = predictions[name]
            st.metric("Predicted sentiment", label)
            if confidence is None:
                st.caption("This model does not provide a class probability.")
            else:
                st.caption(f"Model probability for {label.lower()}: {confidence:.1%}")


def show_performance():
    st.subheader("Reported evaluation")
    st.caption("Precision, recall and F1 are macro averages across both sentiment classes. "
               "A dash marks a value absent from the available reports.")
    try:
        metrics = json.loads((ROOT / "report/model_metrics.json").read_text(encoding="utf-8"))
        rows = []
        for name in MODELS:
            report = metrics.get(name, {})
            row = {"Model": name}
            for key, heading in (
                ("accuracy", "Accuracy"), ("precision", "Precision"),
                ("recall", "Recall"), ("f1", "F1-score"),
                ("training_seconds", "Training time"),
            ):
                value = report.get(key)
                row[heading] = "—" if value is None else (
                    f"{value:,.2f} s" if key == "training_seconds" else f"{value:.2%}"
                )
            row["Source"] = report.get("source", "—")
            rows.append(row)
        st.dataframe(rows, hide_index=True, width="stretch")

        for name, report in metrics.items():
            if "confusion_matrix" in report:
                st.subheader(f"{name} confusion matrix")
                st.caption("Rows are actual labels. Columns are predicted labels.")
                matrix = pd.DataFrame(
                    report["confusion_matrix"],
                    index=[f"Actual {label.lower()}" for label in LABELS.values()],
                    columns=[f"Predicted {label.lower()}" for label in LABELS.values()],
                )
                st.dataframe(matrix, width="stretch")
    except (OSError, ValueError, TypeError):
        LOGGER.exception("Cannot display evaluation report")
        st.error("The evaluation report could not be read. Check report/model_metrics.json.")


st.set_page_config(page_title="Amazon Food Review Sentiment", layout="wide")
st.title("Amazon Food Review Sentiment")
st.caption("Ohio State CSE 3521 · Group Alpha")

live, performance, insights = st.tabs(["Live Prediction", "Model Performance", "Dataset Insights"])

with live:
    st.write("Compare how the team's models classify an Amazon Fine Food review.")
    with st.form("review_form"):
        review = st.text_area("Review text", placeholder="This coffee tastes great.", height=150)
        submitted = st.form_submit_button("Analyze Sentiment", type="primary")

    if submitted:
        cleaned = clean_text(review)
        st.session_state["review_analysis"] = None
        st.session_state["empty_review"] = not cleaned
        if cleaned:
            predictions, errors = analyze_review(cleaned)
            st.session_state["review_analysis"] = (
                review, len(cleaned.split(maxsplit=150)) > 150, predictions, errors
            )

    if st.session_state.get("empty_review"):
        st.warning("Please enter a review with at least one word or number.")

    analysis = st.session_state.get("review_analysis")
    if analysis is not None:
        analyzed_review, truncated, predictions, errors = analysis
        st.text(f"Submitted review: {analyzed_review}")
        if truncated:
            st.caption("BiLSTM uses the first 150 tokens, matching training.")
        show_predictions(predictions, errors)

with performance:
    show_performance()

with insights:
    st.subheader("Amazon Fine Food Reviews")
    for column, (label, count) in zip(
        st.columns(4),
        [("Shared sample", 50000), ("After cleaning", 46318),
         ("Training split", 37054), ("Test split", 9264)],
    ):
        column.metric(label, f"{count:,}")
    st.write("Reviews with 1–3 stars are labeled negative. Reviews with 4–5 stars are labeled positive.")
    st.bar_chart({"Sentiment": ["Negative", "Positive"], "Test reviews": [2019, 7245]},
                 x="Sentiment", y="Test reviews")
    st.caption("Sources: project README and saved split counts and test output in the BiLSTM notebook.")
