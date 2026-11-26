import json
import logging
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference import LABELS, clean_text, load_lstm, load_tfidf, predict_lstm, predict_tfidf


ROOT = Path(__file__).resolve().parent
MODELS = {
    "Naive Bayes": ("naive_bayes.joblib", "tfidf_vectorizer.joblib"),
    "Logistic Regression": ("logistic_regression.joblib", "tfidf_vectorizer.joblib"),
    "BiLSTM": ("lstm_model.pt", "word2idx.json", "lstm_config.json"),
}
LOGGER = logging.getLogger(__name__)
SENTIMENT_COLORS = {"Negative": "#f87171", "Positive": "#4ade80"}
TEST_COUNTS = {"Negative": 2019, "Positive": 7245}
TEST_TOTAL = sum(TEST_COUNTS.values())
MAJORITY_LABEL = max(TEST_COUNTS, key=TEST_COUNTS.get)
MAJORITY_ACCURACY = TEST_COUNTS[MAJORITY_LABEL] / TEST_TOTAL


@st.cache_resource(show_spinner=False)
def load_model(name, paths, fingerprints):
    # Fingerprints invalidate the cache when an export is replaced.
    if name == "BiLSTM":
        return load_lstm(*paths)
    return load_tfidf(*paths)


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
                bundle = load_model(name, paths, fingerprints)
                predict = predict_lstm if name == "BiLSTM" else predict_tfidf
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
        with column, st.container(border=True, height=190):
            st.subheader(name)
            if name in errors:
                st.error(errors[name])
                continue

            label, confidence = predictions[name]
            color = {"Positive": "green", "Negative": "red"}[label]
            st.markdown(f":{color}[**{label}**]")
            if confidence is None:
                st.caption("This model does not provide a class probability.")
            else:
                st.progress(confidence, text=f"Predicted-class probability · {confidence:.1%}")


def show_confusion_matrix(matrix):
    labels = list(LABELS.values())
    cells = pd.DataFrame(
        [(actual, predicted, matrix[i][j])
         for i, actual in enumerate(labels) for j, predicted in enumerate(labels)],
        columns=["Actual", "Predicted", "Count"],
    )
    chart = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=labels),
        tooltip=["Actual:N", "Predicted:N", alt.Tooltip("Count:Q", format=",d")],
    )
    rectangles = chart.mark_rect(cornerRadius=4).encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues", zero=True), legend=None),
    )
    counts = chart.mark_text(fontSize=16, fontWeight="bold").encode(
        text=alt.Text("Count:Q", format=",d"),
        color=alt.condition(
            alt.datum.Count > cells["Count"].max() / 2, alt.value("white"), alt.value("#111827")
        ),
    )
    st.altair_chart((rectangles + counts).properties(height=165), use_container_width=True)


def show_performance():
    st.subheader("Reported evaluation")
    st.caption(
        f"Majority-class baseline · **{MAJORITY_ACCURACY:.2%}** by predicting "
        f"{MAJORITY_LABEL.lower()} for all {TEST_TOTAL:,} test reviews. "
        "Precision, recall and F1 are macro averages. A dash marks an unverified value."
    )
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
                    f"{value:.4g} s" if key == "training_seconds" else f"{value:.2%}"
                )
            rows.append(row)
        st.dataframe(rows, hide_index=True, width="stretch")

        matrices = {name: report["confusion_matrix"] for name, report in metrics.items()
                    if "confusion_matrix" in report}
        if matrices:
            st.subheader("Confusion matrix")
            selected = st.selectbox("Model", list(matrices), index=len(matrices) - 1)
            show_confusion_matrix(matrices[selected])

        with st.expander("Evaluation sources"):
            for name, report in metrics.items():
                if report.get("source"):
                    st.markdown(f"**{name}** · {report['source']}")
            st.caption("The BiLSTM time covers the training and validation loop. "
                       "Training times for the two TF-IDF exports could not be verified.")
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
        review = st.text_area("Review text", placeholder="This coffee tastes great.", height=100)
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
        _, truncated, predictions, errors = analysis
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
         ("Training split", 37054), ("Test split", TEST_TOTAL)],
    ):
        column.metric(label, f"{count:,}")
    st.write("Reviews with 1–3 stars are labeled negative. Reviews with 4–5 stars are labeled positive.")
    st.subheader("Test-set sentiment")
    distribution = pd.DataFrame(TEST_COUNTS.items(), columns=["Sentiment", "Count"])
    distribution["Share"] = distribution["Count"] / TEST_TOTAL
    for column, (label, count, share) in zip(st.columns(2), distribution.itertuples(index=False)):
        column.metric(label, f"{count:,} ({share:.2%})")
    chart = alt.Chart(distribution).mark_bar(size=38, cornerRadiusEnd=4).encode(
        x=alt.X("Count:Q", title="Test reviews", scale=alt.Scale(zero=True, domainMin=0)),
        y=alt.Y("Sentiment:N", sort=list(TEST_COUNTS), title=None),
        color=alt.Color("Sentiment:N", scale=alt.Scale(
            domain=list(SENTIMENT_COLORS), range=list(SENTIMENT_COLORS.values())
        ), legend=None),
        tooltip=["Sentiment:N", alt.Tooltip("Count:Q", format=",d"), alt.Tooltip("Share:Q", format=".2%")],
    ).properties(height=120)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Test counts verified on the seed-42 held-out split.")
