import streamlit as st

st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="wide"
)

st.title("Amazon Food Review Sentiment Analyzer")

st.write(
    "Enter an Amazon food review to predict "
    "whether its sentiment is positive or negative."
)

tab1, tab2 = st.tabs([
    "Sentiment Predictor",
    "Model Comparison"
])

with tab1:
    review = st.text_area(
        "Enter your review:",
        placeholder="This product tastes amazing!",
        height=150
    )

    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            st.info(
                "Review received. Model predictions "
                "will appear here once integrated."
            )

with tab2:
    st.subheader("Model Performance")
    st.write(
        "Accuracy, precision, recall, F1-score "
        "and confusion matrices will appear here."
    )
