
# Group Alpha
## Amazon Food Review Sentiment Analysis

# Project Overview
An AI-powered text intelligence platform for classifying sentiment in Amazon food reviews.

## Dataset
Amazon Fine Food Reviews
Shared sample: 50,000 reviews

## Feature Engineering
- TF-IDF
- Word2Vec (100-dimensional embeddings)

## Machine Learning Models
- Multinomial Naive Bayes
- Logistic Regression
- LSTM

## Deployment
Streamlit Community Cloud

## Team Members
| Name | Primary Role |
|------|--------------|
| Yousef Alshanqiti | Project Lead |
| Ezana Worku | Data Engineer |
| Mark Oyster | ML Engineer |
| Adam Rehan | App Engineer |
| Alec Lusk | Analyst / Writer |

## Milestones
- M1: November 10, 2026
- M2: November 18, 2026
- M3: November 22, 2026
- M4: November 29, 2026
- Demo: TBD

## Run the app

Use Python 3.12 from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## M2 Progress
- Cleaned dataset: 46,318 reviews.
- Implemented TF-IDF (5,000 features) and Word2Vec.
- Naive Bayes accuracy: 82.66% (majority baseline: 78.01%).
- Completed confusion matrix and error analysis.
- Built initial Streamlit UI (predictions not integrated yet).

## Next Steps - M3
- Train Logistic Regression and a deep learning model.
- Compare model performance and tune parameters.
- Integrate two trained models into Streamlit.
- Complete deployment testing and documentation.

## Data
Dataset source: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Raw and processed CSV files should not be committed.
