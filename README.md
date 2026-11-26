
# Group Alpha
## Amazon Food Review Sentiment Analysis

# Project Overview
An AI-powered text intelligence platform for classifying sentiment in Amazon food reviews.

## Dataset
Amazon Fine Food Reviews
Shared sample: 50,000 reviews (random seed 42). After removing missing values, duplicate review texts, and empty cleaned reviews, 46,318 remain: 37,054 for training and 9,264 for testing. Reviews with 1-3 stars are labeled negative, 4-5 stars are labeled positive. 

## Feature Engineering
- TF-IDF (5,000 features for the classical models)
- Word2Vec (100-dimensional embeddings for the BiLSTM)

## Machine Learning Models
- Multinomial Naive Bayes
- Logistic Regression
- Bidirectional LSTM (BiLSTM)

## Deployment
Streamlit application with three tabs:
- Live Predication: Enter a review and compare predictions.
- Model Performance: View reported evaluation metrics and confusion matrices.
- Dataset Insights: View the data split and sentiment distribution.
Deployment: https://groupalpha.streamlit.app

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
- M3: November 25, 2026
- M4a: November 29, 2026
- Demo: Dec 3, 2026

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

## M3 Progress 
- Trained and evaluated Naive Bayes, Logistic Regression, and BiLSTM on the shared dataset split.
- Compared model accuracy, precision, recall, F1-score, and confusion matrices; recorded BiLSTM training and validation losses.
- Built a Streamlit interface showing live predictions from all three models, model performance, and dataset insights.

## Data
Dataset source: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Raw and processed CSV files should not be committed.
