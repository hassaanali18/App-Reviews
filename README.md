# Google Play Sentiment Analysis

A Python‑based project to analyze user reviews from Google Play, determine customer sentiment, and build predictive models of satisfaction.

## 📖 Overview

This project ingests and processes user reviews from Google Play to:
- **Understand user sentiment** via text analytics and rating distributions  
- **Identify key factors** driving satisfaction (e.g., review text, thumbs‑up counts)  
- **Build classifiers** (Logistic Regression, Random Forest, Neural Network) to predict whether a review is “Satisfied” or “Not Satisfied” based on both numerical and textual features :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.

## 🗂 Dataset

- **Source:** Google Play user reviews (162 rows, 11 columns)  
- **Key columns:**  
  - `review_id`, `user_name`  
  - `rating` (1–5)  
  - `review_description` (text)  
  - `thumbs_up` (helpfulness count)  
  - `review_date`, `app_version`, `language_code`, `country_code`  
  - `developer_response`, `developer_response_date` :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}.

## 🔧 Data Preparation

1. **Loading & Exploration**  
   - Used Pandas to load CSV and inspect schema.  
2. **Cleaning**  
   - Filled missing `developer_response`/date with “No response”  
   - Imputed missing `app_version` by mode  
   - Dropped `review_title` (all null)  
   - Removed duplicate `review_description` entries :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}.  
3. **Type Conversion & Outliers**  
   - Converted dates to `datetime`; set invalid developer‑response dates to `NaT`  
   - Removed outliers in `thumbs_up` via IQR method.  
4. **Feature Engineering**  
   - Min–Max scaling for `rating` and `thumbs_up`  
   - Label‐encoding `app_version`; one‑hot for `language_code` and `country_code`  
   - TF‑IDF vectorization (top 500 tokens) of `review_description` :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}.

## 📊 Exploratory Analysis

- **Ratings Distribution:** heavily skewed to 5★, with few low‐rating outliers.  
- **Thumbs‑Up:** most reviews have 0, a few highly‐liked outliers.  
- **App Versions:** concentrated on recent versions (e.g., “18.0”, “17.0”), reflecting active user feedback.  
- **Text Insights:** common positive words (`great`, `easy`, `love`), but also mentions of `issue`, `backup` :contentReference[oaicite:8]{index=8}&#8203;:contentReference[oaicite:9]{index=9}.

## 🤖 Modeling

Three classification algorithms were evaluated:

| Model                | Key Settings                | Test Accuracy |
|----------------------|-----------------------------|--------------:|
| Logistic Regression  | max_iter=1000               |        96.67% |
| Random Forest        | n_estimators=100            |        96.67% |
| Neural Network       | 2 hidden dense layers, Dropout, 20 epochs | 96.67% |

- **Features:** concatenated scaled numeric features + TF‑IDF matrix  
- **Evaluation:** accuracy, precision, recall, F1‑score, cross‑validation :contentReference[oaicite:10]{index=10}&#8203;:contentReference[oaicite:11]{index=11}.

## 🚀 Results & Insights

- **High performance:** All models achieved ~96.7% accuracy, showing ratings and review text strongly predict satisfaction.  
- **Feature importance:** Rating is the strongest predictor (corr. 0.86 with satisfaction) versus weak relation of thumbs‑up (corr. 0.07).  
- **Version effects:** Certain app versions saw dips in satisfaction, suggesting targeted areas for improvement :contentReference[oaicite:12]{index=12}&#8203;:contentReference[oaicite:13]{index=13}.

## 🔮 Future Work

- Incorporate additional metadata (user profiles, review timestamps).  
- Experiment with advanced NLP (word embeddings, transformer models).  
- Deploy real‑time analysis pipeline for live sentiment monitoring.  
- Address class imbalance and minority review patterns.
