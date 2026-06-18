# 🛡️ Supervised Machine Learning: SMS & Email Spam Detector

A high-performance, production-ready natural language processing (NLP) and supervised machine learning classification pipeline designed to isolate, track, and tag malicious communication vectors (spam) from normal situational correspondence (ham) with a verified **97.04% classification accuracy**.

🔗 **Live Interactive Application Layout:** [Launch Web Application Workspace](https://ml-spam-detector-mdfa7t8phvs3rxewbusman.streamlit.app/)

---

## 🚀 System Architecture & Core Pipeline

The classification engine ingests unstructured raw message text strings and passes them through a modular data science track to map semantic word tokens into linear classification boundaries.

1. **Data Preprocessing Layer:** Cleans incoming text streams, encodes targets into binary integer channels (`ham` $\rightarrow$ 0, `spam` $\rightarrow$ 1), and splits rows using stratified ratios to maintain data balance across the training (80%) and testing (20%) datasets.
2. **Feature Extraction Engine (NLP):** Utilizes an optimized `TfidfVectorizer` to strip English stop-words, convert characters to lowercase, and compute token weights via Term Frequency-Inverse Document Frequency calculations.
3. **Supervised Classification Core:** Trains a `MultinomialNB` (Naive Bayes) classifier model, which evaluates combined token probability vectors to execute real-time decision outputs.

---

## 📊 Performance Matrix & Verification Logs

The pipeline achieved an outstanding overall accuracy score with **zero false positives**, ensuring critical correspondence is never mistakenly routed to junk categories:

```text
==================================================
🏆 MODEL TRAINING ACCURACY COMPLETED: 97.04%
==================================================

📝 DETAILED CLASSIFICATION REPORT:
                  precision    recall  f1-score   support

    Ham (Normal)       0.97      1.00      0.98       966
Spam (Malicious)       1.00      0.78      0.88       149

        accuracy                           0.97      1115
       macro avg       0.98      0.89      0.93      1115
    weighted avg       0.97      0.97      0.97      1115
