# Email Spam Classifier

A beginner-friendly Machine Learning project that classifies messages as Spam or Ham.

## Requirements

Install Python 3.9+.

## Installation

Open the VS Code terminal inside this folder and run:

```bash
pip install -r requirements.txt
```

## Train the model

Run:

```bash
python train_model.py
```

This downloads the UCI SMS Spam Collection dataset and creates:

- spam_model.pkl
- tfidf_vectorizer.pkl

## Run the application

After training, run:

```bash
streamlit run app.py
```

The Streamlit web app will open in your browser.

## Project flow

Email/Message
    ↓
TF-IDF Vectorization
    ↓
Naive Bayes Model
    ↓
Spam or Ham
