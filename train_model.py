import pandas as pd
import urllib.request
import zipfile
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_FILE = "dataset.zip"
DATA_FILE = "SMSSpamCollection"

print("Downloading dataset...")

urllib.request.urlretrieve(URL, ZIP_FILE)

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(".")

data = pd.read_csv(
    DATA_FILE,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

data["label"] = data["label"].map({"ham": 0, "spam": 1})

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and vectorizer saved successfully!")
