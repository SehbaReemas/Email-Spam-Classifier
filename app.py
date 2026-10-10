import streamlit as st
import pickle

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Classifier")
st.write(
    "Enter an email or message below and the Machine Learning model "
    "will classify it as Spam or Ham."
)

message = st.text_area(
    "Enter your email/message:",
    height=200,
    placeholder="Example: Congratulations! You have won a prize..."
)

if st.button("🔍 Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        message_tfidf = vectorizer.transform([message])

        prediction = model.predict(message_tfidf)[0]
        probability = model.predict_proba(message_tfidf)[0]

        if prediction == 1:
            confidence = probability[1] * 100
            st.error("🚨 SPAM")
            st.write(f"Confidence: **{confidence:.2f}%**")

        else:
            confidence = probability[0] * 100
            st.success("✅ HAM / NOT SPAM")
            st.write(f"Confidence: **{confidence:.2f}%**")
