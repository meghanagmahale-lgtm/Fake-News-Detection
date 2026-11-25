import streamlit as st
import pickle

# Load saved model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check the model's prediction.")

# Input box
news_text = st.text_area("Enter news text here:")

# Button
if st.button("Check News"):
    if news_text.strip():
        news_vector = vectorizer.transform([news_text])
        prediction = model.predict(news_vector)[0]

        if prediction == 0 or prediction == "FAKE":
            st.warning("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")
    else:
        st.info("Please enter some news text.")
