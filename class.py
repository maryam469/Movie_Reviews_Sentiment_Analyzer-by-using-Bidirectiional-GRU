# app.py

import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = 100000
maxlen = 100

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    words = text.split()
    sequence = tokenizer.texts_to_sequences([" ".join(words)])
    padded = pad_sequences(sequence, maxlen=maxlen)
    return padded

# UI
st.title("Movie Review Sentiment Analyzer")
st.write("Enter a review to check if it's **Positive** or Negative.")

user_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed = preprocess_text(user_input)
        prediction = model.predict(processed)[0][0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        st.subheader(f"Prediction: {prediction:.2f} → {sentiment}")




