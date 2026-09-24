import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
PREDICT_URL = f"{BACKEND_URL}/predict"

st.title("Next Word Predictor")
st.write("Choose a model and enter a sentence to predict the next top 3 words.")

MODEL_OPTIONS = ["new_fine", "new_lstm"]
selected_model = st.selectbox("Select model", MODEL_OPTIONS, index=0)

text = st.text_input("Enter some text")

if st.button("Predict"):
    if text:
        try:
            response = requests.post(
                PREDICT_URL,
                json={"text": text, "model": selected_model},
                timeout=15,
            )

            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, list) and payload:
                    st.success(f"Top 3 Predicted Words using {selected_model}:")
                    for i, word in enumerate(payload, 1):
                        st.write(f"{i}. {word}")
                elif isinstance(payload, dict):
                    msg = payload.get("next_word") or payload.get("error") or "No word predicted"
                    st.success(f"Prediction: {msg}")
                else:
                    st.warning("No valid words predicted.")
            else:
                err = response.json() if response.headers.get("Content-Type", "").startswith("application/json") else {}
                msg = err.get("error", "Failed to get prediction from server.")
                st.error(f"Error {response.status_code}: {msg}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
    else:
        st.warning("Please enter some text to predict.")
