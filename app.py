import streamlit as st
import os
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Hugging Face model repo info
HF_REPO = "arpitarout01/amazon-review-sentiment-model"  # Replace with your HF repo
BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt"
]

def download_file(filename):
    if not os.path.exists(filename):
        st.info(f"Downloading {filename} from Hugging Face...")
        url = f"{BASE_URL}/{filename}"
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)

# Download all files needed for the model
for file in FILES:
    download_file(file)

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(".")
    model = AutoModelForSequenceClassification.from_pretrained(".", local_files_only=True)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

st.title("Amazon Product Review Sentiment Analysis")

user_input = st.text_area("Enter product review text here:")

def sentiment_emoji(pos_prob):
    if pos_prob > 0.85:
        return "😄🎉🔥"
    elif pos_prob > 0.6:
        return "🙂👍"
    elif pos_prob > 0.4:
        return "😐"
    elif pos_prob > 0.15:
        return "😕"
    else:
        return "😠👎"

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        neg_prob, pos_prob = probs[0], probs[1]

        # Emoji meter
        emoji = sentiment_emoji(pos_prob)
        st.markdown(f"### Sentiment Meter: {emoji}")

        # Bar chart
        df = pd.DataFrame({
            'Sentiment': ['Negative', 'Positive'],
            'Probability': [neg_prob, pos_prob]
        })
        st.bar_chart(df.set_index('Sentiment'))

        # Progress bar
        progress = st.progress(0)
        progress.progress(int(pos_prob * 100))
        st.write(f"Positive sentiment score: {pos_prob*100:.2f}%")

        # Text feedback
        if pos_prob > 0.75:
            st.success("Looks like a glowing review! 🌟")
        elif pos_prob > 0.5:
            st.info("Pretty positive vibes here! 🙂")
        elif pos_prob > 0.3:
            st.warning("Some mixed feelings detected 🤔")
        else:
            st.error("Uh oh, looks like this review isn’t happy 😞")

