import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from tensorflow.keras.utils import pad_sequences
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

MAIN_COLOR = "#28e1ff"
SENTIMENT_COLOR = "#ffffff"
NEGATIVE_COLOR = "#ff6f61"
NEUTRAL_COLOR = "#ffd700"
POSITIVE_COLOR = "#61ff61"

st.set_page_config(page_title="WealthGPT", layout="wide")

st.markdown(
    f"""
    <style>
        body {{
            background-color: #0c1b29;
            color: #fff;
            font-family: 'Arial', sans-serif;
        }}
        .title {{
            font-size: 50px;
            color: {MAIN_COLOR};
            text-align: center;
            font-weight: bold;
            margin-top: 30px;
        }}
        .subtitle {{
            font-size: 24px;
            color: {MAIN_COLOR};
            text-align: center;
            margin-top: 10px;
            margin-bottom: 40px;
        }}
        .button {{
            background-color: {MAIN_COLOR};
            color: white;
            font-size: 20px;
            padding: 12px 24px;
            border-radius: 5px;
            border: none;
            transition: background-color 0.3s ease;
        }}
        .button:hover {{
            background-color: {MAIN_COLOR};
            color: white;
        }}
        .stTextInput>div>div>input {{
            background-color: #1a2733;
            color: white;
            font-size: 18px;
            border: none;
            padding: 12px;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }}
        .stTextInput>div>div>input:hover {{
            background-color: {MAIN_COLOR};
            color: white;
        }}
        .stTextInput>div>div>label {{
            color: {MAIN_COLOR};
            font-size: 18px;
        }}
        .stMarkdown {{
            color: #b1b1b1;
            font-size: 20px;
        }}
        .divider {{
            border: 1px solid {MAIN_COLOR};
            margin: 40px 0;
        }}
        .sentiment-text {{
            font-size: 28px;
            color: {SENTIMENT_COLOR};
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }}
        .confidence {{
            font-size: 20px;
            color: {MAIN_COLOR};
            text-align: center;
            margin-top: 5px;
        }}
        .table {{
            margin-top: 20px;
            width: 100%;
            color: white;
            font-size: 16px;
        }}
        .table th, .table td {{
            text-align: center;
            padding: 12px;
        }}
        .table th {{
            background-color: {MAIN_COLOR};
            font-size: 18px;
        }}
        .table td {{
            background-color: #1a2733;
        }}
        .table tr:hover {{
            background-color: #333;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
MAX_LENGTH = 64

def load_trained_model():
    model = torch.load("sentiment_model.pth", map_location=torch.device('cpu'))
    model.eval()
    return model

def preprocess_input(input_sentence, tokenizer, max_len=MAX_LENGTH):
    encoded = tokenizer.encode(input_sentence, add_special_tokens=True, max_length=max_len, truncation=True)
    input_ids = pad_sequences([encoded], maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]
    return torch.tensor(input_ids), torch.tensor(attention_masks)

def predict_sentiment(input_sentence, model, tokenizer):
    input_ids, attention_mask = preprocess_input(input_sentence, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = input_ids.to(device)
        mask = attention_mask.to(device)
        outputs = model(inputs, attention_mask=mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1).max().item() * 100

    return LABEL_MAP[prediction], confidence

st.markdown("<div class='title'>WEALTHGPT - AI NEWS CLASSIFIER</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Classify news headlines as Negative, Neutral, or Positive.</div>", unsafe_allow_html=True)

if os.path.exists("sentiment_model.pth"):
    model = load_trained_model()

    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = []
        st.session_state.sentiment_count = {'Negative': 0, 'Neutral': 0, 'Positive': 0}

    input_sentence = st.text_area("Enter your sentence:", key="input_sentence", height=100)

    if st.button("Analyze", key="analyze_button"):
        with st.spinner('Analyzing sentence...'):
            if input_sentence:
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                sentiment, confidence = predict_sentiment(input_sentence, model, tokenizer)
                st.markdown(f"<div class='sentiment-text'>Sentiment: {sentiment} <span class='confidence'>({confidence:.2f}% Confidence)</span></div>", unsafe_allow_html=True)

                st.session_state.sentiment_count[sentiment] += 1

                st.session_state.analysis_data.insert(0, (input_sentence, sentiment, f"{confidence:.2f}%"))

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.write("### Sentiment Analysis Results:")
                data = pd.DataFrame(st.session_state.analysis_data, columns=["Sentence", "Sentiment", "Confidence"])
                with st.expander("View Results", expanded=True):
                    st.table(data)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                col1 = st.columns([1, 3, 1])[1]
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sentiments = ['Negative', 'Neutral', 'Positive']
                    counts = [st.session_state.sentiment_count['Negative'], 
                              st.session_state.sentiment_count['Neutral'], 
                              st.session_state.sentiment_count['Positive']]
                    bar_colors = [NEGATIVE_COLOR, NEUTRAL_COLOR, POSITIVE_COLOR]
                    bars = ax.bar(sentiments, counts, color=bar_colors, edgecolor="none", width=0.6)

                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', 
                                ha='center', va='bottom', fontsize=12, color=SENTIMENT_COLOR)

                    ax.set_title('Sentiment Distribution', fontsize=18, color=SENTIMENT_COLOR, pad=15)
                    ax.set_ylabel('Frequency', fontsize=14, color=SENTIMENT_COLOR)
                    ax.set_xlabel('Sentiments', fontsize=14, color=SENTIMENT_COLOR)
                    ax.tick_params(colors=SENTIMENT_COLOR, labelsize=12)
                    ax.set_facecolor('#1a2733')
                    fig.patch.set_facecolor('#0c1b29')

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    ax.set_yticks(range(0, max(counts) + 1))
                    ax.grid(color='#333', linestyle='--', linewidth=0.5, axis='y')
                    st.pyplot(fig)
            else:
                st.write("Please enter a sentence to analyze.")
else:
    st.write("Please upload the trained model first.")
