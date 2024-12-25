import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Set page configuration
st.set_page_config(
    page_title="AI News Classifier",
    page_icon="🎈",
    layout="centered"
)

# Load BERT model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Load the pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,  # Negative, Neutral, Positive
        output_attentions=False,
        output_hidden_states=False
    )
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Load resources
st.write("Loading BERT model...")
tokenizer, model, device = load_model_and_tokenizer()
st.success("Model loaded successfully!")

# App title
st.title("🎈 AI News Classifier")
st.write("Enter a news headline below to classify it as Negative, Neutral, or Positive.")

# Input for user to test headlines
headline = st.text_input("Enter a news headline:")

# Function to predict the sentiment of a headline
def predict_headline(headline):
    # Tokenize and encode the input text
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    # Put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        # Get model predictions
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Define label mapping
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predicted_label = label_map[probabilities.argmax()]
    confidence = probabilities.max()
    return predicted_label, confidence

# Analyze button
if st.button("Analyze"):
    if headline:
        st.write(f"Analyzing: {headline}")
        predicted_label, confidence = predict_headline(headline)
        st.success(f"Classification: {predicted_label} (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter a headline to analyze.")

# Footer
st.markdown(
    """---
    ##### Made with ❤️ using Streamlit
    © 2024 Your Name
    """
)
