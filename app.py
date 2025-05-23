import os
import openai
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY is not set. Please ensure it's in the .env file.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

def predict_dish(image: Image.Image) -> str:
    """Predicts the dish name from an image using BLIP model."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    caption = model.generate(**inputs)
    return processor.batch_decode(caption, skip_special_tokens=True)[0]

def query_openai(prompt: str) -> str:
    """Queries OpenAI's GPT model for information."""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a food and nutrition expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I'm sorry, I have no knowledge on this query. Error: {e}"

def process_uploaded_file(uploaded_file):
    """Handles file upload and processes image or text/pdf files."""
    try:
        if uploaded_file.type in ["image/png", "image/jpeg"]:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            predicted_dish = predict_dish(image)
            st.write(f"**Predicted Dish:** {predicted_dish}")
            dish_info = query_openai(f"Tell me about {predicted_dish}, including its nutritional benefits.")
            st.write(dish_info)
        else:
            st.write("Processing text or PDF files is under development.")
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

def rerun_app():
    """Compatibility layer for rerunning Streamlit app."""
    from streamlit.runtime.scriptrunner import RerunException
    from streamlit.runtime.state import get_script_run_ctx
    ctx = get_script_run_ctx()
    raise RerunException(ctx)

def main():
    """Main function to run the Streamlit application."""
    st.title("🍽️ Food & Nutrition AI Assistant")

    with st.sidebar:
        st.subheader("⚙️ Session Controls")
        if st.button("🧹 Clear Chat History"):
            st.session_state.clear()
            rerun_app()
        if st.button("🔒 Terminate API Usage"):
            os.environ.pop("OPENAI_API_KEY", None)
            st.warning("API Key removed. Restart required.")

    query = st.text_input("💬 Ask me anything about food, cuisine, or nutrition:")
    if query:
        response = query_openai(query)
        st.write(response)

    uploaded_file = st.file_uploader("📂 Upload an image (PNG, JPEG) or a text/PDF file:", type=["png", "jpeg", "txt", "pdf"])
    if uploaded_file:
        process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
