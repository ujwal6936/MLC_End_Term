# app.py
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Importing the required libraries for image captioning
import torch
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import numpy as np

# Load pre-trained BERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
model.eval()

def generate_image_caption(image):
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),
    ])
    image = preprocess(image).unsqueeze(0)

    # Run image through BERT model
    with torch.no_grad():
        outputs = model(image.to(device))
        embeddings = outputs.last_hidden_state

    # Average pooling to get image embedding
    image_embedding = torch.mean(embeddings, dim=1).squeeze()

    # Generate caption
    inputs = tokenizer.encode("Describe the image:", add_special_tokens=True, return_tensors="pt").to(device)
    inputs_embeds = model.embeddings.word_embeddings(inputs)
    inputs_embeds[:, 1] = image_embedding
    outputs = model(inputs_embeds=inputs_embeds)

    # Get the prediction
    logits = outputs.logits
    caption_ids = torch.argmax(logits, dim=-1)

    # Decode the prediction
    caption = tokenizer.decode(caption_ids.squeeze().tolist())

    return caption

    
def main():
    st.title("Image Captioning App")
    st.write("Upload an image and get its description.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Generating description...")

        # Generate and display image caption
        caption = generate_image_caption(image)
        st.write("### Description:")
        st.write(caption)

if __name__ == '__main__':
    main()
