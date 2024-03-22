import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import requests
from io import BytesIO

# Function to load and preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Function to perform image classification
def classify_image(image):
    model = models.resnet18(pretrained=True)
    model.eval()
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    # Load ImageNet class labels
    classes = []
    with open("imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Map prediction index to class name
    predicted_label = classes[predicted]

    return predicted_label

# Streamlit app
def main():
    st.title("Image Classification App")

    st.write("This app takes an image as input and performs image classification.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Perform image classification
        prediction = classify_image(image_tensor)

        # Output the prediction
        st.write("Prediction:")
        st.write(prediction)

if __name__ == '__main__':
    main()
