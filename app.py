import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open("best_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# Preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match the training size
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = np.array(image).flatten()  # Flatten the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Display the title and description
st.title("Satellite Image Classification")
st.write("Upload a satellite image, and the model will predict its category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Standardize the image
    scaler = StandardScaler()
    processed_image = scaler.fit_transform(processed_image)
    
    # Predict the category
    categories = ["cloudy", "desert", "green_area", "water"]
    prediction = model.predict(processed_image)
    predicted_category = categories[prediction[0]]
    
    # Display the prediction
    st.write(f"The model predicts this image is: **{predicted_category}**")

# Plotting the distribution of categories (if desired)
if st.checkbox("Show category distribution"):
    # Assuming you saved the DataFrame with images and labels
    df = pd.read_csv("satellite_images.csv")
    
    # Display the count plot
    st.subheader("Category Distribution in the Dataset")
    st.bar_chart(df['label'].value_counts())
