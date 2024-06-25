import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle

# Load the model and preprocessing pipeline
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('preprocessing_pipeline.pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = pickle.load(pipeline_file)

categories = ["cloudy", "desert", "green_area", "water"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    img_array = np.array(image).flatten().reshape(1, -1)
    img_array = preprocessing_pipeline.transform(img_array)
    return img_array

# Streamlit app settings
st.set_page_config(
    page_title="Satellite Image Classification App",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styles for modern design and animations
st.markdown(
    """
    <style>
    .title {
        font-size: 3em;
        padding: 20px;
        background-color: #00bfff; /* Colorful background */
        color: white;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .emoji {
        font-size: 1.5em;
        margin-left: 10px;
    }
    .satellite {
        position: absolute;
        width: 60px;
        height: auto;
        top: 20px;
        left: 20px;
        animation: satelliteAnimation 8s ease-in-out infinite alternate;
    }
    @keyframes satelliteAnimation {
        0% { transform: translateX(-20px) rotate(0deg); }
        100% { transform: translateX(20px) rotate(360deg); }
    }
    .star {
        position: absolute;
        width: 20px;
        height: auto;
        top: 50px;
        right: 50px;
        animation: starAnimation 6s ease-in-out infinite alternate;
    }
    @keyframes starAnimation {
        0% { transform: translateY(0px) rotate(0deg); }
        100% { transform: translateY(-10px) rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

# App title with emoji and animated satellite/star images
st.markdown('''
    <h1 class="title">🛰️ Satellite Image Classification App <span class="emoji">🌍</span></h1>
    <img class="satellite" src="https://github.com/kavyasri2099/Satellite_app/blob/main/assets/Background.jpg" alt="Satellite">
    <img class="star" src="https://github.com/kavyasri2099/Satellite_app/blob/main/assets/Background.jpg" alt="Star">
    ''', unsafe_allow_html=True)

# Description
st.write("Upload an image, and the model will classify it into one of the following categories: Cloudy, Desert, Green Area, Water")

# File uploader for image selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Preprocess the image and make a prediction
        img_array = preprocess_image(image)
        prediction = rf_model.predict(img_array)
        category = categories[prediction[0]]
        
        # Display the prediction result with emoji
        st.success(f"The image is classified as: **{category.capitalize()}** 🎉")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
