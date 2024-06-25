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

# CSS styles for colorful theme and title
st.markdown(
    """
    <style>
    .title {
        font-size: 3em;
        padding: 10px;
        background-color: #00bfff; /* Colorful background */
        color: white;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
        position: relative;
    }
    .emoji {
        font-size: 1.5em;
        margin-left: 10px;
    }
    .satellite {
        position: absolute;
        width: 30px;
        height: auto;
        z-index: 1;
    }
    .star {
        position: absolute;
        width: 10px;
        height: auto;
        z-index: 1;
    }
    .satellite1 {
        top: 50px;
        left: 20%;
    }
    .satellite2 {
        top: 120px;
        left: 70%;
    }
    .star1 {
        top: 30px;
        left: 30%;
    }
    .star2 {
        top: 100px;
        left: 60%;
    }
    </style>
    """
    , unsafe_allow_html=True)

# App title with emoji and satellite/star images
st.markdown('''
    <h1 class="title">🛰️ Satellite Image Classification App <span class="emoji">🌍</span></h1>
    <img class="satellite satellite1" src="https://example.com/satellite.png" alt="Satellite 1">
    <img class="satellite satellite2" src="https://example.com/satellite.png" alt="Satellite 2">
    <img class="star star1" src="https://example.com/star.png" alt="Star 1">
    <img class="star star2" src="https://example.com/star.png" alt="Star 2">
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
