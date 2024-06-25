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

# CSS for animated title
st.markdown("""
    <style>
        .title-container {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
            }
        }
        
        .title-text {
            font-size: 3rem;
            font-weight: bold;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .content {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app settings
st.set_page_config(
    page_title="Satellite Image Classification App",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Animated title
st.markdown('<div class="title-container"><h1 class="title-text">🛰️ Satellite Image Classification App 🌍</h1></div>', unsafe_allow_html=True)

# Description
st.write("Upload an image, and the model will classify it into one of the following categories: Cloudy, Desert, Green Area, Water")

# File uploader for image selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image and make a prediction
        img_array = preprocess_image(image)
        prediction = rf_model.predict(img_array)
        category = categories[prediction[0]]

        # Display the prediction result
        st.success(f"The image is classified as: {category.capitalize()}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
