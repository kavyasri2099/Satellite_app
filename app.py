import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler

# Load the model and preprocessing pipeline
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('preprocessing_pipeline.pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = pickle.load(pipeline_file)

# Assuming StandardScaler is part of the pipeline, you may need to redefine it
# to match the correct number of features.
# Example:
# scaler = StandardScaler()
# scaler.mean_ = preprocessing_pipeline.named_steps['scaler'].mean_
# scaler.scale_ = preprocessing_pipeline.named_steps['scaler'].scale_

categories = ["Cloudy", "Desert", "Green Area", "Water"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    img_array = np.array(image)  # No need to flatten
    img_array = img_array.reshape(-1, 128, 128, 1)  # Reshape to match expected input
    img_array = preprocessing_pipeline.transform(img_array)
    return img_array

# Function to fetch the background image
def fetch_background_image():
    url = "https://github.com/kavyasri2099/Satellite_app/raw/main/assets/Background.jpg"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.warning("Failed to load background image.")
            return None
    except Exception as e:
        st.error(f"Error fetching background image: {e}")
        return None

# Function to classify the image
def classify_image(image):
    img_array = preprocess_image(image)
    prediction = rf_model.predict(img_array)
    return categories[prediction[0]]

# Streamlit app settings
st.set_page_config(
    page_title="Satellite Image Classification App",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fetch and display background image
background_image = fetch_background_image()
if background_image:
    st.image(background_image, use_column_width=True)

# App title and description
st.markdown(
    """
    <style>
    .title {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: white;
        font-size: 3em;
        z-index: 1;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """
    , unsafe_allow_html=True)

st.markdown('<div class="title">🛰️ Satellite Image Classification App 🌍</div>', unsafe_allow_html=True)
st.markdown(
    "Upload an image, and the model will classify it into one of the following categories: Cloudy, Desert, Green Area, Water"
)

# File uploader for image selection
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Classify the image
        prediction = classify_image(image)

        # Display the prediction result
        st.success(f"The image is classified as: **{prediction}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")
