import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import pickle

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

# Load the model and preprocessing pipeline
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('preprocessing_pipeline.pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = pickle.load(pipeline_file)

categories = ["Cloudy", "Desert", "Green Area", "Water"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image).flatten().reshape(1, -1)
    img_array = preprocessing_pipeline.transform(img_array)
    return img_array

# Function to classify the image
def classify_image(image):
    img_array = preprocess_image(image)
    prediction = rf_model.predict(img_array)
    return categories[prediction[0]]

# Function to add snow effect using JavaScript and CSS
def snow_effect():
    return """
    <style>
    body {
        overflow-x: hidden;
    }
    .snowflake {
        position: absolute;
        user-select: none;
        pointer-events: none;
        z-index: 1000;
        color: #fff;
        font-size: 2em;
        animation: snow linear infinite;
        transform: translate3d(0, -100%, 0);
    }
    @keyframes snow {
        0% {
            transform: translate3d(0, -100%, 0);
        }
        100% {
            transform: translate3d(100vw, 100vh, 0);
        }
    }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const numFlakes = 30;
        const flakesContainer = document.createElement('div');
        flakesContainer.className = 'snowflake-container';
        document.body.appendChild(flakesContainer);

        for (let i = 0; i < numFlakes; i++) {
            const snowflake = document.createElement('div');
            snowflake.className = 'snowflake';
            snowflake.innerHTML = '&bull;';
            snowflake.style.left = Math.random() * 100 + 'vw';
            snowflake.style.animationDuration = (Math.random() * 3 + 3) + 's';
            flakesContainer.appendChild(snowflake);
        }
    });
    </script>
    """

# Streamlit app settings
st.set_page_config(
    page_title="Satellite Image Classification App",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🛰️ Satellite Image Classification App 🌍")
st.markdown(
    "Upload an image, and the model will classify it into one of the following categories: Cloudy, Desert, Green Area, Water"
)

# Fetch and display background image
background_image = fetch_background_image()
if background_image:
    st.image(background_image, use_column_width=True)

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

        # Display snow effect after prediction result
        st.markdown(snow_effect(), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
