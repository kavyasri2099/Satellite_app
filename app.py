import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
import requests

# Load the model and preprocessing pipeline
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('preprocessing_pipeline.pkl', 'rb') as pipeline_file:
    preprocessing_pipeline = pickle.load(pipeline_file)

categories = ["Cloudy", "Desert", "Green Area", "Water"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    img_array = np.array(image).flatten().reshape(1, -1)
    img_array = preprocessing_pipeline.transform(img_array)
    return img_array

# Function to fetch a random wallpaper
def fetch_wallpaper():
    url = "https://source.unsplash.com/random/1600x900"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.url
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching wallpaper: {e}")
        return None

# Streamlit app settings
st.set_page_config(
    page_title="Satellite Image Classification App",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Satellite Image Classification App")
st.markdown(
    "Upload an image, and the model will classify it into one of the following categories: Cloudy, Desert, Green Area, Water"
)

# Sidebar with wallpaper and other widgets
st.sidebar.title("Customize Your Experience")
wallpaper_url = fetch_wallpaper()
if wallpaper_url:
    st.sidebar.image(wallpaper_url, caption="Wallpaper", use_column_width=True)
else:
    st.sidebar.warning("Failed to load wallpaper. Check your internet connection.")

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
        st.success(f"The image is classified as: **{category}**")

        # Additional interactive elements
        st.markdown("---")
        st.subheader("Explore More")
        if st.button("Show Random Wallpaper"):
            wallpaper_url = fetch_wallpaper()
            if wallpaper_url:
                st.image(wallpaper_url, caption="Random Wallpaper", use_column_width=True)
            else:
                st.warning("Failed to load wallpaper. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
