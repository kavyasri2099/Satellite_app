import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
            return response.content
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching wallpaper: {e}")
        return None

# Function to create animated sparkles effect
def animate_sparkles():
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])
    scat = ax.scatter([], [], s=150, color='gold', alpha=0.6)

    def update(frame):
        scat.set_offsets(np.random.rand(10, 2))
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=range(100), interval=100)
    return ani.to_jshtml()

# Streamlit app settings
st.set_page_config(
    page_title="Satellite Image Classification App",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description with sparkles effect
st.title("🛰️ Satellite Image Classification App 🌍")
st.markdown(
    "Upload an image, and the model will classify it into one of the following categories: Cloudy, Desert, Green Area, Water"
)

# Sidebar with animated sparkles and other widgets
st.sidebar.title("Customize Your Experience ✨")
st.sidebar.markdown(animate_sparkles(), unsafe_allow_html=True)

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

        # Display the prediction result with animated text
        st.success(f"The image is classified as: **{category}**")

        # Additional interactive elements
        st.markdown("---")
        st.subheader("Explore More 🌟")
        if st.button("Show Random Wallpaper"):
            wallpaper_image = fetch_wallpaper()
            if wallpaper_image:
                st.image(wallpaper_image, caption="Random Wallpaper", use_column_width=True)
            else:
                st.warning("Failed to load wallpaper. Please try again.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
