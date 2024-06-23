import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps

# Load the models and preprocessing pipeline
models = {}
model_names = ["random_forest", "decision_tree", "k_nearest_neighbors", "support_vector_machine"]
for model_name in model_names:
    with open(f'{model_name}_model.pkl', 'rb') as f:
        models[model_name] = pickle.load(f)

with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# Convert to grayscale function
def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

# Augment image function
def augment_image(image):
    angle = np.random.uniform(-20, 20)  # Random rotation
    image = image.rotate(angle)
    if np.random.rand() > 0.5:  # Random horizontal flip
        image = ImageOps.mirror(image)
    return image

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = augment_image(image)
    image = convert_to_grayscale(image)
    image_array = np.array(image).flatten()
    image_array = preprocessing_pipeline.transform([image_array])
    return image_array

# Streamlit app
st.title("Satellite Image Classifier")

uploaded_file = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    preprocessed_image = preprocess_image(image)

    categories = ["cloudy", "desert", "green_area", "water"]

    for model_name, model in models.items():
        prediction = model.predict(preprocessed_image)
        st.write(f"{model_name.replace('_', ' ').title()} Prediction: {categories[prediction[0]]}")
