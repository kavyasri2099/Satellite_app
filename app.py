import os
import streamlit as st
import pickle
import numpy as np
from PIL import Image, ImageOps

# Determine absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Directory where models are stored

# Load models
models = {}
model_names = ["random_forest", "decision_tree", "k_nearest_neighbors", "support_vector_machine"]
for model_name in model_names:
    model_path = os.path.join(MODELS_DIR, f'{model_name}_model.pkl')  # Path to each model
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
    else:
        st.error(f"Model file {model_path} not found!")

# Load preprocessing pipeline (if needed)
preprocessing_pipeline_path = os.path.join(BASE_DIR, 'preprocessing_pipeline.pkl')
if os.path.exists(preprocessing_pipeline_path):
    with open(preprocessing_pipeline_path, 'rb') as f:
        preprocessing_pipeline = pickle.load(f)
else:
    preprocessing_pipeline = None

# Streamlit app
st.title("Satellite Image Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    image_array = np.array(image).flatten()
    if preprocessing_pipeline is not None:
        image_array = preprocessing_pipeline.transform([image_array])

    # Predict using each model
    for model_name, model in models.items():
        prediction = model.predict(image_array)
        st.write(f"{model_name.replace('_', ' ').title()} Prediction: {prediction[0]}")

# Display predictions
