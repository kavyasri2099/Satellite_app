import streamlit as st
import pickle
import numpy as np
from PIL import Image, ImageOps

# Load the models and preprocessing pipeline
model_files = {
    "Random Forest": "random_forest_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "K-Nearest Neighbors": "k_nearest_neighbors_model.pkl",
    "Support Vector Machine": "support_vector_machine_model.pkl"
}

models = {}
for model_name, model_file in model_files.items():
    with open(model_file, 'rb') as f:
        models[model_name] = pickle.load(f)

with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    image_array = np.array(image).flatten()
    image_array = preprocessing_pipeline.transform([image_array])
    return image_array

# Streamlit app
st.title("Satellite Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict with each model
    for model_name, model in models.items():
        prediction = model.predict(image_array)
        st.write(f"{model_name} Prediction: {prediction[0]}")
