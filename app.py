import streamlit as st
import pickle
import numpy as np
from PIL import Image, ImageOps

# Load models
models = {}
model_names = ["random_forest", "decision_tree", "k_nearest_neighbors", "support_vector_machine"]
for model_name in model_names:
    with open(f'{model_name}_model.pkl', 'rb') as f:
        models[model_name] = pickle.load(f)

# Load preprocessing pipeline
with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# App title
st.title("Satellite Image Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    image = image.resize((128, 128))
    image = ImageOps.grayscale(image)
    image_array = np.array(image).flatten()
    image_array = preprocessing_pipeline.transform([image_array])

    # Predict using each model
    for model_name, model in models.items():
        prediction = model.predict(image_array)
        st.write(f"{model_name.replace('_', ' ').title()} Prediction: {prediction[0]}")

# Display predictions
