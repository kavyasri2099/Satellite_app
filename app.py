import streamlit as st
import pickle
import numpy as np
from PIL import Image, ImageOps

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.fit(image, (128, 128))
    image = ImageOps.grayscale(image)
    image = np.array(image).flatten()
    return image

# Load all models
model_paths = {
    'Random Forest': 'rf_best_model.pkl',
    'KNN': 'knn_best_model.pkl',
    'Decision Tree': 'dt_best_model.pkl',
    'Naive Bayes': 'nb_best_model.pkl',
    'Logistic Regression': 'lr_best_model.pkl'
}

models = {}

for model_name, model_path in model_paths.items():
    with open(model_path, 'rb') as f:
        models[model_name] = pickle.load(f)

# Streamlit UI
st.title("Satellite Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image).reshape(1, -1)

    # Display predictions from each model
    for model_name, model in models.items():
        prediction = model.predict(processed_image)
        st.write(f"{model_name} Prediction: {prediction[0]}")
