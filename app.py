import streamlit as st
import pickle
import numpy as np
from PIL import Image, ImageOps

# Load the models
model_paths = {
    "Random Forest": 'rf_best_model.pkl',
    "KNN": 'knn_best_model.pkl',
    "Decision Tree": 'dt_best_model.pkl',
    "Naive Bayes": 'nb_best_model.pkl',
    "Logistic Regression": 'lr_best_model.pkl'
}

models = {}

for model_name, model_path in model_paths.items():
    with open(model_path, 'rb') as f:
        models[model_name] = pickle.load(f)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.fit(image, (128, 128), Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    image = np.array(image).flatten().reshape(1, -1)
    return image

# Streamlit app interface
st.title("Satellite Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    preprocessed_image = preprocess_image(image)

    model_choice = st.selectbox("Choose a model", list(models.keys()))
    
    if st.button('Classify'):
        model = models[model_choice]
        prediction = model.predict(preprocessed_image)
        category = categories[prediction[0]]
        st.write(f"Predicted category: {category}")
