import streamlit as st
import os
from PIL import Image, ImageOps
import numpy as np
import pickle

# Load models
rf_model_path = 'rf_best_model.pkl'
knn_model_path = 'knn_best_model.pkl'
dt_model_path = 'dt_best_model.pkl'
nb_model_path = 'nb_best_model.pkl'
lr_model_path = 'lr_best_model.pkl'

with open(rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)

with open(knn_model_path, 'rb') as f:
    knn_model = pickle.load(f)

with open(dt_model_path, 'rb') as f:
    dt_model = pickle.load(f)

with open(nb_model_path, 'rb') as f:
    nb_model = pickle.load(f)

with open(lr_model_path, 'rb') as f:
    lr_model = pickle.load(f)

# Function to preprocess uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize to match model's expected sizing
    img = ImageOps.grayscale(img)  # Convert to grayscale
    img_array = np.array(img).flatten()  # Flatten into a 1D array
    return img_array

# Streamlit app
def main():
    st.title('Satellite Image Classifier')
    st.sidebar.title('Navigation')

    page = st.sidebar.selectbox("Choose a page", ["Home", "Upload Image"])

    if page == "Home":
        st.write("## Welcome to Satellite Image Classifier App")
        st.write("Navigate to 'Upload Image' from the sidebar to classify your own satellite image.")
        st.image('satellite_image.jpg', use_column_width=True)

    elif page == "Upload Image":
        st.write("## Upload Your Satellite Image")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

            # Preprocess the image
            img_array = preprocess_image(uploaded_file)
            img_array = img_array.reshape(1, -1)  # Reshape for model prediction

            # Classify with all models
            rf_prediction = rf_model.predict(img_array)[0]
            knn_prediction = knn_model.predict(img_array)[0]
            dt_prediction = dt_model.predict(img_array)[0]
            nb_prediction = nb_model.predict(img_array)[0]
            lr_prediction = lr_model.predict(img_array)[0]

            # Display predictions
            st.write("## Predictions")
            st.write(f"Random Forest Prediction: {rf_prediction}")
            st.write(f"KNN Prediction: {knn_prediction}")
            st.write(f"Decision Tree Prediction: {dt_prediction}")
            st.write(f"Naive Bayes Prediction: {nb_prediction}")
            st.write(f"Logistic Regression Prediction: {lr_prediction}")

if __name__ == '__main__':
    main()
