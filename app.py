import streamlit as st
import os
import numpy as np
from PIL import Image
import pickle

# Load saved models and preprocessing pipeline
def load_models():
    preprocessing_pipeline = pickle.load(open('preprocessing_pipeline.pkl', 'rb'))
    rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
    knn_model = pickle.load(open('knn_model.pkl', 'rb'))
    dt_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
    nb_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))
    lr_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
    return preprocessing_pipeline, rf_model, knn_model, dt_model, nb_model, lr_model

# Load models and preprocessing pipeline
preprocessing_pipeline, rf_model, knn_model, dt_model, nb_model, lr_model = load_models()

# Function to preprocess uploaded image
def preprocess_image(image):
    # Resize image to 128x128
    image = image.resize((128, 128))
    # Convert image to numpy array
    img_array = np.array(image)
    # Flatten image array
    img_array = img_array.flatten().reshape(1, -1)
    # Scale image using preprocessing pipeline
    img_array = preprocessing_pipeline.transform(img_array)
    return img_array

# Function to make predictions
def predict_image(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
def main():
    st.title('Satellite Image Classifier')

    # Sidebar for uploading image
    st.sidebar.title('Upload Image')
    uploaded_file = st.sidebar.file_uploader("Choose a satellite image...", type=['jpg', 'png'])

    if uploaded_file is not None:
        # Display uploaded image
        st.sidebar.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Preprocess and classify image
        image = Image.open(uploaded_file)
        prediction_rf = predict_image(image, rf_model)
        prediction_knn = predict_image(image, knn_model)
        prediction_dt = predict_image(image, dt_model)
        prediction_nb = predict_image(image, nb_model)
        prediction_lr = predict_image(image, lr_model)

        categories = ['cloudy', 'desert', 'green_area', 'water']

        st.subheader('Random Forest Prediction:')
        st.write(f"Predicted category: {categories[prediction_rf[0]]}")

        st.subheader('KNN Prediction:')
        st.write(f"Predicted category: {categories[prediction_knn[0]]}")

        st.subheader('Decision Tree Prediction:')
        st.write(f"Predicted category: {categories[prediction_dt[0]]}")

        st.subheader('Naive Bayes Prediction:')
        st.write(f"Predicted category: {categories[prediction_nb[0]]}")

        st.subheader('Logistic Regression Prediction:')
        st.write(f"Predicted category: {categories[prediction_lr[0]]}")

    else:
        st.warning('Please upload an image.')

    st.sidebar.title('About')
    st.sidebar.info('This web app is designed to classify satellite images into categories: cloudy, desert, green_area, and water.')

if __name__ == '__main__':
    main()
