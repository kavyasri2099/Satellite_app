import streamlit as st
import requests
import pickle
import numpy as np

# Function to load model from GitHub
def load_model_from_github(url):
    response = requests.get(url)
    response.raise_for_status()
    model = pickle.loads(response.content)
    return model

# GitHub repository base URL for model files
base_url = "https://github.com/kavyasri2099/Satellite_app/main/"

# Model file URLs
model_urls = {
    "Random Forest": base_url + "random_forest_model.pkl",
    "KNN": base_url + "knn_model.pkl",
    "Decision Tree": base_url + "decision_tree_model.pkl",
    "Naive Bayes": base_url + "naive_bayes_model.pkl",
    "Logistic Regression": base_url + "logistic_regression_model.pkl"
}

# Load models
models = {name: load_model_from_github(url) for name, url in model_urls.items()}

# Streamlit app
st.title("Satellite Image Classification")

# Select model
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

# Load selected model
model = models[model_name]

# Define feature input fields
st.header(f"Input features for {model_name}")
feature_1 = st.number_input("Feature 1", min_value=0.0, max_value=1.0, value=0.5)
feature_2 = st.number_input("Feature 2", min_value=0.0, max_value=1.0, value=0.5)
feature_3 = st.number_input("Feature 3", min_value=0.0, max_value=1.0, value=0.5)
feature_4 = st.number_input("Feature 4", min_value=0.0, max_value=1.0, value=0.5)
feature_5 = st.number_input("Feature 5", min_value=0.0, max_value=1.0, value=0.5)
feature_6 = st.number_input("Feature 6", min_value=0.0, max_value=1.0, value=0.5)
feature_7 = st.number_input("Feature 7", min_value=0.0, max_value=1.0, value=0.5)
feature_8 = st.number_input("Feature 8", min_value=0.0, max_value=1.0, value=0.5)
feature_9 = st.number_input("Feature 9", min_value=0.0, max_value=1.0, value=0.5)
feature_10 = st.number_input("Feature 10", min_value=0.0, max_value=1.0, value=0.5)
feature_11 = st.number_input("Feature 11", min_value=0.0, max_value=1.0, value=0.5)
feature_12 = st.number_input("Feature 12", min_value=0.0, max_value=1.0, value=0.5)
feature_13 = st.number_input("Feature 13", min_value=0.0, max_value=1.0, value=0.5)
feature_14 = st.number_input("Feature 14", min_value=0.0, max_value=1.0, value=0.5)
feature_15 = st.number_input("Feature 15", min_value=0.0, max_value=1.0, value=0.5)
feature_16 = st.number_input("Feature 16", min_value=0.0, max_value=1.0, value=0.5)
feature_17 = st.number_input("Feature 17", min_value=0.0, max_value=1.0, value=0.5)
feature_18 = st.number_input("Feature 18", min_value=0.0, max_value=1.0, value=0.5)

# Button to make prediction
if st.button("Predict"):
    features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10, feature_11, feature_12, feature_13, feature_14, feature_15, feature_16, feature_17, feature_18]])
    prediction = model.predict(features)
    st.write(f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("Satellite Image Classification App")
