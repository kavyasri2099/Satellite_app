import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(layout="wide")

# Function to load model from local file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        try:
            model = pickle.load(file)
            return model
        except Exception as e:
            st.error(f"Error loading model {file_path}: {e}")
            return None

# Model file paths
model_files = {
    "Random Forest": "random_forest_model.pkl",
    "KNN": "knn_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl"
}

# Load models
models = {name: load_model(file) for name, file in model_files.items()}

# Filter out None values
models = {name: model for name, model in models.items() if model is not None}

# Streamlit app
st.title("Satellite Image Classification")

# Check if models are loaded successfully
if not models:
    st.error("No models could be loaded.")
else:
    # Select model
    model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

    # Load selected model
    model = models[model_name]

    # Define a dummy input for prediction (as an example)
    features = np.random.rand(1, 18)

    # Button to make prediction
    if st.button("Predict"):
        prediction = model.predict(features)
        st.write(f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    st.title("Satellite Image Classification App")
