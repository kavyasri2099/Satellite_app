import streamlit as st
import pandas as pd
import pickle
import os

st.title('Satellite Image Classification')

# Define the paths to the model files
model_files = {
    'Random Forest': 'random_forest_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'K Nearest Neighbors': 'k_nearest_neighbors_model.pkl',
    'Support Vector Machine': 'support_vector_machine_model.pkl'
}

# Check if the model files exist
for model_name, model_file in model_files.items():
    if not os.path.exists(model_file):
        st.error(f"Model file {model_file} not found. Please ensure that the file exists and the path is correct.")
        st.stop()

# Load the models
models = {}
for model_name, model_file in model_files.items():
    with open(model_file, 'rb') as f:
        models[model_name] = pickle.load(f)

# Load the preprocessing pipeline
preprocessing_pipeline_file = 'preprocessing_pipeline.pkl'
if not os.path.exists(preprocessing_pipeline_file):
    st.error(f"Preprocessing pipeline file {preprocessing_pipeline_file} not found. Please ensure that the file exists and the path is correct.")
    st.stop()

with open(preprocessing_pipeline_file, 'rb') as f:
    preprocessing_pipeline = pickle.load(f)

# Function to make predictions
def predict(model, data):
    data_preprocessed = preprocessing_pipeline.transform(data)
    prediction = model.predict(data_preprocessed)
    return prediction

# UI for file upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    
    model_choice = st.selectbox('Choose a model', list(models.keys()))
    if st.button('Predict'):
        model = models[model_choice]
        predictions = predict(model, data)
        st.write('Predictions:')
        st.write(predictions)
