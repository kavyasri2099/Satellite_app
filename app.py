import streamlit as st
import pickle
import os

# Function to load models
def load_models(model_files):
    models = {}
    for model_name, model_file in model_files.items():
        try:
            with open(model_file, 'rb') as f:
                models[model_name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file {model_file} not found. Please ensure that the file exists and the path is correct.")
            st.stop()
    return models

# Main Streamlit app code
def main():
    st.title("Machine Learning Models Demo")
    
    # Check current directory and list files
    st.subheader("Current Directory Contents:")
    files_in_directory = os.listdir('.')
    for file_name in files_in_directory:
        st.write(file_name)
    
    # Define model files and their names
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'K Nearest Neighbors': 'k_nearest_neighbors_model.pkl',
        'Support Vector Machine': 'support_vector_machine_model.pkl'
    }

    # Load models
    models = load_models(model_files)
    
    # Load preprocessing pipeline
    try:
        with open('preprocessing_pipeline.pkl', 'rb') as f:
            preprocessing_pipeline = pickle.load(f)
    except FileNotFoundError:
        st.error("Preprocessing pipeline file 'preprocessing_pipeline.pkl' not found. Please ensure that the file exists and the path is correct.")
        st.stop()
    
    # Display loaded models and preprocessing pipeline details
    st.subheader("Loaded Models:")
    for model_name, model in models.items():
        st.write(f"Model: {model_name}")
        st.write(model)

    st.subheader("Preprocessing Pipeline:")
    st.write(preprocessing_pipeline)

# Run the main function
if __name__ == '__main__':
    main()
