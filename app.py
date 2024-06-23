import streamlit as st
import os
import pickle
from skimage.io import imread
from skimage.transform import resize

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

# Function to load an image
def load_image(image_file):
    return imread(image_file)

# Main Streamlit app code
def main():
    st.title("Image Classification Demo")
    
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

    # File uploader for image input
    st.subheader("Upload Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Load and preprocess image (if needed)
        image = imread(uploaded_file)
        image = resize(image, (100, 100))  # Resize image to match training data size
        
        # Make predictions
        st.subheader("Prediction Results:")
        for model_name, model in models.items():
            prediction = model.predict(image.reshape(1, -1))[0]
            st.write(f"Model: {model_name}")
            st.write(f"Prediction: {prediction}")

# Run the main function
if __name__ == '__main__':
    main()
