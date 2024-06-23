# Satellite Image Classification Project

This project involves training multiple machine learning models to classify satellite images into categories such as cloudy, desert, green area, and water. After training, the best models are saved using pickle for deployment in a Streamlit web application.

### Files:

1. **train_and_save_models.py**: This script preprocesses satellite images, trains several machine learning models (Random Forest, KNN, Decision Tree, Naive Bayes, Logistic Regression), evaluates their performance using cross-validation, and saves the best models using pickle.

2. **app.py**: This is a Streamlit web application for uploading an image and using the trained models to predict its category. It loads the saved models and preprocessing pipeline from pickle files and allows users to classify satellite images interactively.

3. **preprocessing_pipeline.pkl**: Pickle file containing the preprocessing pipeline (StandardScaler) used to preprocess images before feeding them into the models.

4. **rf_model.pkl, knn_model.pkl, dt_model.pkl, nb_model.pkl, lr_model.pkl**: Pickle files containing the trained Random Forest, KNN, Decision Tree, Naive Bayes, and Logistic Regression models respectively.

5. **requirements.txt**: List of Python dependencies required to run the scripts. Install these using `pip install -r requirements.txt`.

6. **satellite_images.csv**: CSV file containing preprocessed images and their corresponding labels after augmentation and conversion to grayscale.

### Usage:

1. **Training Models:**
   - Run `python train_and_save_models.py` to preprocess images, train models, evaluate their performance, and save the best models using pickle.

2. **Running the Streamlit App:**
   - After training and saving models, run `streamlit run app.py` to start the Streamlit web application locally.
   - Upload a satellite image, select a model, and see the predicted category based on the trained models.

### Requirements:

Install the required Python dependencies using:

**pip install -r requirements.txt**


### Libraries Used:

- Streamlit
- Pandas
- NumPy
- PIL (Pillow)
- Scikit-learn
- Seaborn
- Matplotlib

