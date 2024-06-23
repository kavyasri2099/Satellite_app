import os
import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.io import imread
from skimage.transform import resize
from numpy import array

# Function to load images and labels
def load_dataset(data_directory):
    X = []
    y = []
    for class_name in os.listdir(data_directory):
        class_dir = os.path.join(data_directory, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                try:
                    # Load and resize image to a common size if needed
                    image = imread(image_path)
                    image = resize(image, (100, 100))  # Resize images to 100x100 pixels
                    X.append(image)
                    y.append(class_name)
                except Exception as e:
                    print(f"Error loading image: {image_path}. Error: {e}")
    return array(X), array(y)

# Path to your dataset directory (replace with your GitHub repository path)
data_dir = '/path/to/your/dataset'

# Load dataset
X, y = load_dataset(data_dir)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42)
}

# Train and save models
for model_name, model in models.items():
    start_time = time.time()
    # If needed, preprocess your data here (e.g., flatten images, scale features)
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    y_pred = model.predict(X_test.reshape(len(X_test), -1))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save model
    model_filename = f'{model_name.lower().replace(" ", "_")}_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"{model_filename} saved successfully.")

    # Print evaluation metrics
    print(f"\n{model_name} Model\n")
    print(f"Accuracy: {accuracy:.2f}\n")
    print(report)
    print(f"Training time: {time.time() - start_time} seconds\n")

# Save preprocessing pipeline (if applicable)
# Example: Scaling images or feature extraction
# preprocessing_pipeline = Pipeline([('scaler', StandardScaler())])
# preprocessing_pipeline.fit(X_train.reshape(len(X_train), -1))
# with open('preprocessing_pipeline.pkl', 'wb') as f:
#     pickle.dump(preprocessing_pipeline, f)

print("\nTraining script has been successfully completed.")
