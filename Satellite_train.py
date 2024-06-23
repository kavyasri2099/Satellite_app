import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image

# Function to load images and convert them to numpy arrays
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img).flatten())
    return images

# Path to data folders
data_folders = ["cloudy", "desert", "green_area", "water"]
base_path = os.path.dirname(os.path.abspath(__file__))

# Load images and labels
X = []
y = []
for idx, folder in enumerate(data_folders):
    folder_path = os.path.join(base_path, "data", folder)
    images = load_images_from_folder(folder_path)
    X.extend(images)
    y.extend([idx] * len(images))

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "random_forest": RandomForestClassifier(random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "k_nearest_neighbors": KNeighborsClassifier(),
    "support_vector_machine": SVC(random_state=42)
}

# Train models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")

    # Save model as pickle file
    model_filename = f"{model_name}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

# Save preprocessing pipeline (if needed)
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessing_filename = "preprocessing_pipeline.pkl"
with open(preprocessing_filename, 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)

print("Models and preprocessing pipeline saved successfully.")
