import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import warnings
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# Path to dataset
data_path = "./"

# Categories in the dataset
categories = ["cloudy", "desert", "green_area", "water"]

# Augmentation function
def augment_image(image):
    angle = np.random.uniform(-20, 20)  # Random rotation
    image = image.rotate(angle)
    if np.random.rand() > 0.5:  # Random horizontal flip
        image = ImageOps.mirror(image)
    return image

# Convert to grayscale function
def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

# Process images and save to CSV
def process_and_save_images(data_path, categories):
    images = []
    labels = []
    for category in categories:
        category_path = os.path.join(data_path, category)
        label = categories.index(category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path)
            img = img.resize((128, 128))  # Resize images to a standard size
            img = augment_image(img)  # Augmentation
            img = convert_to_grayscale(img)  # Convert to grayscale
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

images, labels = process_and_save_images(data_path, categories)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

# Train and save models
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, max_features='sqrt'),
    "decision_tree": DecisionTreeClassifier(),
    "k_nearest_neighbors": KNeighborsClassifier(),
    "support_vector_machine": SVC(kernel='linear')
}

for model_name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    y_pred = model.predict(X_test)

    print(f"{model_name.replace('_', ' ').title()} Model")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    print(f"Training time: {end_time - start_time} seconds\n")

    with open(f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print(f"Saved {model_name}_model.pkl")

# Save the preprocessing pipeline
with open('preprocessing_pipeline.pkl', 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)
    print("Saved preprocessing_pipeline.pkl")

print("Training script has been successfully completed. You can now run app.py.")
