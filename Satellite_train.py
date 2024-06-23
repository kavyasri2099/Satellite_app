import os
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Categories in the dataset
categories = ["cloudy", "desert", "green_area", "water"]

# Function to augment images
def augment_image(image):
    angle = np.random.uniform(-20, 20)
    image = image.rotate(angle)
    if np.random.rand() > 0.5:
        image = ImageOps.mirror(image)
    return image

def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

def process_images(data_path, categories):
    images, labels = [], []
    for category in categories:
        category_path = os.path.join(data_path, category)
        label = categories.index(category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img = augment_image(img)
            img = convert_to_grayscale(img)
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Data path where the folders are located
data_path = "."

# Process images
images, labels = process_images(data_path, categories)

# Check if any images were processed
if images.size == 0 or labels.size == 0:
    raise ValueError("No images or labels were processed.")

# Save processed images to CSV
df = pd.DataFrame(images)
df['label'] = labels
df.to_csv("satellite_images.csv", index=False)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([('scaler', StandardScaler())])
X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

# Train and save models
models = {
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

params = {
    "RandomForest": {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']},
    "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
    "DecisionTree": {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [None, 10, 20, 30]},
    "LogisticRegression": {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
}

best_accuracy = 0
best_model_name = ""
best_model = None

for model_name in models:
    print(f"Training {model_name}...")
    if model_name in params:
        grid_search = GridSearchCV(models[model_name], params[model_name], cv=3, verbose=1)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        models[model_name].fit(X_train, y_train)
        model = models[model_name]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    with open(f"{model_name.lower()}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f"Best model: {best_model_name} with accuracy {best_accuracy}")

# Save the preprocessing pipeline
with open("preprocessing_pipeline.pkl", 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)
