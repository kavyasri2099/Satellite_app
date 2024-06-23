# Import necessary libraries
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pandas as pd
import pickle
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Define functions for image augmentation, brightness adjustment, and grayscale conversion
def augment_image(image):
    angle = np.random.uniform(-20, 20)  # Random rotation
    image = image.rotate(angle)
    if np.random.rand() > 0.5:  # Random horizontal flip
        image = ImageOps.mirror(image)
    return image

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

# Path to your dataset
data_path = "C:/Users/Kavya/data"

# Categories in the dataset
categories = ["cloudy", "desert", "green_area", "water"]

# Function to preprocess images, extract features, and save to CSV
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
            
            img_array = np.array(img).flatten()  # Flatten image array
            images.append(img_array)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Save images and labels to CSV
    df = pd.DataFrame(images)
    df['label'] = labels
    df.to_csv("satellite_images.csv", index=False)

    return images, labels

# Preprocess images and save to CSV
images, labels = process_and_save_images(data_path, categories)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define preprocessing pipeline (in this case, StandardScaler)
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Fit preprocessing pipeline on training data and transform both training and testing data
X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

# Define machine learning models
models = {
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Dictionary to store best models
best_models = {}

# Function to train models, perform GridSearchCV, evaluate performance, and save best models
def train_models(models, X_train, X_test, y_train, y_test):
    for name, model in models.items():
        print(f"Training {name} model...")
        
        if name == 'Random Forest':
            params = {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']}
        elif name == 'KNN':
            params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        elif name == 'Decision Tree':
            params = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [None, 10, 20, 30]}
        elif name == 'Logistic Regression':
            params = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
        else:
            params = {}
        
        grid_search = GridSearchCV(model, params, cv=3, verbose=1)
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        y_pred = best_model.predict(X_test)
        
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))
        print(f"Training time: {end_time - start_time} seconds")
        
        # Save best model using pickle
        with open(f'{name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"{name} model saved successfully.\n")

# Train models
train_models(models, X_train, X_test, y_train, y_test)

# Visualize count plot of categories
sns.countplot(x=labels, palette="viridis")
plt.xticks(ticks=np.arange(len(categories)), labels=categories)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Distribution of Satellite Image Categories')
plt.show()

print("Training and saving models completed successfully.")
