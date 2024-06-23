import os
import pickle
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import PyGithub
from github import Github

# Path to your dataset
data_path = "C:/Users/Lenovo/data1"

# Categories in the dataset
categories = ["cloudy", "desert", "green_area", "water"]

# Function to augment image
def augment_image(image):
    angle = np.random.uniform(-20, 20)  # Random rotation
    image = image.rotate(angle)
    
    if np.random.rand() > 0.5:  # Random horizontal flip
        image = ImageOps.mirror(image)
    
    return image

# Function to adjust brightness
def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# Function to convert to grayscale
def convert_to_grayscale(image):
    return ImageOps.grayscale(image)

# Function to process and save images to CSV
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

    df = pd.DataFrame(images)  # Save to CSV
    df['label'] = labels
    df.to_csv("satellite_images.csv", index=False)

    return images, labels

# Load and process images
images, labels = process_and_save_images(data_path, categories)

# Define models globally
models = {
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Function to train and evaluate models
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion
        }
        print(f"{name} - Accuracy: {accuracy:.4f}")

    return results

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess data
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

# Train and evaluate models
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Print and visualize results
print("\nModel Evaluation Results:")
for name, result in results.items():
    print(f"\n{name}:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification Report:")
    print(result['classification_report'])
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

# Save models as .pkl files
model_files = {}
for name, model in models.items():
    filename = f"{name.lower().replace(' ', '_')}_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    model_files[name] = filename

# Commit files to GitHub repository
def commit_to_github(repo_path, github_token, model_files):
    # Initialize GitHub instance
    g = Github(github_token)
    repo = g.get_repo(repo_path)
    
    # Add and commit each .pkl file
    for model_name, file_path in model_files.items():
        file_content = open(file_path, 'rb').read()
        try:
            # Check if the file exists
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, f"Update {model_name} model", file_content, contents.sha)
            print(f"Updated {model_name} model on GitHub.")
        except Exception as e:
            # File doesn't exist, create a new file
            repo.create_file(file_path, f"Add {model_name} model", file_content)
            print(f"Added {model_name} model to GitHub.")

# Replace with your GitHub repository details
github_token = 'token'
repo_path = 'kavyasri2099/Satellite_app'

# Commit .pkl files to GitHub
commit_to_github(repo_path, github_token, model_files)
