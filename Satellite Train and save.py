import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your dataset
data_path = "data"

# Categories in the dataset
categories = ["cloudy", "desert", "green_area", "water"]

# Function to load and preprocess images
def load_and_preprocess_images(data_path, categories):
    images = []
    labels = []

    for category_id, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        for file_name in os.listdir(category_path):
            image_path = os.path.join(category_path, file_name)
            image = Image.open(image_path)
            image = image.resize((128, 128))  # Resize images to a standard size

            # Data augmentation: random rotation and horizontal flip
            angle = np.random.uniform(-20, 20)
            image = image.rotate(angle)
            if np.random.rand() > 0.5:
                image = ImageOps.mirror(image)

            # Convert to grayscale
            image = ImageOps.grayscale(image)

            # Convert image to numpy array and flatten
            image_array = np.array(image).flatten()

            # Append image array and label to lists
            images.append(image_array)
            labels.append(category_id)

    return np.array(images), np.array(labels)

# Load and preprocess images
images, labels = load_and_preprocess_images(data_path, categories)

# Save preprocessed images and labels to CSV
df = pd.DataFrame(images)
df['label'] = labels
df.to_csv("satellite_images.csv", index=False)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Apply preprocessing pipeline to training and testing sets
X_train = preprocessing_pipeline.fit_transform(X_train)
X_test = preprocessing_pipeline.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Train models and evaluate performance
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print()

    # Save the best model using pickle
    pickle.dump(model, open(f"{model_name.lower().replace(' ', '_')}_model.pkl", 'wb'))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.show()

    # Clear plot to avoid overlapping in loop
    plt.clf()

# Save preprocessing pipeline using pickle
pickle.dump(preprocessing_pipeline, open('preprocessing_pipeline.pkl', 'wb'))

# Visualize count plot of categories
sns.countplot(x=labels, palette="viridis")
plt.xticks(ticks=np.arange(len(categories)), labels=categories)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Distribution of Satellite Image Categories')
plt.savefig('count_plot.png')
plt.show()
