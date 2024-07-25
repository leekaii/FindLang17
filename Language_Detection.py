import string
import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, pipeline, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Language Detection.csv')

# Enhanced Data Preprocessing
def remove_pun(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower()
    return text

df['Text'] = df['Text'].apply(remove_pun)

# Feature Engineering
vec = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3), analyzer='char')

def train_model(model_type):
    # Choose the model based on the input
    if model_type == 'logistic_regression':
        clf = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svc':
        clf = SVC(kernel='linear')  # Linear kernel for simplicity
    else:
        raise ValueError("Invalid model type. Choose from 'logistic_regression', 'random_forest', 'svc'.")
    
    model_pipe = pipeline.Pipeline([
        ('vec', vec),
        ('clf', clf)
    ])

    # Train the model
    X = df['Text']
    Y = df['Language']  # Assuming 'Language' is the target column
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model_pipe.fit(X_train, Y_train)

    # Evaluate the model
    predict_val = model_pipe.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, predict_val) * 100
    conf_matrix = metrics.confusion_matrix(Y_test, predict_val)

    print(f'Accuracy: {accuracy:.2f}%')
    print('Confusion Matrix:')
    print(conf_matrix)

    # Save the model
    model_path = f'{model_type}_model.pckl'
    with open(model_path, 'wb') as newfile:
        pickle.dump(model_pipe, newfile)
    
    # Check if model file was saved successfully
    if os.path.exists(model_path):
        print(f"Model saved successfully as {model_path}")
    else:
        print(f"Failed to save the model as {model_path}")

    # Plot confusion matrix
    conf_matrix_df = pd.DataFrame(conf_matrix, index=model_pipe.classes_, columns=model_pipe.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(f'{model_type}_confusion_matrix.png')  # Save confusion matrix plot
    plt.show()

    return model_pipe

# Example usage
model_type = 'random_forest'  # Choose 'logistic_regression', 'random_forest', or 'svc'
trained_model = train_model(model_type)

model_type = 'logistic_regression'  # Choose 'logistic_regression', 'random_forest', or 'svc'
trained_model = train_model(model_type)

model_type = 'svc'  # Choose 'logistic_regression', 'random_forest', or 'svc'
trained_model = train_model(model_type)

