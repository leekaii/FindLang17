import pickle
import streamlit as st
import os
import pandas as pd
from sklearn import feature_extraction

# Load dataset for testing
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

# Model descriptions and file names
model_info = {
    "logistic_regression": {
        "description": "Logistic Regression is a statistical model used for binary classification problems. It predicts the probability of a binary outcome based on one or more predictor variables.",
        "file_name": "logistic_regression_model.pckl"
    },
    "random_forest": {
        "description": "Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.",
        "file_name": "random_forest_model.pckl"
    },
    "svc": {
        "description": "Support Vector Classifier (SVC) is a supervised learning model that performs classification by finding the hyperplane that best divides a dataset into classes.",
        "file_name": "svc_model.pckl"
    }
}

# Load the model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found. Please check the path.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Initialize the Streamlit app
st.title("FINDLANG17")

# Model selection
model_option = st.selectbox(
    "Choose the model",
    list(model_info.keys())
)

# Display model description
model_desc = model_info[model_option]["description"]
model_file_name = model_info[model_option]["file_name"]

st.write(f"### {model_option.replace('_', ' ').title()}")
st.write(model_desc)

# Load the selected model
Lrdetect_Model = load_model(model_file_name)

# Check if model is loaded
if Lrdetect_Model:
    # Display model accuracy if available
    try:
        X_test_vec = vec.fit_transform(df['Text'])
        Y_test = df['Language']
        accuracy = Lrdetect_Model.score(X_test_vec, Y_test) * 100
        st.write(f"**Accuracy:** {accuracy:.2f}%")
    except Exception as e:
        st.write("**Accuracy information not available for this model.**")

    # User input and prediction
    input_test = st.text_input("Provide your text input here")

    if st.button("Get Language Name"):
        if input_test:
            try:
                input_test_vec = vec.transform([input_test])
                prediction = Lrdetect_Model.predict(input_test_vec)
                st.text(f"Predicted Language: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please provide some text input.")
