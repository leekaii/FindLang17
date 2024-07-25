import pickle
import streamlit as st
import os
import pandas as pd
df = pd.read_csv('Language Detection.csv')


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
model_path = f'{model_option.replace(" ", "_").lower()}_model.pckl'
Lrdetect_Model = load_model(model_path)

# Check if model is loaded
if Lrdetect_Model:
    # Display model accuracy if available
    if hasattr(Lrdetect_Model, 'score'):
        X_test = df['Text']  
        Y_test = df['Language'] 
        accuracy = Lrdetect_Model.score(X_test, Y_test) * 100
        st.write(f"**Accuracy:** {accuracy:.2f}%")
    else:
        st.write("**Accuracy information not available for this model.**")

    # User input and prediction
    input_test = st.text_input("Provide your text input here")

    if st.button("Get Language Name"):
        if input_test:
            try:
                prediction = Lrdetect_Model.predict([input_test])
                st.text(f"Predicted Language: {prediction[0]}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please provide some text input.")
