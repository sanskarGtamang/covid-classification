import streamlit as st
import pandas as pd
import pickle

# Title of the app
st.title("Covid Classification")

# User inputs for symptoms
Cough_symptoms = st.radio("Cough Symptoms", [True, False])
Fever = st.radio("Fever", [True, False])
Sore_throat = st.radio("Sore throat", [True, False])
Shortness_of_breath = st.radio("Shortness of breath", [True, False])
Headache = st.radio("Headache", [True, False])
Known_contact = st.selectbox("Known contact", ['Abroad', 'Contact with confirmed', 'Other'])

# Convert Known_contact to numeric values
if Known_contact == 'Abroad':
    Known_contact = 0
elif Known_contact == 'Contact with confirmed':
    Known_contact = 1
else:
    Known_contact = 2

# Prepare input data for prediction
df = pd.DataFrame({
    'Cough_symptoms': [Cough_symptoms],
    'Fever': [Fever],
    'Sore_throat': [Sore_throat],
    'Shortness_of_breath': [Shortness_of_breath],
    'Headache': [Headache],
    'Known_contact': [Known_contact]
})

# Load the model
load_model = pickle.load(open('Covid_Classification.pickle', 'rb'))
# Button to submit and show results
if st.button("Submit"):
    try:
        pred = load_model.predict(df)
        st.write("Prediction: ", "Positive" if pred == 1 else "Negative")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
