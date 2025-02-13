import streamlit as st
import pandas as pd
import pickle
import os


 
st.title("Covid Classification")
 
Cough_symptoms = st.radio("Cough Symptoms", [True, False])
Fever = st.radio("Fever", [True, False])
Sore_throat = st.radio("Sore throat", [True, False])
Shortness_of_breath  = st.radio("Shortness_of_breath", [True, False])
Headache = st.radio("Headache", [True, False])
Known_contact = st.selectbox("Known_contact", ['Abroad', 'Contact with confirmed','Other'])
 
if Known_contact == 'Abroad':
    Known_contact = 0
elif Known_contact == 'Contact with confirmed':
    Known_contact = 1
else:
    Known_contact = 2
 
df = pd.DataFrame({'Cough_symptoms':[Cough_symptoms],
      'Fever':[Fever],
      'Sore_throat':[Sore_throat],
       'Shortness_of_breath':[Shortness_of_breath],
       'Headache':[Headache],
       'Known_contact':[Known_contact]
})
 
try:
    load_model = pickle.load(open('Covid_Classification.pickle', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

if st.button("Submit"):
    st.write("Success")
    pred = load_model.predict(df)
    if pred == 1:
        st.write("Positive")
    else:
        st.write("Negative")

model_path = os.path.join(os.getcwd(), 'Covid_Classification.pickle')
load_model = pickle.load(open(model_path, 'rb'))