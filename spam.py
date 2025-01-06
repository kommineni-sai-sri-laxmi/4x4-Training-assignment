import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    feature_extraction = pickle.load(vectorizer_file)

# Title of the web application
st.title("Email Spam Classifier")

# Instructions
st.write("""
    This app classifies an email as either "Spam" or "Ham".
    Enter the text of the email below and click on "Classify".
""")

# Input text from the user
input_email = st.text_area("Enter the email content:")

# Button to make prediction
if st.button("Classify"):
    if input_email:
        # Transform input email text using the loaded vectorizer
        input_data_features = feature_extraction.transform([input_email])
        
        # Make prediction using the trained model
        prediction = model.predict(input_data_features)
        
        # Show the result
        if prediction[0] == 1:
            st.success("This is a **Ham** email.")
        else:
            st.warning("This is a **Spam** email.")
    else:
        st.error("Please enter the email content.")
