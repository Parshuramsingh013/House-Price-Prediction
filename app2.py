import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('house_price_model.pkl')

# Title
st.title('House Price Prediction')

# Input fields
num_bedrooms = st.number_input('Enter the number of bedrooms:', min_value=1, max_value=10)
num_bathrooms = st.number_input('Enter the number of bathrooms:', min_value=1, max_value=10)
square_footage = st.number_input('Enter the square footage:', min_value=500, max_value=10000)
age_of_house = st.number_input('Enter the age of the house:', min_value=0, max_value=100)

# Predict button
if st.button('Predict House Price'):
    features = np.array([num_bedrooms, num_bathrooms, square_footage, age_of_house]).reshape(1, -1)
    prediction = model.predict(features)
    st.write(f'Predicted House Price: ${prediction[0]:,.2f}')
