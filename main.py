import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('Blood-pressure-data.csv')

# Split data into input and output variables
X = data['Pulse'].values.reshape(-1, 1)
y = data[['Systolic Pressure', 'Diastolic Pressure']].values

# Train the model
model = LinearRegression()
model.fit(X, y)

# Define app
st.title("Blood Pressure Predictor")

pulse = st.slider("Enter your pulse rate:",
                  min_value=40, max_value=140, step=1)

# Make prediction
prediction = model.predict([[pulse]])

# Display prediction
st.subheader("Prediction:")
st.write("Systolic Pressure:", round(prediction[0][0], 2))
st.write("Diastolic Pressure:", round(prediction[0][1], 2))
