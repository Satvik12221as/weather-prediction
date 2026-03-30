import streamlit as st
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

reg_model = joblib.load(os.path.join(BASE_DIR, "models", "temperature_regressor.pkl"))
clf_model = joblib.load(os.path.join(BASE_DIR, "models", "weather_classifier.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))

# Load data
data = pd.read_csv(os.path.join(BASE_DIR, "Data", "Daily weather data.csv"))

# Preprocess data
X = data[['humidity', 'wind_speed', 'meanpressure']]
y_temp = data['temperature']
y_weather = data['weather']

# Streamlit UI
st.title(" Weather Prediction App")
st.markdown("Enter today's data to predict temperature and weather condition:")

# Retrain models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
reg_model = RandomForestRegressor()
clf_model = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y_weather, test_size=0.2, random_state=42)
reg_model.fit(X, y_temp)
clf_model.fit(X, y_weather)

# User Inputs
humidity = st.slider("Humidity (%)", 0, 100, 70)
wind_speed = st.slider("Wind Speed (km/h)", 0.0, 15.0, 3.5 ,step=0.1)
mean_pressure = st.slider("Mean Pressure", 950, 1100, 1010 ,step=1)

if st.button("Predict Weather"):
    input_data = np.array([[humidity, wind_speed, mean_pressure]])

    # Predict temperature
    predicted_temp = reg_model.predict(input_data)[0]

    # Predict weather label
    predicted_weather_code = clf_model.predict(input_data)
    predicted_weather = le.inverse_transform(predicted_weather_code)[0]

    st.success(f"🌡️ Predicted Temperature: {predicted_temp:.2f} °C")
    st.success(f"🌦️ Predicted Weather Condition: {predicted_weather}")
if st.button("Predict Weather"):
    input_data = np.array([[humidity, wind_speed, mean_pressure]])

    # Predict temperature with retrained model
    predicted_temp = reg_model.predict(input_data)[0]

    # Predict weather label with retrained model
    predicted_weather = clf_model.predict(input_data)[0]

    st.success(f" Predicted Temperature: {predicted_temp:.2f} °C")
    st.success(f" Predicted Weather Condition: {predicted_weather}")