import streamlit as st
import joblib
import numpy as np

# Load models
reg_model = joblib.load("models/temperature_regressor.pkl")
clf_model = joblib.load("models/weather_classifier.pkl")
le = joblib.load("models/label_encoder.pkl")

# Streamlit UI
st.title("🌤️ Weather Prediction App")
st.markdown("Enter today's data to predict temperature and weather condition:")

# User Inputs
humidity = st.slider("Humidity (%)", 0, 100, 70)
wind_speed = st.slider("Wind Speed (km/h)", 0.0, 15.0, 3.5)
mean_pressure = st.slider("Mean Pressure", 950, 1100, 1010)

if st.button("Predict Weather"):
    input_data = np.array([[humidity, wind_speed, mean_pressure]])

    # Predict temperature
    predicted_temp = reg_model.predict(input_data)[0]

    # Predict weather label
    predicted_weather_code = clf_model.predict(input_data)
    predicted_weather = le.inverse_transform(predicted_weather_code)[0]

    st.success(f"🌡️ Predicted Temperature: {predicted_temp:.2f} °C")
    st.success(f"🌦️ Predicted Weather Condition: {predicted_weather}")
