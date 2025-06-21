# 🌤️ Weather Prediction Webapp

This is a simple machine learning project that predicts:
- 🌡️ Temperature (using regression)
- 🌦️ Weather condition (using classification)

It uses real weather data and is deployed with a clean interface using **Streamlit**.

---

## 📊 Dataset

The model was trained on daily weather data containing:
- Humidity
- Wind Speed
- Mean Pressure
- Temperature
- Weather Type

---

## 🔧 Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn (RandomForest)
- Joblib (for saving models)
- Streamlit (for UI)

---

## 📦 Project Structure

WeatherPredictionProject/
├── app.py # Streamlit app
├── weather_prediction.ipynb # Jupyter notebook for model training
├── models/
│ ├── temperature_regressor.pkl
│ ├── weather_classifier.pkl
│ └── label_encoder.pkl
├── data/
│ └── Daily weather data.csv
├── requirements.txt
└── README.md


---

## ▶️ How to Run the App

1. Clone this repository  
   ```bash
   git clone https://github.com/Satvik12221as/weatherprediction

   streamlit run app.py

## Skills Learned
Data cleaning and preprocessing
Regression & classification using scikit-learn
Model evaluation (MAE, RMSE, R², accuracy)
Web deployment using Streamlit

![app screenshot](https://github.com/user-attachments/assets/9a4575bf-c270-427e-b1f1-c66b58af0a30)

![prediction output](https://github.com/user-attachments/assets/ff7d7def-5feb-4207-af8c-4610b41cc155)





