import joblib
import numpy as np

model = joblib.load('./model/fraud_model.pkl')
scaler = joblib.load('./model/scaler.pkl')

def predict_transaction(features):
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    return "Fraudulent" if prediction == 1 else "Legitimate"
