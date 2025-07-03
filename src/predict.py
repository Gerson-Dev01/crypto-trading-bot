import torch
import joblib
import numpy as np
import pandas as pd
from preprocess_data import preprocess_data
from train_model import LSTMModel

def load_model():
    model = LSTMModel(input_size=8)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_next_hour(model, scaler, latest_data):
    """Predice el próximo precio usando los últimos 60 puntos"""
    # Escalar datos
    scaled = scaler.transform(latest_data)
    # Crear secuencia temporal
    sequence = scaled[-60:].reshape(1, 60, -1)
    # Convertir a tensor
    tensor = torch.tensor(sequence, dtype=torch.float32)
    # Predecir
    with torch.no_grad():
        prediction = model(tensor).item()
    return prediction

if __name__ == "__main__":
    # Cargar modelo y escalador
    model = load_model()
    scaler = joblib.load("models/scaler.pkl")
    
    # Cargar datos más recientes (últimas 72 horas)
    data = pd.read_csv("data/btc.csv").tail(72)
    prediction = predict_next_hour(model, scaler, data)
    
    print(f"Predicción para la próxima hora: ${prediction:.2f}")