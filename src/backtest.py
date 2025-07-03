import numpy as np
import pandas as pd
import joblib
import torch
from train_model import LSTMModel
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os
import time

def load_model():
    """Carga el modelo entrenado y el escalador"""
    model = LSTMModel(input_size=8)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def predict_batch(model, scaler, data, feature_cols):
    """Predice en batches para mayor velocidad"""
    # Preparar datos en formato batch
    sequences = []
    for i in range(60, len(data)):
        sequences.append(data[i-60:i])
    
    # Convertir a array numpy
    sequences = np.array(sequences)
    
    # Escalar los datos manteniendo los nombres de características
    # Convertir a DataFrame temporal para evitar warnings
    seqs_flat = sequences.reshape(-1, sequences.shape[-1])
    seqs_df = pd.DataFrame(seqs_flat, columns=feature_cols)
    scaled_seqs = scaler.transform(seqs_df)
    scaled_seqs = scaled_seqs.reshape(sequences.shape)
    
    # Convertir a tensor
    tensor_seqs = torch.tensor(scaled_seqs, dtype=torch.float32)
    
    # Predecir en batches
    with torch.no_grad():
        predictions = model(tensor_seqs).squeeze().numpy()
    
    return predictions

def backtest_strategy():
    """Prueba retrospectiva de la estrategia de trading"""
    start_time = time.time()
    
    # Cargar recursos
    model, scaler = load_model()
    
    # Cargar datos procesados
    if not os.path.exists("data/btc_processed.csv"):
        raise FileNotFoundError("Primero ejecuta preprocess_data.py para generar btc_processed.csv")
    
    data = pd.read_csv("data/btc_processed.csv")
    
    # Obtener solo las columnas de características en el orden correcto
    feature_cols = ['close', 'rsi', 'macd', 'ema_12', 'volatility', 'adx', 'volume_ma', 'obv']
    
    # Verificar que tenemos todas las columnas necesarias
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en datos: {missing_cols}")
    
    data = data[feature_cols].dropna()
    
    # Predecir todo el conjunto de prueba en batches
    print("Iniciando predicciones...")
    predictions = predict_batch(model, scaler, data.values, feature_cols)
    print(f"Predicciones completadas en {time.time()-start_time:.2f} segundos")
    
    # Obtener precios reales (desplazados para coincidir con predicciones)
    actual_prices = data['close'].values[60:]
    
    # Validar longitudes
    if len(predictions) != len(actual_prices):
        min_len = min(len(predictions), len(actual_prices))
        predictions = predictions[:min_len]
        actual_prices = actual_prices[:min_len]
    
    # Simular trading
    capital = 1000
    position = 0
    capital_history = [capital]
    buy_signals = []
    sell_signals = []
    
    for i in range(len(predictions)):
        current_price = actual_prices[i]
        predicted_price = predictions[i]
        
        # Estrategia simple
        if predicted_price > current_price * 1.01 and position == 0:
            position = capital / current_price
            capital = 0
            buy_signals.append(i)
        elif predicted_price < current_price * 0.99 and position > 0:
            capital = position * current_price
            position = 0
            sell_signals.append(i)
        
        # Calcular valor actual del portafolio
        current_value = capital + (position * current_price)
        capital_history.append(current_value)
    
    # Liquidar posición final si queda
    if position > 0:
        capital = position * actual_prices[-1]
        current_value = capital
    else:
        current_value = capital
    
    # Métricas de desempeño
    roi = (current_value / 1000 - 1) * 100
    returns = np.diff(capital_history) / np.array(capital_history[:-1])
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365*24) if np.std(returns) > 0 else 0
    r2 = r2_score(actual_prices, predictions)
    
    print("\nResultados Backtesting:")
    print(f"Capital inicial: $1000.00")
    print(f"Capital final: ${current_value:.2f}")
    print(f"Retorno: {roi:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Tiempo total: {time.time()-start_time:.2f} segundos")
    
    # Visualizar resultados
    plt.figure(figsize=(15, 10))
    
    # Precios y predicciones
    plt.subplot(2, 1, 1)
    plt.plot(actual_prices, label='Precio Real', alpha=0.7)
    plt.plot(predictions, label='Predicciones', alpha=0.7)
    plt.scatter(buy_signals, actual_prices[buy_signals], marker='^', color='g', label='Compras')
    plt.scatter(sell_signals, actual_prices[sell_signals], marker='v', color='r', label='Ventas')
    plt.title('Precios vs Predicciones')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True)
    
    # Evolución del capital
    plt.subplot(2, 1, 2)
    plt.plot(capital_history, label='Capital')
    plt.title('Evolución del Capital')
    plt.ylabel('Valor ($)')
    plt.xlabel('Paso de tiempo')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Asegurar que existe la carpeta results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/backtest_results.png")
    plt.close()
    print("Resultados de backtest guardados en results/backtest_results.png")
    
    return current_value

if __name__ == "__main__":
    final_capital = backtest_strategy()