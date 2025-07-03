import numpy as np
import pandas as pd
import joblib
import torch
from train_model import LSTMModel
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def load_model():
    """Carga el modelo entrenado y el escalador"""
    model = LSTMModel(input_size=8)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def predict_sequence(model, scaler, data):
    """Predice una secuencia completa de datos"""
    predictions = []
    for i in range(60, len(data)):
        sequence = data[i-60:i]
        scaled_seq = scaler.transform(sequence)
        tensor_seq = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(tensor_seq).item()
        predictions.append(pred)
    return predictions

def backtest_strategy():
    """Prueba retrospectiva de la estrategia de trading"""
    # Cargar recursos
    model, scaler = load_model()
    data = pd.read_csv("data/btc.csv")
    
    # Obtener solo las columnas de características
    feature_cols = ['close', 'rsi', 'macd', 'ema_12', 'volatility', 'adx', 'volume_ma', 'obv']
    data = data[feature_cols].dropna()
    
    # Predecir todo el conjunto de prueba
    predictions = predict_sequence(model, scaler, data.values)
    
    # Obtener precios reales (desplazados para coincidir con predicciones)
    actual_prices = data['close'].values[60:]
    
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
        if predicted_price > current_price * 1.01:  # Predice subida >1%
            if position == 0:  # Comprar
                position = capital / current_price
                capital = 0
                buy_signals.append(i)
        elif predicted_price < current_price * 0.99:  # Predice bajada >1%
            if position > 0:  # Vender
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
    
    # Métricas de desempeño
    roi = (current_value / 1000 - 1) * 100
    sharpe_ratio = (np.mean(capital_history) - 1000) / np.std(capital_history)
    r2 = r2_score(actual_prices, predictions)
    
    print("\nResultados Backtesting:")
    print(f"Capital inicial: $1000.00")
    print(f"Capital final: ${current_value:.2f}")
    print(f"Retorno: {roi:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"R² Score: {r2:.4f}")
    
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
    plt.savefig("results/backtest_results.png")
    plt.close()
    print("Resultados de backtest guardados en results/backtest_results.png")
    
    return current_value

if __name__ == "__main__":
    final_capital = backtest_strategy()