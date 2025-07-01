import pandas as pd
import numpy as np
import talib as ta
from sklearn.preprocessing import StandardScaler  # Cambiado a StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(arr):
    """Limpia datos reemplazando NaNs e infinitos"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e5, neginf=-1e5)
    return arr

def verify_data(data, name):
    """Verifica y reporta problemas en los datos"""
    if np.isnan(data).any():
        logging.warning(f"NaN encontrado en {name}: {np.isnan(data).sum()}")
    if np.isinf(data).any():
        logging.warning(f"Inf encontrado en {name}: {np.isinf(data).sum()}")
    
    logging.info(f"{name} - Min: {np.min(data):.4f}, Max: {np.max(data):.4f}, Mean: {np.mean(data):.4f}")

def preprocess_data(file_path, test_size=0.2, n_splits=5, sequence_length=60):
    """
    Preprocesa datos de trading con mejor manejo de valores extremos
    """
    # 1. Carga de datos con verificación de integridad
    try:
        data = pd.read_csv(file_path)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Verificar columnas requeridas
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")
        
        # Verificar datos faltantes
        if data.isnull().sum().any():
            null_count = data.isnull().sum().sum()
            logging.warning(f"{null_count} valores nulos encontrados. Imputando...")
            data = data.ffill().bfill()
        
        # Convertir timestamp
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
    except Exception as e:
        logging.error(f"Fallo al cargar datos: {str(e)}")
        return None

    # 2. Feature Engineering con manejo de errores
    logging.info("Añadiendo indicadores técnicos...")
    try:
        # Indicadores básicos
        data['returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(window=24).std() * np.sqrt(24)
        
        # Osciladores
        data['rsi'] = ta.RSI(data['close'], timeperiod=14)
        data['macd'], data['macd_signal'], _ = ta.MACD(data['close'])
        data['stoch_k'], data['stoch_d'] = ta.STOCH(data['high'], data['low'], data['close'])
        
        # Tendencia
        data['ema_12'] = ta.EMA(data['close'], timeperiod=12)
        data['ema_26'] = ta.EMA(data['close'], timeperiod=26)
        data['adx'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Volumen
        data['volume_ma'] = data['volume'].rolling(window=5).mean()
        data['obv'] = ta.OBV(data['close'], data['volume'])
        
        # Manejar valores infinitos generados por indicadores
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.ffill().bfill()
    except Exception as e:
        logging.error(f"Error en feature engineering: {str(e)}")
        return None

    # 3. División temporal ANTES de escalar
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # 4. Escalado SEPARADO con StandardScaler
    scaler = StandardScaler()  # Mejor para redes neuronales
    
    # Columnas a escalar
    feature_cols = ['close', 'rsi', 'macd', 'ema_12', 'volatility', 'adx', 'volume_ma', 'obv']
    feature_cols = [col for col in feature_cols if col in data.columns]
    
    # Escalar y limpiar datos
    train_scaled = scaler.fit_transform(train_data[feature_cols])
    test_scaled = scaler.transform(test_data[feature_cols])
    
    train_scaled = clean_data(train_scaled)
    test_scaled = clean_data(test_scaled)
    
    # 5. Verificar datos escalados
    verify_data(train_scaled, "train_scaled")
    verify_data(test_scaled, "test_scaled")
    
    # 6. Crear secuencias temporales
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length - 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, 0])  # Predecir 'close'
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)
    
    # Limpiar y verificar secuencias
    X_train = clean_data(X_train)
    y_train = clean_data(y_train)
    X_test = clean_data(X_test)
    y_test = clean_data(y_test)
    
    verify_data(X_train, "X_train")
    verify_data(y_train, "y_train")
    verify_data(X_test, "X_test")
    verify_data(y_test, "y_test")
    
    # 7. Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_splits = []
    for train_index, val_index in tscv.split(X_train):
        cv_splits.append((
            X_train[train_index], y_train[train_index],
            X_train[val_index], y_train[val_index]
        ))
    
    logging.info(f"Preprocesamiento completado. Dimensiones:")
    logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logging.info(f"CV splits: {len(cv_splits)}")
    
    return {
        'train': (X_train, y_train),
        'test': (X_test, y_test),
        'cv_splits': cv_splits,
        'scaler': scaler,
        'feature_cols': feature_cols
    }

# Ejemplo de uso
if __name__ == "__main__":
    processed_data = preprocess_data('data/btc.csv')