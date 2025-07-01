import pandas as pd
import numpy as np
import talib as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

def preprocess_data(file_path, test_size=0.2, n_splits=5, sequence_length=60):
    """
    Preprocesa datos de trading corrigiendo data leakage y añadiendo features avanzados
    :param file_path: Ruta al archivo CSV con datos históricos
    :param test_size: Porcentaje de datos para test (0.0-1.0)
    :param n_splits: Número de splits para validación cruzada temporal
    :param sequence_length: Longitud de secuencias para LSTM
    :return: Tuplas con datos de entrenamiento y validación
    """
    # 1. Carga de datos con verificación de integridad
    try:
        data = pd.read_csv(file_path)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Columna requerida faltante: {col}")
        
        # Verificar datos faltantes
        if data.isnull().sum().any():
            print(f"[WARN] {data.isnull().sum().sum()} valores nulos encontrados. Imputando...")
            data = data.ffill().bfill()
    except Exception as e:
        print(f"[ERROR] Fallo al cargar datos: {str(e)}")
        return None

    # 2. Feature Engineering Avanzado
    print("Añadiendo indicadores técnicos...")
    # Indicadores básicos
    data['returns'] = np.log(data['close'] / data['close'].shift(1))
    data['volatility'] = data['returns'].rolling(window=24).std() * np.sqrt(24)  # Volatilidad horaria
    
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
    
    # 3. División temporal ANTES de escalar (evita data leakage)
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # 4. Escalado SEPARADO para train/test
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Columnas a escalar (excluyendo volumen y retornos)
    feature_cols = ['close', 'rsi', 'macd', 'ema_12', 'volatility', 'adx', 'volume_ma', 'obv']
    
    # Escalar train
    train_scaled = scaler.fit_transform(train_data[feature_cols])
    # Escalar test con parámetros de train (IMPORTANTE!)
    test_scaled = scaler.transform(test_data[feature_cols])
    
    # 5. Crear secuencias temporales para LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length - 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, 0])  # Predecir próximo 'close'
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)
    
    # 6. Validación cruzada temporal
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_splits = []
    for train_index, val_index in tscv.split(X_train):
        cv_splits.append((
            X_train[train_index], y_train[train_index],
            X_train[val_index], y_train[val_index]
        ))
    
    print(f"Preprocesamiento completado. Dimensiones:")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"CV splits: {len(cv_splits)}")
    
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