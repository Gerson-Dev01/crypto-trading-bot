import pandas as pd
import numpy as np
import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(arr):
    """Limpia datos reemplazando NaNs e infinitos con valores seguros"""
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e5, neginf=-1e5)
    return arr

def verify_data(data, name):
    """Verifica y reporta problemas en los datos con más detalle"""
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        logging.warning(f"NaN encontrado en {name}: {nan_count} valores")
    
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        logging.warning(f"Inf encontrado en {name}: {inf_count} valores")
    
    # Calcular estadísticas de forma segura
    try:
        data_min = np.min(data)
        data_max = np.max(data)
        data_mean = np.mean(data)
        logging.info(f"{name} - Min: {data_min:.4f}, Max: {data_max:.4f}, Mean: {data_mean:.4f}")
        
        # Verificar si los datos son constantes
        if np.all(data == data_mean):
            logging.error(f"¡ADVERTENCIA CRÍTICA! {name} es completamente constante")
        elif data_max - data_min < 1e-10:
            logging.warning(f"¡ADVERTENCIA! {name} tiene variación mínima")
            
    except Exception as e:
        logging.error(f"Error al verificar {name}: {str(e)}")

def preprocess_data(file_path, test_size=0.2, n_splits=5, sequence_length=60):
    """
    Preprocesa datos de trading con mejor manejo de valores extremos
    """
    # 1. Carga de datos con verificación de integridad mejorada
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo de datos no encontrado: {file_path}")
        
        data = pd.read_csv(file_path)
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Verificar columnas requeridas
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columnas requeridas faltantes: {missing_cols}")
        
        # Verificar datos faltantes
        null_count = data.isnull().sum().sum()
        if null_count > 0:
            logging.warning(f"{null_count} valores nulos encontrados. Imputando...")
            data = data.ffill().bfill()
            if data.isnull().sum().sum() > 0:
                data = data.dropna()
                logging.warning(f"Eliminadas filas con valores nulos persistentes")
        
        # Convertir timestamp
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        logging.info(f"Datos cargados correctamente. Dimensiones: {data.shape}")
    except Exception as e:
        logging.error(f"Fallo al cargar datos: {str(e)}")
        return None

    # 2. Feature Engineering con manejo robusto de errores
    logging.info("Añadiendo indicadores técnicos...")
    try:
        # Indicadores básicos
        data['returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(window=24).std() * np.sqrt(24)
        
        # Osciladores - con manejo de divisiones por cero
        data['rsi'] = ta.RSI(data['close'], timeperiod=14)
        
        # MACD con verificación de valores extremos
        macd, macd_signal, _ = ta.MACD(data['close'])
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        
        # Tendencia
        data['ema_12'] = ta.EMA(data['close'], timeperiod=12)
        data['ema_26'] = ta.EMA(data['close'], timeperiod=26)
        data['adx'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Volumen
        data['volume_ma'] = data['volume'].rolling(window=5).mean()
        data['obv'] = ta.OBV(data['close'], data['volume'])
        
        # Manejar valores infinitos y NaNs
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.ffill().bfill()
        data = data.dropna()
        
        # Verificar indicadores técnicos
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                logging.warning(f"Columna {col} tiene {data[col].isnull().sum()} NaNs después de limpieza")
            if np.isinf(data[col]).any():
                logging.warning(f"Columna {col} tiene valores infinitos")
                
        logging.info(f"Datos después de indicadores: {data.shape}")
    except Exception as e:
        logging.error(f"Error en feature engineering: {str(e)}")
        return None

    # 3. División temporal ANTES de escalar
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # 4. Escalado SEPARADO con manejo de columnas constantes
    feature_cols = ['close', 'rsi', 'macd', 'ema_12', 'volatility', 'adx', 'volume_ma', 'obv']
    feature_cols = [col for col in feature_cols if col in data.columns]
    
    # Verificar y eliminar columnas constantes
    constant_cols = [col for col in feature_cols if train_data[col].nunique() == 1]
    if constant_cols:
        logging.warning(f"Columnas constantes en entrenamiento: {constant_cols}. Serán eliminadas.")
        feature_cols = [col for col in feature_cols if col not in constant_cols]
    
    scaler = StandardScaler()
    
    # Escalar y limpiar datos
    try:
        train_scaled = scaler.fit_transform(train_data[feature_cols])
        test_scaled = scaler.transform(test_data[feature_cols])
    except ValueError as e:
        logging.error(f"Error en escalado: {str(e)}")
        # Verificar varianza cero
        for col in feature_cols:
            if train_data[col].std() < 1e-10:
                logging.error(f"Columna '{col}' tiene desviación estándar casi cero: {train_data[col].std()}")
        return None
    
    train_scaled = clean_data(train_scaled)
    test_scaled = clean_data(test_scaled)
    
    # 5. Verificar datos escalados CRÍTICAMENTE
    verify_data(train_scaled, "train_scaled")
    verify_data(test_scaled, "test_scaled")
    
    # Detener el proceso si los datos de entrenamiento son constantes
    if np.all(train_scaled == train_scaled[0]):
        logging.error("¡ERROR CRÍTICO! Datos de entrenamiento son completamente constantes después del escalado")
        return None

    # 6. Crear secuencias temporales con validación
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length - 1):
            seq = data[i:(i + seq_length)]
            target = data[i + seq_length, 0]  # Predecir 'close'
            
            # Validar secuencia
            if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)) and not np.isnan(target) and not np.isinf(target):
                X.append(seq)
                y.append(target)
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test, y_test = create_sequences(test_scaled, sequence_length)
    
    # Verificar dimensiones
    if len(X_train) == 0 or len(y_train) == 0:
        logging.error("¡No se generaron secuencias de entrenamiento válidas!")
        return None
    
    # Limpiar y verificar secuencias
    X_train = clean_data(X_train)
    y_train = clean_data(y_train)
    X_test = clean_data(X_test)
    y_test = clean_data(y_test)
    
    verify_data(X_train, "X_train")
    verify_data(y_train, "y_train")
    verify_data(X_test, "X_test")
    verify_data(y_test, "y_test")
    
    # 7. Validación cruzada temporal con verificación
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_splits = []
        for train_index, val_index in tscv.split(X_train):
            # Validar índices
            if len(train_index) == 0 or len(val_index) == 0:
                logging.warning("¡Split de CV vacío! Saltando...")
                continue
                
            cv_splits.append((
                X_train[train_index], y_train[train_index],
                X_train[val_index], y_train[val_index]
            ))
    except Exception as e:
        logging.error(f"Error en CV: {str(e)}")
        cv_splits = []
    
    feature_cols = ['close', 'rsi', 'macd', 'ema_12', 'volatility', 'adx', 'volume_ma', 'obv']
    data = data[['timestamp'] + feature_cols]  # Orden explícito

    processed_file_path = 'data/btc_processed.csv'
    #data.reset_index(inplace=True)  # Convertir el índice a columna
    data.reset_index(drop=True)
    data.to_csv(processed_file_path, index=False)
    logging.info(f"Datos preprocesados guardados en {processed_file_path}")

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