# features.py
import pandas as pd
import numpy as np

def clean_data(df):
    """Limpieza robusta de datos antes de calcular indicadores"""
    df = df.copy()
    
    # 1. Manejo de valores infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. Rellenar NaNs con método apropiado para cada columna
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            # Para OHLC usar interpolación, para volumen usar fillna
            if col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].interpolate()
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # 3. Eliminar filas con NaNs restantes
    df = df.dropna()
    
    return df

def safe_divide(numerator, denominator):
    """División segura que devuelve una Serie de pandas"""
    # Convertir a Series si son arrays de NumPy
    if isinstance(numerator, np.ndarray):
        numerator = pd.Series(numerator)
    if isinstance(denominator, np.ndarray):
        denominator = pd.Series(denominator)
    
    # Realizar división segura
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            (denominator != 0) & (~np.isinf(numerator)) & (~np.isinf(denominator)),
            numerator / denominator,
            0
        )
    
    # Convertir a Serie manteniendo el índice
    result = pd.Series(
        np.nan_to_num(result, nan=0, posinf=1e10, neginf=-1e10),
        index=numerator.index if hasattr(numerator, 'index') else None
    )
    return result

def calculate_technical_indicators(df):
    """Versión robusta del cálculo de indicadores"""
    # Verificar que es un DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    df = clean_data(df)
    
    # 1. Retornos y volatilidad con protección
    price_ratio = safe_divide(df['Close'], df['Close'].shift(1))
    df['log_return'] = np.log(price_ratio)  # Cambiado de log1p a log
    
    # Asegurar que seguimos trabajando con un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Los datos deben permanecer como DataFrame")
    
    df['volatilidad'] = df['log_return'].rolling(window=10, min_periods=5).std()
    
    # 2. Medias móviles con mínimo de periodos
    for window in [5, 10, 20, 50, 100]:
        df[f'ma_{window}'] = df['Close'].rolling(window, min_periods=int(window/2)).mean()
    
    # 3. RSI con protección
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.rolling(14, min_periods=7).mean()
    avg_loss = loss.rolling(14, min_periods=7).mean()
    
    rs = safe_divide(avg_gain, avg_loss)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].clip(0, 100)

    # ... (resto de tus funciones de indicadores se mantienen igual)
    # Solo asegúrate de que cada operación mantenga el DataFrame

    return df

def generate_labels(df, horizon=3, buy_threshold=0.005, sell_threshold=-0.005):
    """Generación de labels con protección"""
    df = df.copy()
    
    # 1. Retorno futuro protegido
    df['future_return'] = safe_divide(df['Close'].shift(-horizon), df['Close']) - 1
    
    # 2. Señales con umbrales dinámicos
    median_return = df['future_return'].median()
    buy_threshold = max(buy_threshold, median_return * 1.5)
    sell_threshold = min(sell_threshold, median_return * 1.5)
    
    conditions = [
        (df['future_return'] <= sell_threshold),
        (df['future_return'] > sell_threshold) & (df['future_return'] < buy_threshold),
        (df['future_return'] >= buy_threshold)
    ]
    choices = [0, 1, 2]
    df['signal'] = np.select(conditions, choices, default=1)  # Default: Mantener
    
    return df

def generate_features(df, min_rows_required=1000):
    """Pipeline completo con validación mejorada"""
    # Verificar que la entrada es un DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    
    # Validación inicial de columnas
    required_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Datetime']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas faltantes: {missing}")
    
    # Convertir y ordenar datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    
    try:
        # Calcular indicadores
        df = calculate_technical_indicators(df)
        
        # Generar labels
        df = generate_labels(df)
        
        # Validación final
        initial_rows = len(df)
        df = df.dropna()
        
        if len(df) < min_rows_required:
            raise ValueError(
                f"Datos insuficientes. Se requieren {min_rows_required} filas, hay {len(df)}"
            )
        
        print(f"✅ Datos procesados: {initial_rows} → {len(df)} filas")
        return df
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    # Ejemplo con verificación mejorada
    try:
        data = pd.read_csv("data/btc_preprocessed.csv")
        processed_data = generate_features(data)
        processed_data.to_csv("data/btc_with_features.csv", index=False)
        print("✅ Proceso completado exitosamente")
    except Exception as e:
        print(f"❌ Error: {str(e)}")