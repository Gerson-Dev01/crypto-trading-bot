import pandas as pd
import numpy as np
import os

def procesar_datos():
    print("\n⚙️ Iniciando preprocesamiento...")
    
    try:
        # 1. Cargar datos
        df = pd.read_csv('data/btc.csv')
        print("Columnas encontradas:", df.columns.tolist())
        
        # 2. Verificar y limpiar columna Datetime
        if 'Datetime' not in df.columns:
            # Si no existe, intentar con la primera columna
            df = df.rename(columns={df.columns[0]: 'Datetime'})
        
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.dropna(subset=['Datetime'])
        
        # 3. Verificar columnas OHLCV
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Columna requerida faltante: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. Calcular indicadores
        df = calcular_indicadores(df)
        
        # 5. Guardar
        df.to_csv('data/btc_preprocessed.csv', index=False)
        print(f"✅ Preprocesamiento completado: {len(df)} filas guardadas")
        return True
        
    except Exception as e:
        print(f"❌ Error crítico: {str(e)}")
        return False

def calcular_indicadores(df):
    """Calcula indicadores técnicos"""
    # 1. Convertir a numérico
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Ordenar por fecha
    df = df.sort_values('Datetime')
    
    # 3. Indicadores básicos
    df['log_return'] = np.log(df['Close']/df['Close'].shift(1))
    df['volatilidad'] = df['log_return'].rolling(24).std()
    
    # 4. Medias móviles
    for window in [5, 20, 50]:
        df[f'ma_{window}'] = df['Close'].rolling(window).mean()
    
    # 5. RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 6. Eliminar NaNs
    return df.dropna()

if __name__ == "__main__":
    if procesar_datos():
        print("\nVerificación final:")
        print(pd.read_csv('data/btc_preprocessed.csv').head())
    else:
        print("Error en el preprocesamiento. Verifica:")
        print("1. El archivo data/btc.csv existe")
        print("2. Tiene columnas: Datetime, Open, High, Low, Close, Volume")