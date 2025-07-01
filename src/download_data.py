import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración de rutas
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, 'btc.csv')

# Parámetros de descarga
symbol = "BTC-USD"
interval = "1h"
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

def limpiar_y_renombrar(df):
    """Convierte columnas multi-index y asegura nombres estándar"""
    # 1. Limpiar multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 2. Normalizar nombres de columnas
    column_mapping = {
        'Date': 'timestamp',
        'Datetime': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Adj Close': 'adj_close'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # 3. Eliminar columnas no necesarias
    keep_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return df[[col for col in keep_columns if col in df.columns]]

def descargar_datos():
    """Descarga completa de datos históricos"""
    logging.info(f"⏳ Descargando datos históricos de {symbol} ({interval})")
    
    try:
        # Descargar datos
        data = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            logging.error("Datos vacíos recibidos de Yahoo Finance")
            return False
        
        # Procesar y renombrar columnas
        data = limpiar_y_renombrar(data.reset_index())
        
        # Eliminar zona horaria si existe
        if 'timestamp' in data.columns and hasattr(data['timestamp'], 'dt'):
            if data['timestamp'].dt.tz is not None:
                data['timestamp'] = data['timestamp'].dt.tz_convert(None)
        
        # Guardar
        data.to_csv(csv_path, index=False)
        logging.info(f"✅ Datos guardados en {csv_path} ({len(data)} filas)")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error en descarga: {str(e)}", exc_info=True)
        return False

def actualizar_datos():
    """Actualización incremental de datos"""
    try:
        # Si no existe archivo, descarga completa
        if not os.path.exists(csv_path):
            logging.warning("Archivo no encontrado. Descargando datos completos...")
            return descargar_datos()

        # Leer datos existentes
        existing = pd.read_csv(csv_path)
        
        # Verificar estructura
        if 'timestamp' not in existing.columns:
            logging.warning("Columna 'timestamp' no encontrada. Renombrando...")
            if 'Datetime' in existing.columns:
                existing = existing.rename(columns={'Datetime': 'timestamp'})
            elif 'Date' in existing.columns:
                existing = existing.rename(columns={'Date': 'timestamp'})
        
        existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        
        # Determinar última fecha disponible
        last_date = existing['timestamp'].max()
        start_update = (last_date + timedelta(minutes=1)).strftime("%Y-%m-%d")
        
        # Descargar nuevos datos
        new_data = yf.download(
            symbol,
            start=start_update,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        if new_data.empty:
            logging.info("✅ No hay nuevos datos disponibles")
            return True
        
        # Procesar nuevos datos
        new_data = limpiar_y_renombrar(new_data.reset_index())
        
        if 'timestamp' in new_data.columns and hasattr(new_data['timestamp'], 'dt'):
            if new_data['timestamp'].dt.tz is not None:
                new_data['timestamp'] = new_data['timestamp'].dt.tz_convert(None)
        
        # Combinar con datos existentes
        combined = pd.concat([existing, new_data])
        combined = combined.drop_duplicates('timestamp').sort_values('timestamp')
        
        # Guardar
        combined.to_csv(csv_path, index=False)
        logging.info(f"✅ Datos actualizados: +{len(new_data)} filas")
        return True
    
    except Exception as e:
        logging.error(f"❌ Error en actualización: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if not actualizar_datos():
        logging.warning("⚠️ Falló la actualización, intentando descarga completa...")
        descargar_datos()