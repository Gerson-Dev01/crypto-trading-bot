import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

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

def limpiar_columnas_multiindex(df):
    """Convierte columnas multi-index a nombres simples"""
    if isinstance(df.columns, pd.MultiIndex):
        # Tomar solo el primer nivel de los nombres de columnas
        df.columns = df.columns.get_level_values(0)
    return df

def descargar_datos():
    """Descarga datos garantizando estructura correcta"""
    print(f"⏳ Descargando datos de {symbol}...")
    
    try:
        # Descargar datos con auto_adjust explícito
        data = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            raise ValueError("Datos vacíos recibidos de Yahoo Finance")
        
        # 1. Limpiar estructura de columnas
        data = limpiar_columnas_multiindex(data)
        
        # 2. Resetear índice y renombrar
        data = data.reset_index().rename(columns={'Date': 'Datetime'})
        
        # 3. Eliminar zona horaria si existe (manera moderna)
        if isinstance(data['Datetime'].dtype, pd.DatetimeTZDtype):
            data['Datetime'] = data['Datetime'].dt.tz_localize(None)
        
        # 4. Ordenar y seleccionar columnas necesarias
        columnas_necesarias = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[columnas_necesarias].sort_values('Datetime')
        
        # 5. Guardar
        data.to_csv(csv_path, index=False)
        
        print(f"✅ Datos guardados en {csv_path} ({len(data)} filas)")
        print("Columnas generadas:", data.columns.tolist())
        return True
        
    except Exception as e:
        print(f"❌ Error en descarga: {str(e)}")
        return False

def actualizar_datos():
    """Versión actualizada con manejo robusto de datos"""
    if not os.path.exists(csv_path):
        return descargar_datos()

    try:
        # Leer datos existentes
        existing = pd.read_csv(csv_path)
        
        # Verificar estructura
        if 'Datetime' not in existing.columns:
            existing = existing.rename(columns={'Date': 'Datetime'})
        
        existing['Datetime'] = pd.to_datetime(existing['Datetime'])
        last_date = existing['Datetime'].max()
        
        # Descargar nuevos datos
        new_data = yf.download(
            symbol,
            start=last_date + timedelta(hours=1),
            end=end_date,
            interval=interval,
            auto_adjust=True
        )
        
        if not new_data.empty:
            # Limpiar y preparar nuevos datos
            new_data = limpiar_columnas_multiindex(new_data)
            new_data = new_data.reset_index().rename(columns={'Date': 'Datetime'})
            
            if isinstance(new_data['Datetime'].dtype, pd.DatetimeTZDtype):
                new_data['Datetime'] = new_data['Datetime'].dt.tz_localize(None)
            
            # Combinar y guardar
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates('Datetime')
            combined.to_csv(csv_path, index=False)
            
            print(f"✅ Datos actualizados: +{len(new_data)} filas")
            return True
    
    except Exception as e:
        print(f"❌ Error en actualización: {str(e)}")
    
    return False

if __name__ == "__main__":
    if not actualizar_datos():
        print("⚠️ Falló la actualización, intentando descarga completa...")
        descargar_datos()