import pandas as pd
import os
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'btc_preprocessed.csv')

def backtest_cruce_medias():
    try:
        print("📊 Cargando datos desde:", data_path)
        df = pd.read_csv(data_path, parse_dates=['Datetime'], index_col='Datetime')
        print("✅ Datos cargados. Filas:", len(df))

        if 'Close' not in df.columns:
            raise ValueError("❌ La columna 'Close' no existe en el CSV.")
        if df['Close'].isnull().all():
            raise ValueError("❌ La columna 'Close' está completamente vacía.")

        # Calcular medias móviles
        df['media_corta'] = df['Close'].rolling(window=12).mean()
        print("✅ Media móvil corta calculada")

        if 'media_movil' not in df.columns:
            raise ValueError("❌ La columna 'media_movil' no está en el archivo CSV. Asegúrate de que esté incluida.")
        if df['media_movil'].isnull().all():
            raise ValueError("❌ La columna 'media_movil' está completamente vacía.")

        df['signal'] = 0
        df['cruce_positivo'] = (df['media_corta'] > df['media_movil']) & (df['media_corta'].shift(1) <= df['media_movil'].shift(1))
        df['cruce_negativo'] = (df['media_corta'] < df['media_movil']) & (df['media_corta'].shift(1) >= df['media_movil'].shift(1))
        df.loc[df['cruce_positivo'], 'signal'] = 1
        df.loc[df['cruce_negativo'], 'signal'] = -1
        df.drop(columns=['cruce_positivo', 'cruce_negativo'], inplace=True)
        print("✅ Señales generadas")

        df['retorno'] = df['Close'].pct_change()
        df['estrategia'] = df['signal'].shift(1) * df['retorno']
        df.dropna(subset=['retorno', 'estrategia'], inplace=True)
        print("✅ Retornos calculados")

        # Graficar resultados
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['retorno'].cumsum(), label='Retorno del activo')
        plt.plot(df.index, df['estrategia'].cumsum(), label='Estrategia')
        plt.title('Backtest estrategia cruce de medias')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno acumulado')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print("✅ Gráfico mostrado")

        rendimiento_total = df['estrategia'].cumsum().iloc[-1]
        print(f"📈 Rendimiento total de la estrategia: {rendimiento_total:.2%}")
    
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")

if __name__ == "__main__":
    backtest_cruce_medias()
