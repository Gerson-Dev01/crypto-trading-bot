import pandas as pd
import matplotlib.pyplot as plt

def analyze_data():
    # 1. Verificar datos crudos
    raw = pd.read_csv("data/btc.csv")
    print("Datos crudos:")
    print(raw.describe())
    raw.plot(x='timestamp', y='close', title='Precio BTC Crudo')
    plt.savefig("results/raw_prices.png")
    
    # 2. Verificar datos procesados
    processed = pd.read_csv("data/btc_processed.csv")
    print("\nDatos procesados:")
    print(processed.describe())
    
    # 3. Verificar distribuciones
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    features = ['close', 'rsi', 'macd', 'volatility', 'adx', 'volume_ma', 'obv']
    for i, feat in enumerate(features):
        ax = axes[i//3, i%3]
        processed[feat].hist(ax=ax, bins=50)
        ax.set_title(feat)
    plt.tight_layout()
    plt.savefig("results/feature_distributions.png")

if __name__ == "__main__":
    analyze_data()