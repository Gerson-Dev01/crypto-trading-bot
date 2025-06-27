import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """Carga y normaliza los datos con manejo robusto de valores extremos"""
    print("🔍 Cargando y normalizando datos...")
    data = pd.read_csv("data/btc_with_features.csv")
    
    # 1. Limpieza inicial de valores infinitos/nulos
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 2. Identificar columnas numéricas
    non_features = ['signal', 'future_return', 'Datetime']
    features = [col for col in data.columns if col not in non_features and data[col].dtype in ['float64', 'int64']]
    
    # 3. Manejo de valores extremos antes de escalar
    for col in features:
        # Calcular percentiles para identificar outliers
        p1, p99 = np.percentile(data[col].dropna(), [1, 99])
        data[col] = np.clip(data[col], p1, p99)
        
        # Rellenar cualquier NaN restante con la mediana
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
    
    # 4. Normalización robusta
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    print(f"✅ Datos normalizados. Forma final: {data.shape}")
    return data

def prepare_features_labels(data):
    """Prepara features y balancea clases"""
    X = data.drop(columns=['signal', 'future_return', 'Datetime'], errors='ignore')
    y = data['signal']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Balanceo de clases con SMOTE
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y_encoded)
    
    print("\n📊 Distribución de clases después de SMOTE:")
    print(pd.Series(y_res).value_counts(normalize=True))
    
    return X_res, y_res, le

def optimize_model(X_train, y_train):
    """Optimización de hiperparámetros"""
    print("\n🔎 Optimizando hiperparámetros...")
    
    param_grid = {
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1]
    }
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric=['mlogloss', 'merror'],
        n_estimators=500,
        tree_method='hist',
        early_stopping_rounds=20
    )
    
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        verbose=2
    )
    
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_final_model(X, y, le):
    """Entrenamiento final con validación"""
    print("\n🚀 Entrenando modelo optimizado...")
    
    # División temporal
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Modelo optimizado
    model = optimize_model(X_train, y_train)
    
    # Entrenamiento con early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Evaluación detallada
    print("\n📊 Evaluación Final Optimizada:")
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=["Vender", "Mantener", "Comprar"]))
    
    # Visualización
    plot_confusion_matrix(y_val, y_pred)
    plot_feature_importance(model, X.columns)
    
    return model

def plot_confusion_matrix(y_true, y_pred):
    """Matriz de confusión interactiva"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=["Vender", "Mantener", "Comprar"],
               yticklabels=["Vender", "Mantener", "Comprar"])
    plt.title('Matriz de Confusión')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """Importancia de features mejorada"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(10,8))
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(f'Top {top_n} Features más Importantes')
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.show()

def main():
    try:
        # 1. Carga y preprocesamiento
        data = load_and_preprocess_data()
        
        # 2. Preparación de features y balanceo
        X, y, le = prepare_features_labels(data)
        
        # 3. Entrenamiento optimizado
        final_model = train_final_model(X, y, le)
        
        # 4. Guardar modelo
        joblib.dump({'model': final_model, 'label_encoder': le}, 
                   'models/optimized_trading_model.pkl')
        
        print("\n✅ Entrenamiento optimizado completado!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()