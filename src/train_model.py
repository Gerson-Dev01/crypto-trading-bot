import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Configurar dispositivo (GPU si está disponible)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 por bidireccional
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_size)
    
    def forward(self, x):
        # Capa LSTM
        x, _ = self.lstm(x)
        # Solo tomar el último paso de tiempo
        x = x[:, -1, :]
        x = self.dropout(x)
        # Capas completamente conectadas
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)

def train_model(processed_data, epochs=100, batch_size=64, learning_rate=0.001):
    # 1. Preparar datos
    X_train, y_train = processed_data['train']
    X_test, y_test = processed_data['test']
    
    # Convertir a tensores PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Crear DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Inicializar modelo
    model = LSTMModel(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. Entrenamiento
    best_loss = float('inf')
    train_history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Pérdida promedio por batch
        avg_loss = epoch_loss / len(train_loader)
        train_history['loss'].append(avg_loss)
        
        # Validación
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor).item()
            train_history['val_loss'].append(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"Época {epoch+1}: Nuevo mejor modelo - Val Loss: {val_loss:.6f}")
        
        print(f"Época {epoch+1}/{epochs} - Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping si no mejora en 10 épocas
        if epoch > 10 and val_loss > min(train_history['val_loss'][-10:]):
            print("Early stopping activado")
            break
    
    # 4. Evaluar con test set
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
    
    # Convertir a numpy para métricas
    y_pred = test_outputs.cpu().numpy().flatten()
    
    # Métricas adicionales
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Resultados Finales")
    print("="*50)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # 5. Guardar recursos finales
    torch.save(model.state_dict(), "models/final_model.pth")
    joblib.dump(processed_data['scaler'], "models/scaler.pkl")
    print("Modelo y escalador guardados en /models")
    
    # 6. Visualización de resultados
    plot_results(train_history, y_test, y_pred)
    
    return model, train_history

def plot_results(history, y_true, y_pred):
    """Visualiza métricas de entrenamiento y predicciones"""
    plt.figure(figsize=(15, 10))
    
    # Pérdidas de entrenamiento
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Predicciones vs Reales
    plt.subplot(2, 1, 2)
    plt.plot(y_true[:200], label='True Prices', alpha=0.7)
    plt.plot(y_pred[:200], label='Predicted Prices', alpha=0.7)
    plt.title('True vs Predicted Prices (First 200 Samples)')
    plt.ylabel('Price')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/training_results.png")
    plt.show()

if __name__ == "__main__":
    # Cargar datos preprocesados
    from preprocess_data import preprocess_data
    
    print("Cargando y preprocesando datos...")
    processed_data = preprocess_data('data/btc.csv')
    
    print("\nIniciando entrenamiento del modelo...")
    start_time = time.time()
    model, history = train_model(processed_data, epochs=100, batch_size=64)
    print(f"\nEntrenamiento completado en {time.time() - start_time:.2f} segundos")