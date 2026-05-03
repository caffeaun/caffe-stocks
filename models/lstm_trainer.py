import os
import torch
import torch.nn as nn
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'
MODEL_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/trading_model.h5'
SCALER_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/scaler.pkl'


# Compute labels (1 = +15% before -3% in 10 days, 0 = otherwise)
def compute_label(row, df):
    close = row['close']
    date = row['date']
    future = df[df['date'] > date].head(10)
    if len(future) == 0:
        return 0
    if (future['close'].max() / close) >= 1.15 and (future['close'].min() / close) >= 0.97:
        return 1
    return 0


# PyTorch LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    import joblib

    # Load data
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM candles ORDER BY date', conn)
    conn.close()

    df['label'] = df.apply(compute_label, axis=1, df=df)

    # Prepare features
    features = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'volume_ratio']
    X = df[features].values
    y = df['label'].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences (20-day lookback)
    seq_len = 20
    X_seq = []
    y_seq = []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq.append(y[i+seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split data (time-series aware)
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))
    X_train = X_seq[:train_size]
    X_val = X_seq[train_size:train_size+val_size]
    X_test = X_seq[train_size+val_size:]
    y_train = y_seq[:train_size]
    y_val = y_seq[train_size:train_size+val_size]
    y_test = y_seq[train_size+val_size:]

    # Build model
    input_size = X_train.shape[2]
    hidden_size = 64
    model = LSTMModel(input_size, hidden_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    num_epochs = 50

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model trained and saved to {MODEL_PATH}')

    # Test accuracy
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_preds = torch.sigmoid(test_outputs).round().int()
        test_acc = (test_preds.squeeze() == y_test_tensor.squeeze()).float().mean().item()
        print(f'Test Accuracy: {test_acc:.4f}')

    # Save scaler for inference
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f'Scaler saved to {SCALER_PATH}')