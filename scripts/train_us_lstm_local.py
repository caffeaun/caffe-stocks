"""
Train US LSTM model locally (no Modal needed for this dataset size).
Saves to ~/projects/caffe-stocks/models/lstm/us_trading_model.h5
"""
import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import io

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/us_candles.db'
MODEL_DIR = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm'
MODEL_PATH = os.path.join(MODEL_DIR, 'us_trading_model.h5')
SEQ_LEN = 20
FEATURE_COLS = ['rsi', 'macd', 'atr', 'volume_ratio']


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def prepare_training_data():
    from sklearn.preprocessing import MinMaxScaler

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        'SELECT date, symbol, close, rsi, macd, atr, volume_ratio '
        'FROM us_candles WHERE rsi IS NOT NULL AND macd IS NOT NULL AND atr IS NOT NULL '
        'ORDER BY symbol, date',
        conn
    )
    conn.close()

    X_seqs, y_seqs = [], []

    for symbol, group in df.groupby('symbol'):
        group = group.reset_index(drop=True)
        group['label'] = (group['close'].shift(-1) > group['close']).astype(float)
        group = group.dropna(subset=['label'])

        features = group[FEATURE_COLS].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)

        scaler = MinMaxScaler()
        features = scaler.fit_transform(features).astype(np.float32)

        for i in range(len(features) - SEQ_LEN):
            X_seqs.append(features[i:i + SEQ_LEN])
            y_seqs.append(labels[i + SEQ_LEN])

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_seqs, dtype=np.float32)
    print(f'Prepared {len(X)} training sequences from {df["symbol"].nunique()} symbols')
    return X, y


def train(X, y):
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device} | train={len(X_train)}, val={len(X_val)}')

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(device)

    X_train_t = to_tensor(X_train)
    y_train_t = to_tensor(y_train).unsqueeze(1)
    X_val_t = to_tensor(X_val)
    y_val_t = to_tensor(y_val).unsqueeze(1)

    model = LSTMModel(input_size=4, hidden_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_state = None
    batch_size = 128

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(X_train_t), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X_train_t), batch_size):
            idx_b = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[idx_b]), y_train_t[idx_b])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/100 | Loss: {epoch_loss/n_batches:.4f} | Val Loss: {val_loss:.4f}')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_val_t)) >= 0.5
        acc = (preds == y_val_t.bool()).float().mean().item()
    print(f'US LSTM validation accuracy: {acc:.2%}')
    return model, best_state, acc


def main():
    print('Preparing US training data...')
    X, y = prepare_training_data()

    print('Training US LSTM model locally...')
    model, best_state, acc = train(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    buf = io.BytesIO()
    torch.save({
        'state_dict': best_state,
        'input_size': 4,
        'hidden_size': 64,
        'seq_len': SEQ_LEN,
        'feature_cols': FEATURE_COLS,
        'val_accuracy': acc,
        'market': 'US',
    }, buf)
    model_bytes = buf.getvalue()

    with open(MODEL_PATH, 'wb') as f:
        f.write(model_bytes)

    print(f'US LSTM model saved to {MODEL_PATH} ({len(model_bytes)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
