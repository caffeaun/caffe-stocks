import os
import sqlite3
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn

MODEL_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/trading_model.h5'
SCALER_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/scaler.pkl'
RESULTS_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/last_retrain.md'
DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.sigmoid(self.fc(out)).squeeze(-1)


def load_model_and_scaler():
    with h5py.File(MODEL_PATH, 'r') as f:
        input_size = int(f.attrs['input_size'])
        hidden_size = int(f.attrs['hidden_size'])
        num_layers = int(f.attrs['num_layers'])
        seq_len = int(f.attrs['seq_len'])
        features = f.attrs['features'].split(',')

        model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        state_dict = {}
        for key in f['weights']:
            state_dict[key] = torch.tensor(f['weights'][key][:])
        model.load_state_dict(state_dict)
        model.eval()

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler, seq_len, features


def load_candles_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM candles ORDER BY timestamp', conn)
    conn.close()
    return df


def compute_label(data, seq_end_idx):
    """Label: 1 if +15% hit before -3% within 10 trading days after seq_end_idx."""
    entry_price = data.iloc[seq_end_idx - 1]['close']
    label = 0
    for j in range(seq_end_idx, seq_end_idx + 10):
        if j >= len(data):
            break
        close_price = data.iloc[j]['close']
        if close_price >= 1.15 * entry_price:
            label = 1
            break
        elif close_price <= 0.97 * entry_price:
            label = 0
            break
    return label


def compute_forward_return(data, seq_end_idx):
    """Compute best % gain (if win) or worst % loss (if loss) within 10 days."""
    entry_price = data.iloc[seq_end_idx - 1]['close']
    best_pct = 0.0
    for j in range(seq_end_idx, seq_end_idx + 10):
        if j >= len(data):
            break
        close_price = data.iloc[j]['close']
        pct = (close_price - entry_price) / entry_price * 100
        if close_price >= 1.15 * entry_price:
            return pct, 1
        elif close_price <= 0.97 * entry_price:
            return pct, 0
        best_pct = max(best_pct, pct)
    return best_pct, 0


def main():
    model, scaler, seq_len, features = load_model_and_scaler()
    df = load_candles_data()

    if df.empty:
        print('No data available for validation')
        return

    # Build sequences (same as training)
    X = []
    labels = []
    seq_end_indices = []

    for i in range(len(df) - seq_len - 10):
        seq = df[features].iloc[i:i + seq_len].values
        X.append(seq)
        lbl = compute_label(df, i + seq_len)
        labels.append(lbl)
        seq_end_indices.append(i + seq_len)

    X = np.array(X, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Test split: last 15% of sequences
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_test = X[train_size + val_size:]
    y_test = labels[train_size + val_size:]
    test_seq_ends = seq_end_indices[train_size + val_size:]

    if len(X_test) == 0:
        print('No test samples available')
        return

    # Scale test sequences using fitted scaler
    n, s, f = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, f)).reshape(n, s, f)

    # Run predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        preds = model(X_tensor).numpy()

    # Compute trading metrics
    buy_signals = preds > 0.5
    n_buys = buy_signals.sum()

    wins = []
    losses = []

    for i in range(len(preds)):
        if buy_signals[i]:
            pct, hit = compute_forward_return(df, test_seq_ends[i])
            if hit:
                wins.append(pct)
            else:
                losses.append(pct)

    win_rate = len(wins) / n_buys * 100 if n_buys > 0 else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    max_dd = abs(min(losses)) if losses else 0.0

    print('Walk-Forward Validation Results:')
    print(f'  Test samples:  {len(X_test)}')
    print(f'  Buy signals:   {n_buys}')
    print(f'  Win rate:      {win_rate:.1f}%')
    print(f'  Avg win:       {avg_win:.2f}%')
    print(f'  Max drawdown:  {max_dd:.2f}%')

    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(RESULTS_PATH, 'w') as f:
        f.write(f'# Walk-Forward Validation\n')
        f.write(f'- Date: {timestamp}\n')
        f.write(f'- Test samples: {len(X_test)}\n')
        f.write(f'- Buy signals: {n_buys}\n')
        f.write(f'- Win rate: {win_rate:.1f}%\n')
        f.write(f'- Avg win: {avg_win:.2f}%\n')
        f.write(f'- Max drawdown: {max_dd:.2f}%\n')

    print(f'Results saved to {RESULTS_PATH}')


if __name__ == '__main__':
    main()
