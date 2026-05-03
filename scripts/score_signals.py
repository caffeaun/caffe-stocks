import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib

# Configuration
MIN_LSTM = 0.6
MODEL_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/trading_model.h5'
SCALER_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/scaler.pkl'


class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Load model and scaler if available
use_lstm = False
model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = LSTMModel(input_size=6, hidden_size=64)
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        scaler = joblib.load(SCALER_PATH)
        use_lstm = True
        print('LSTM model loaded for scoring.')
    except Exception as e:
        print(f'Warning: Could not load LSTM model: {str(e)}')
else:
    print('Warning: LSTM model or scaler missing. Skipping LSTM filtering.')

def score_signals():
    # Load candles data
    conn = sqlite3.connect('/home/kanoonth-ai/projects/caffe-stocks/data/candles.db')
    df = pd.read_sql_query('SELECT * FROM candles ORDER BY date', conn)
    conn.close()

    # Process each symbol
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('date')

        # Compute LSTM scores if model available
        if use_lstm:
            features = symbol_df[['rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'volume_ratio']].values
            seq_len = 20

            if len(features) >= seq_len:
                sequences = []
                for i in range(len(features) - seq_len):
                    sequences.append(features[i:i+seq_len])
                X = np.array(sequences)

                # Scale features
                X_scaled = scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

                # Predict using PyTorch
                with torch.no_grad():
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    logits = model(X_tensor)
                    lstm_scores = torch.sigmoid(logits).squeeze().numpy()

                # Assign scores to corresponding rows
                symbol_df['lstm'] = [None] * seq_len + list(lstm_scores)

                # Filter signals
                symbol_df = symbol_df[symbol_df['lstm'] > MIN_LSTM]
            else:
                print(f'Warning: Not enough data for LSTM on {symbol} (need {seq_len} days)')

        # Filter by other criteria (keep existing logic)
        symbol_df = symbol_df[symbol_df['volume_ratio'] > 1.5]
        symbol_df = symbol_df[symbol_df['atr'] > 0.03]

        # Output
        for _, row in symbol_df.iterrows():
            print(f"Signal: {symbol} | {row['date']} | Price: {row['close']} | LSTM: {row.get('lstm', 'N/A')}")

if __name__ == '__main__':
    score_signals()
