import os
import sqlite3
import numpy as np
import pandas as pd
import pickle
import h5py
import torch

try:
    from models.lstm_model import LSTMModel
    from models.feature_eng import prepare_data
except ModuleNotFoundError:
    from lstm_model import LSTMModel
    from feature_eng import prepare_data

# Configuration
MIN_LSTM = 0.6
MODEL_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/trading_model.h5'
SCALER_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/scaler.pkl'
LOOKBACK = 20


class LSTMTrader:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.use_lstm = False
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            print('Warning: LSTM model or scaler missing. Skipping LSTM filtering.', file=__import__('sys').stderr)
            return

        try:
            # Load model config and weights from HDF5
            with h5py.File(MODEL_PATH, 'r') as f:
                input_size = int(f.attrs['input_size'])
                hidden_size = int(f.attrs.get('hidden_size', 64))
                num_layers = int(f.attrs.get('num_layers', 2))
                dropout = float(f.attrs.get('dropout', 0.3))
                use_attention = bool(int(f.attrs.get('use_attention', 0)))
                self.features = f.attrs['features'].split(',')

                self.model = LSTMModel(input_size, hidden_size, num_layers, dropout,
                                       use_attention=use_attention)
                state_dict = {}
                for key in f['weights']:
                    state_dict[key] = torch.tensor(np.array(f['weights'][key]))
                self.model.load_state_dict(state_dict)

            self.model.eval()

            # Load scaler
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)

            self.use_lstm = True
        except Exception as e:
            import logging
            logging.warning(f'Could not load LSTM model: {e}')

    def score_signals(self, df):
        # Full feature pipeline — same as training, so features never diverge
        df, _ = prepare_data(df, features=self.features)

        results = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy().sort_values('date')

            if self.use_lstm and self.features:
                symbol_df = symbol_df.dropna(subset=self.features)

                features_data = symbol_df[self.features].values
                if len(features_data) >= LOOKBACK:
                    sequences = [features_data[i:i + LOOKBACK]
                                 for i in range(len(features_data) - LOOKBACK + 1)]
                    X = np.array(sequences, dtype=np.float32)
                    X_scaled = self.scaler.transform(
                        X.reshape(-1, X.shape[2])).reshape(X.shape)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                    with torch.no_grad():
                        lstm_scores = self.model(X_tensor).numpy().flatten()

                    # Align scores with dataframe (first LOOKBACK-1 rows have no score)
                    score_col = [None] * (LOOKBACK - 1) + list(lstm_scores)
                    symbol_df = symbol_df.iloc[:len(score_col)].copy()
                    symbol_df['lstm'] = score_col
                    symbol_df = symbol_df[symbol_df['lstm'].notna()]
                    symbol_df = symbol_df[symbol_df['lstm'] > MIN_LSTM]
                else:
                    import sys as _sys
                    print(f'Warning: Not enough data for LSTM on {symbol} '
                          f'(have {len(features_data)}, need {LOOKBACK})', file=_sys.stderr)

            symbol_df = symbol_df[symbol_df['volume_ratio'] > 1.5]
            symbol_df = symbol_df[symbol_df['atr'] > 0.03]
            results.append(symbol_df)

        if results:
            return pd.concat(results)
        return pd.DataFrame()


if __name__ == '__main__':
    conn = sqlite3.connect('/home/kanoonth-ai/projects/caffe-stocks/data/candles.db')
    df = pd.read_sql_query('SELECT * FROM candles ORDER BY date', conn)
    conn.close()

    trader = LSTMTrader()
    signals = trader.score_signals(df)
    print(f'\nTotal signals: {len(signals)}')
    for _, row in signals.tail(20).iterrows():
        print(f"  {row['symbol']} | {row['date']} | "
              f"Price: {row['close']:.2f} | LSTM: {row.get('lstm', 'N/A'):.3f}")
