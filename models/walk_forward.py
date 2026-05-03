import pandas as pd
import sqlite3
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import os
from models.lstm_trader import LSTMTrader

# Connect to database
DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'
conn = sqlite3.connect(DB_PATH)

# Load data for COM7
query = 'SELECT date, close FROM candles WHERE symbol = ? ORDER BY date'
data = pd.read_sql_query(query, conn, params=('COM7',))
conn.close()

data = data.sort_values('date').reset_index(drop=True)

# Prepare features (close price only for simplicity)
data['close'] = data['close'].pct_change().dropna()
data = data.dropna()

# Time-series split (6 folds)
tscv = TimeSeriesSplit(n_splits=6)
results = []

for fold, (train_index, test_index) in enumerate(tscv.split(data)):
    train = data.iloc[train_index]
    test = data.iloc[test_index]

    # Prepare sequences (20-day lookback)
    def create_sequences(df, lookback=20):
        X, y = [], []
        for i in range(len(df) - lookback):
            X.append(df['close'].iloc[i:i+lookback].values)
            y.append(df['close'].iloc[i+lookback])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train)
    X_test, y_test = create_sequences(test)

    # Train model
    model = LSTMTrader()
    model.train(X_train, y_train.reshape(-1, 1))  # Simplified training

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(np.isclose(y_pred, y_test, atol=0.01))
    precision = np.mean((y_pred > 0) & (y_test > 0))  # Simplified
    recall = np.mean((y_pred > 0) & (y_test > 0))

    results.append({
        'fold': fold,
        'train_size': len(train),
        'test_size': len(test),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('/home/kanoonth-ai/projects/caffe-stocks/models/walk_forward_results.csv', index=False)

print('Walk-forward validation completed. Results saved to walk_forward_results.csv')