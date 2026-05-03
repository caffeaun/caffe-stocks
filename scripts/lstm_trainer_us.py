import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Database paths
DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/us_candles.db'
MODEL_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/us_trading_model.h5'
SCALER_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/us_scaler.pkl'

# Model parameters
SEQ_LEN = 20  # 20-day lookback
HIDDEN_UNITS = 64
DROPOUT_RATE = 0.3
EPOCHS = 100
BATCH_SIZE = 32


def load_data():
    """Load US candle data and create sequences"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT open, high, low, close, volume FROM us_candles ORDER BY date', conn)
    conn.close()

    # Create features (use close price and volume)
    features = df[['close', 'volume']].values
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - SEQ_LEN):
        X.append(scaled_features[i:i+SEQ_LEN])
        # Binary label: 1 if price rises 15% then drops 3% within 10 days
        label = 1 if (df['close'].iloc[i+SEQ_LEN+10] / df['close'].iloc[i+SEQ_LEN] - 1) > 0.15 and (df['close'].iloc[i+SEQ_LEN] / df['close'].iloc[i+SEQ_LEN+10] - 1) > 0.03 else 0
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/validation/test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def build_model():
    """Build US LSTM model"""
    model = Sequential([
        LSTM(HIDDEN_UNITS, return_sequences=True, input_shape=(SEQ_LEN, 2)),
        Dropout(DROPOUT_RATE),
        LSTM(HIDDEN_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    print('Loading US candle data...')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_data()
    
    print('Building US LSTM model...')
    model = build_model()
    
    print('Training US model...')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'US Model Test Accuracy: {test_acc:.2%}')
    
    # Save model and scaler
    model.save(MODEL_PATH)
    np.save(SCALER_PATH, scaler)
    print(f'Model saved to {MODEL_PATH}')
    print(f'Scaler saved to {SCALER_PATH}')

if __name__ == '__main__':
    main()