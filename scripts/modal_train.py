import modal
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, time
import sys

# Market hours check (SET: 10:00-16:30 ICT)
current_time = datetime.now().time()
set_start = time(10, 0)
set_end = time(16, 30)
if set_start <= current_time <= set_end:
    print('⚠️ Market hours detected. Exiting to avoid trading during market hours.')
    sys.exit(0)

# Budget check
budget_file = '/home/kanoonth-ai/projects/caffe-stocks/data/modal_budget.json'
with open(budget_file, 'r') as f:
    budget = json.load(f)
if budget['spent_thb'] >= 1000:
    print('⚠️ Modal budget exceeded (฿1000 threshold). Exiting.')
    sys.exit(0)

# Modal app definition
app = modal.App('lstm-trainer')

@app.function(gpu='a10g', image=modal.Image.debian_slim().pip_install('torch', 'h5py', 'pandas', 'scikit-learn', 'joblib'))
def train_lstm():
    import torch
    import torch.nn as nn
    import joblib

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1])

    # Load training data
    df = pd.read_csv('/home/kanoonth-ai/projects/caffe-stocks/data/training_data.csv')

    # Process data (using lstm_trainer.py structure)
    features = df[['rsi', 'macd', 'atr', 'volume_ratio']].values
    
    # Split into sequences (20-day lookback)
    seq_len = 20
    X_seq = []
    y_seq = []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i:i+seq_len])
        y_seq.append(df['label'].iloc[i+seq_len])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Convert to tensors
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)
    
    # Initialize model
    input_size = X_seq.shape[2]
    model = LSTMModel(input_size, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Train for 50 epochs
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_tensor), y_tensor).item()
            print(f'Epoch {epoch+1}/50 | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), '/home/kanoonth-ai/projects/caffe-stocks/models/lstm/trading_model.h5')
    print('✅ LSTM model trained and saved successfully')

if __name__ == '__main__':
    app.deploy()
    train_lstm.remote()