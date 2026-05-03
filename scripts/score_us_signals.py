import sqlite3
import sys
import pandas as pd
import json
import os
from datetime import datetime

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/us_candles.db'
MODEL_PATH = '/home/kanoonth-ai/projects/caffe-stocks/models/us_trading_model.h5'
OUTPUT_FILE = '/home/kanoonth-ai/projects/caffe-stocks/data/us_latest_signal.json'


def main():
    # Check for LSTM model
    model_exists = os.path.exists(MODEL_PATH)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='us_candles'")
    if not cursor.fetchone():
        print('No data in us_candles table. Run compute_us_indicators.py first.')
        return

    # Get latest data per symbol
    df = pd.read_sql_query("SELECT * FROM us_candles ORDER BY date DESC, symbol", conn)
    df = df.drop_duplicates(subset=['symbol'], keep='first')

    # Compute 20-day high per symbol for breakout detection
    df_all = pd.read_sql_query("SELECT symbol, date, high FROM us_candles ORDER BY date", conn)
    high_20d = {}
    for sym in df_all['symbol'].unique():
        sym_df = df_all[df_all['symbol'] == sym].sort_values('date')
        if len(sym_df) >= 20:
            high_20d[sym] = sym_df['high'].iloc[-20:].max()

    signals = []
    for _, row in df.iterrows():
        # Apply filters
        atr_pct = row['atr'] / row['close']
        if atr_pct <= 0.04:  # ATR > 4% of close
            continue
        if row['volume_ratio'] <= 2.0:
            continue
        if not (30 <= row['rsi'] <= 65):
            continue
        # Breakout: 20-day high or MACD crossover (same pattern as SET score_signals.py)
        sym_high = high_20d.get(row['symbol'])
        is_20d_breakout = sym_high is not None and row['close'] > sym_high
        if not (is_20d_breakout or row['macd'] > 0):
            continue

        # Generate signal
        signal_id = f"SIG-{datetime.now().strftime('%Y%m%d')}-{row['symbol']}"
        signal = {
            "signal_id": signal_id,
            "symbol": row['symbol'],
            "market": "US",
            "entry_price": row['close'],
            "atr": row['atr'],
            "volume_ratio": row['volume_ratio'],
            "rsi": row['rsi'],
            "breakout": "price > 20-day high" if is_20d_breakout else "MACD crossover",
            "date": row['date']
        }
        signals.append(signal)

    # Output
    if not signals:
        print('No signals')
    else:
        signal_json = json.dumps(signals[0], indent=2)
        if '--dry-run' in sys.argv:
            print(signal_json)
        else:
            with open(OUTPUT_FILE, 'w') as f:
                f.write(signal_json)
            print(f'Signal written to {OUTPUT_FILE}')

if __name__ == '__main__':
    main()