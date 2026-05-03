import sys
import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

import json
import os

sys.path.insert(0, '/home/kanoonth-ai/projects/caffe-stocks')
try:
    from models.lstm_trader import LSTMTrader
    _lstm_available = True
except ImportError as e:
    import logging
    logging.warning(f'Could not import LSTMTrader: {e}')
    _lstm_available = False

status_file = '/home/kanoonth-ai/projects/caffe-stocks/data/system-status.json'
if os.path.exists(status_file):
    with open(status_file, 'r') as f:
        status_data = json.load(f)
        if status_data.get('status') != 'active':
            print(f'Skipping signal generation: system status is {status_data.get('status')}')
            import sys
            sys.exit(0)

# Database path
DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'


def generate_signals():
    """Generate trading signals based on breakout filter.
    Only returns signals for the latest date in the data."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM candles ORDER BY symbol, date', conn)
    conn.close()

    # Filter out index data (not tradeable)
    df = df[~df['symbol'].str.startswith('^')]
    latest_date = df['date'].max()

    # LSTM scoring — pass full history so each symbol has 20+ rows for sequence
    scored_df = pd.DataFrame()
    if _lstm_available:
        try:
            trader = LSTMTrader()
            scored_df = trader.score_signals(df)
            # Keep only latest date scores
            if not scored_df.empty:
                scored_df = scored_df[scored_df['date'] == latest_date]
        except Exception as e:
            print(f'LSTM scoring failed: {e}', file=__import__('sys').stderr)

    if scored_df.empty or 'lstm' not in scored_df.columns:
        return []

    # Apply critical filters on scored results
    df_latest = scored_df[
        (scored_df['atr'] / scored_df['close'] > 0.03) &
        (scored_df['volume_ratio'] > 2.0)
    ].copy()

    # RSI filter
    if 'rsi' in df_latest.columns:
        df_latest = df_latest[(df_latest['rsi'] >= 30) & (df_latest['rsi'] <= 65)]

    signals = []
    for _, row in df_latest.iterrows():
        signals.append({
            'symbol': row['symbol'],
            'date': row['date'],
            'price': row['close'],
            'rsi': round(row.get('rsi', 0), 1),
            'lstm_score': float(row['lstm']),
        })

    return signals


def main():
    signals = generate_signals()

    # Get latest date from DB for context
    conn = sqlite3.connect(DB_PATH)
    latest_date = pd.read_sql_query('SELECT MAX(date) as d FROM candles', conn).iloc[0]['d']
    symbols_count = pd.read_sql_query('SELECT COUNT(DISTINCT symbol) as n FROM candles WHERE date = ?',
                                       conn, params=[latest_date]).iloc[0]['n']
    conn.close()

    if not signals:
        print(f"📊 *Signal Scan — {latest_date}*\n"
              f"Scanned {symbols_count} symbols\n"
              f"No signals today")
    else:
        lines = [f"📊 *Signal Scan — {latest_date}*",
                 f"Scanned {symbols_count} symbols, found {len(signals)} signal(s)\n"]
        for s in signals:
            lines.append(f"🟢 *{s['symbol'].replace('.BK','')}* — ฿{s['price']:.2f}")
            lines.append(f"   LSTM: {s['lstm_score']:.0%} | RSI: {s['rsi']}")
            lines.append("")
        print('\n'.join(lines))

if __name__ == '__main__':
    main()