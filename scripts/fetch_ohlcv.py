import yfinance as yf
import sqlite3
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.symbols import ALL_SYMBOLS

db_path = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'

# Create database directory if it doesn't exist
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Connect to SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table with unique constraint
cursor.execute('''
CREATE TABLE IF NOT EXISTS candles_raw (
    timestamp TEXT,
    symbol TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER
)
''')
# Add unique index if not exists
cursor.execute('''
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_symbol_timestamp
ON candles_raw(symbol, timestamp)
''')
conn.commit()

for symbol in ALL_SYMBOLS:
    try:
        data = yf.download(symbol, period='3y', interval='1d', progress=False)
        if data.empty:
            logging.warning(f'No data for {symbol}')
            continue
        rows = 0
        for index, row in data.iterrows():
            timestamp = index.strftime('%Y-%m-%d')
            cursor.execute('''
            INSERT OR REPLACE INTO candles_raw (timestamp, symbol, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                symbol,
                row['Open'].item(),
                row['High'].item(),
                row['Low'].item(),
                row['Close'].item(),
                int(row['Volume'].item())
            ))
            rows += 1
        conn.commit()
        logging.info(f'{symbol}: {rows} rows')
    except Exception as e:
        logging.error(f'Error with {symbol}: {str(e)}')

conn.close()
logging.info('Data fetching completed.')
