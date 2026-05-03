import os
import sys
import requests
import time
import sqlite3
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from scrape_alert import alert_scrape_error

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/us_candles.db'
US_SYMBOLS = ['PLTR', 'AMD', 'COIN', 'SOFI', 'SMCI', 'MARA', 'CRWD', 'RIVN']


def fetch_alpha_vantage(symbol):
    api_key = os.getenv('ALPHA_VANTAGE_KEY')
    if not api_key:
        raise ValueError('ALPHA_VANTAGE_KEY environment variable not set')

    url = (
        f'https://www.alphavantage.co/query'
        f'?function=TIME_SERIES_DAILY&symbol={symbol}'
        f'&apikey={api_key}&outputsize=full'
    )
    response = requests.get(url)
    data = response.json()

    if 'Time Series (Daily)' not in data:
        msg = data.get('Note', data.get('Information', data.get('Error Message', 'No data')))
        print(f'Error fetching {symbol}: {msg}')
        alert_scrape_error("fetch_alpha_vantage.py", f"{symbol}: {msg}")
        return None

    time.sleep(12)  # 5 requests/min rate limit
    return data['Time Series (Daily)']


def store_data(symbol, data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alpha_vantage_raw (
            timestamp TEXT,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT DEFAULT 'alpha_vantage',
            PRIMARY KEY (timestamp, symbol)
        )
    ''')

    # us_candles_raw is shared with fetch_us_ohlcv.py and read by compute_us_indicators.py
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS us_candles_raw (
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            symbol TEXT,
            source TEXT DEFAULT 'alpha_vantage',
            PRIMARY KEY (date, symbol)
        )
    ''')

    rows_inserted = 0
    for date, values in data.items():
        open_ = float(values['1. open'])
        high = float(values['2. high'])
        low = float(values['3. low'])
        close = float(values['4. close'])
        volume = int(values['5. volume'])

        cursor.execute('''
            INSERT OR REPLACE INTO alpha_vantage_raw
                (timestamp, symbol, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, symbol, open_, high, low, close, volume, 'alpha_vantage'))

        cursor.execute('''
            INSERT OR REPLACE INTO us_candles_raw
                (date, open, high, low, close, volume, symbol, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, open_, high, low, close, volume, symbol, 'alpha_vantage'))

        rows_inserted += 1

    conn.commit()
    conn.close()
    print(f'  {symbol}: stored {rows_inserted} rows')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', help='Single stock symbol (e.g., PLTR)')
    parser.add_argument('--all', action='store_true', dest='all_symbols',
                        help='Fetch all US symbols')
    parser.add_argument('--dry-run', action='store_true', help='Print sample data without fetching')
    args = parser.parse_args()

    if not args.symbol and not args.all_symbols:
        parser.error('Provide --symbol SYMBOL or --all')

    symbols = US_SYMBOLS if args.all_symbols else [args.symbol]

    if args.dry_run:
        for sym in symbols:
            print(f'Sample Alpha Vantage data for {sym} (dry-run):')
            print(f'  timestamp: 2026-02-27')
            print(f'  open: 123.45, high: 125.67, low: 122.34, close: 124.56, volume: 1000000')
            print(f'  source: alpha_vantage')
        return

    print(f'Fetching {len(symbols)} symbol(s) from Alpha Vantage...')
    success = 0
    for sym in symbols:
        print(f'Fetching {sym}...')
        data = fetch_alpha_vantage(sym)
        if data:
            store_data(sym, data)
            success += 1
        else:
            print(f'  {sym}: skipped (no data)')

    print(f'Done: {success}/{len(symbols)} symbols stored in {DB_PATH}')


if __name__ == '__main__':
    main()
