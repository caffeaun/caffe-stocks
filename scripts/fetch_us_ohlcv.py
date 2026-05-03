import yfinance as yf
import sqlite3
import pandas as pd

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/us_candles.db'
SYMBOLS = ['PLTR', 'AMD', 'COIN', 'SOFI', 'SMCI', 'MARA', 'CRWD', 'RIVN']


def fetch_data():
    all_data = []
    for symbol in SYMBOLS:
        print(f'Fetching {symbol}...')
        data = yf.download(symbol, period='3y', interval='1d', auto_adjust=True, progress=False)
        if data.empty:
            print(f'  No data for {symbol}')
            continue

        # Newer yfinance returns MultiIndex columns (e.g. ('Close', 'PLTR')); flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data.columns = [c.lower() for c in data.columns]
        data['symbol'] = symbol

        # Keep only standard OHLCV + symbol
        keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol'] if c in data.columns]
        all_data.append(data[keep])

    if not all_data:
        raise ValueError('No data fetched for any symbol')

    return pd.concat(all_data, ignore_index=True)


def store_data(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('us_candles_raw', conn, if_exists='replace', index=False)
    conn.close()


def main():
    print('Fetching US market data...')
    df = fetch_data()
    symbols_fetched = df['symbol'].nunique()
    print(f'Fetched {len(df)} rows for {symbols_fetched} symbols')
    store_data(df)
    print(f'Data stored in {DB_PATH} (table: us_candles_raw)')


if __name__ == '__main__':
    main()
