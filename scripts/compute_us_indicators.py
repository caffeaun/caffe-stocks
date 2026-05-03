import sqlite3
import pandas as pd
import pandas_ta as ta

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/us_candles.db'


def compute_indicators():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM us_candles_raw', conn)
    conn.close()

    results = []
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('date').copy()

        group['rsi'] = ta.rsi(group['close'], length=14)

        macd_df = ta.macd(group['close'], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            group['macd'] = macd_df.iloc[:, 0].values
        else:
            group['macd'] = None

        group['atr'] = ta.atr(group['high'], group['low'], group['close'], length=14)

        mean_vol = group['volume'].mean()
        group['volume_ratio'] = group['volume'] / mean_vol if mean_vol > 0 else 1.0

        results.append(group)

    result_df = pd.concat(results, ignore_index=True)
    result_df = result_df.dropna(subset=['rsi', 'macd', 'atr', 'volume_ratio'])

    conn = sqlite3.connect(DB_PATH)
    result_df.to_sql('us_candles', conn, if_exists='replace', index=False)
    conn.close()


def main():
    print('Computing indicators for US market data...')
    compute_indicators()
    print('Indicators computed and stored in us_candles table')


if __name__ == '__main__':
    main()
