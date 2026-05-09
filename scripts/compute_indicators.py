import os
import sys
import sqlite3
import pandas as pd
import pandas_ta_classic as ta
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.symbols import STOCK_SECTOR

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'


def main():
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Delete existing data
    cursor.execute('DELETE FROM candles')
    conn.commit()

    # Read candles_raw
    df = pd.read_sql_query('SELECT * FROM candles_raw ORDER BY timestamp, symbol', conn)

    # --- Compute SET Index market regime features ---
    set_regime = None
    if '^SET.BK' in df['symbol'].values:
        set_df = df[df['symbol'] == '^SET.BK'].copy().sort_values('timestamp')
        set_df['set_ret_5d'] = set_df['close'].pct_change(5)
        set_df['set_ret_10d'] = set_df['close'].pct_change(10)
        set_df['set_sma20'] = set_df['close'].rolling(20).mean()
        set_df['set_above_sma20'] = (set_df['close'] > set_df['set_sma20']).astype(float)
        set_df['set_rsi'] = set_df.ta.rsi(length=14)
        set_df['set_volatility'] = set_df['close'].pct_change().rolling(20).std()
        set_regime = set_df[['timestamp', 'set_ret_5d', 'set_ret_10d',
                             'set_above_sma20', 'set_rsi', 'set_volatility']].dropna()
        print(f"SET Index regime features: {len(set_regime)} rows")

    # Filter out index symbols — only compute stock indicators for tradeable symbols
    stock_symbols = [s for s in df['symbol'].unique() if not s.startswith('^')]

    # Process each symbol separately and collect results
    all_parts = []
    skipped_short = []
    for symbol in stock_symbols:
        symbol_df = df[df['symbol'] == symbol].copy().sort_values('timestamp')
        # Skip thin-history symbols — pandas_ta degrades to weird DataFrame
        # returns when input is shorter than the indicator window. 30 rows
        # comfortably covers RSI(14), MACD(12,26,9), BB(20), ATR(14).
        if len(symbol_df) < 30:
            skipped_short.append((symbol, len(symbol_df)))
            continue

        # Compute indicators using pandas_ta
        symbol_df['rsi'] = symbol_df.ta.rsi(length=14)
        macd_result = symbol_df.ta.macd()
        symbol_df['macd'] = macd_result['MACD_12_26_9']

        # Bollinger Bands
        bb = symbol_df.ta.bbands(length=20, std=2)
        bb_upper_col = [c for c in bb.columns if c.startswith('BBU_')][0]
        bb_lower_col = [c for c in bb.columns if c.startswith('BBL_')][0]
        symbol_df['bb_upper'] = bb[bb_upper_col]
        symbol_df['bb_lower'] = bb[bb_lower_col]

        # ATR
        symbol_df['atr'] = symbol_df.ta.atr(length=14)

        # Volume ratio
        volume_ma = symbol_df['volume'].rolling(20).mean()
        symbol_df['volume_ratio'] = symbol_df['volume'] / volume_ma

        # Derived features for LSTM
        symbol_df['ret_1d'] = symbol_df['close'].pct_change(1)
        symbol_df['ret_5d'] = symbol_df['close'].pct_change(5)
        symbol_df['ret_10d'] = symbol_df['close'].pct_change(10)

        bb_width = symbol_df['bb_upper'] - symbol_df['bb_lower']
        symbol_df['bb_position'] = np.where(bb_width > 0,
                                             (symbol_df['close'] - symbol_df['bb_lower']) / bb_width, 0.5)

        symbol_df['atr_pct'] = symbol_df['atr'] / symbol_df['close']
        symbol_df['vol_momentum'] = symbol_df['volume_ratio'] / symbol_df['volume_ratio'].rolling(5).mean()
        symbol_df['rsi_momentum'] = symbol_df['rsi'] - symbol_df['rsi'].shift(3)

        # Merge SET Index regime features by date
        if set_regime is not None:
            symbol_df = symbol_df.merge(set_regime, on='timestamp', how='left')

        all_parts.append(symbol_df)

    # Combine all symbols
    df_all = pd.concat(all_parts, ignore_index=True)

    # Drop rows with NaN after computation
    df_clean = df_all.dropna(subset=['rsi', 'macd', 'bb_upper', 'bb_lower', 'atr', 'volume_ratio',
                                      'ret_1d', 'ret_5d', 'ret_10d', 'bb_position', 'atr_pct',
                                      'vol_momentum', 'rsi_momentum'])

    # Add date column
    df_clean = df_clean.copy()
    df_clean['date'] = pd.to_datetime(df_clean['timestamp']).dt.date.astype(str)

    # --- Sector-relative features (from our own stock data, 100% coverage) ---
    df_clean['sector'] = df_clean['symbol'].map(STOCK_SECTOR)

    # Per-sector daily aggregates
    sector_daily = df_clean.groupby(['date', 'sector']).agg(
        sector_avg_ret_1d=('ret_1d', 'mean'),
        sector_breadth=('ret_1d', lambda x: (x > 0).mean()),
        sector_avg_volume_ratio=('volume_ratio', 'mean'),
    ).reset_index()
    df_clean = df_clean.merge(sector_daily, on=['date', 'sector'], how='left')
    df_clean['stock_vs_sector_ret'] = df_clean['ret_1d'] - df_clean['sector_avg_ret_1d']
    df_clean.drop(columns=['sector'], inplace=True)

    has_sector_rel = df_clean['sector_avg_ret_1d'].notna().sum()
    print(f"Sector-relative features: {has_sector_rel} rows")

    # --- Market breadth features (daily across all stocks, 100% coverage) ---
    # Need SMA20 per stock for above_sma20 calculation
    sma20_parts = []
    for symbol in stock_symbols:
        sdf = df_clean[df_clean['symbol'] == symbol].copy().sort_values('date')
        sdf['_sma20'] = sdf['close'].rolling(20).mean()
        sdf['_above_sma20'] = (sdf['close'] > sdf['_sma20']).astype(float)
        sdf['_is_20d_high'] = (sdf['close'] >= sdf['high'].rolling(20).max()).astype(float)
        sma20_parts.append(sdf[['symbol', 'date', '_above_sma20', '_is_20d_high']])
    sma20_df = pd.concat(sma20_parts, ignore_index=True)
    df_clean = df_clean.merge(sma20_df, on=['symbol', 'date'], how='left')

    market_daily = df_clean.groupby('date').agg(
        market_breadth_adv=('ret_1d', lambda x: (x > 0).mean()),
        market_breadth_above_sma20=('_above_sma20', 'mean'),
        market_new_highs=('_is_20d_high', 'mean'),
    ).reset_index()
    df_clean = df_clean.merge(market_daily, on='date', how='left')
    df_clean.drop(columns=['_above_sma20', '_is_20d_high'], inplace=True)

    has_breadth = df_clean['market_breadth_adv'].notna().sum()
    print(f"Market breadth features: {has_breadth} rows")

    # --- Merge supplemental data from fetch_market_data.py tables ---

    # Intraday first-hour features
    try:
        intraday = pd.read_sql_query('SELECT * FROM intraday_features', conn)
        if len(intraday) > 0:
            # Compute rolling average for first_hour_vol_ratio normalization
            intraday_cols = ['date', 'symbol', 'first_hour_volume', 'first_hour_range',
                             'first_hour_vol_ratio']
            df_clean = df_clean.merge(intraday[intraday_cols], on=['date', 'symbol'], how='left')
            has_intraday = df_clean['first_hour_volume'].notna().sum()
            print(f"Merged intraday features: {has_intraday} rows")
    except Exception:
        pass  # table may not exist yet

    # Sector index features (compute sector returns from daily snapshots)
    try:
        sector_df = pd.read_sql_query('SELECT * FROM sector_indices ORDER BY date', conn)
        if len(sector_df) > 0:
            df_clean['stock_sector'] = df_clean['symbol'].map(STOCK_SECTOR)
            sector_pivot = sector_df.pivot_table(index='date', columns='sector',
                                                  values='change_pct', aggfunc='last')
            # Vectorized merge: create (date, sector) → change_pct lookup
            sector_lookup = sector_df.drop_duplicates(['date', 'sector'], keep='last').set_index(['date', 'sector'])['change_pct']
            keys = list(zip(df_clean['date'], df_clean['stock_sector']))
            df_clean['sector_change_pct'] = [sector_lookup.get(k, np.nan) for k in keys]
            df_clean.drop(columns=['stock_sector'], inplace=True)
            has_sector = df_clean['sector_change_pct'].notna().sum()
            print(f"Merged sector features: {has_sector} rows")
    except Exception:
        pass  # table may not exist yet

    # Foreign investor flows
    try:
        foreign = pd.read_sql_query('SELECT * FROM foreign_flows', conn)
        if len(foreign) > 0:
            foreign_cols = ['date', 'net_value', 'buy_pct', 'sell_pct']
            foreign_renamed = foreign[foreign_cols].rename(columns={
                'net_value': 'foreign_net', 'buy_pct': 'foreign_buy_pct',
                'sell_pct': 'foreign_sell_pct'
            })
            df_clean = df_clean.merge(foreign_renamed, on='date', how='left')
            has_foreign = df_clean['foreign_net'].notna().sum()
            print(f"Merged foreign flow features: {has_foreign} rows")
    except Exception:
        pass  # table may not exist yet

    # Monthly foreign flows (from SET XLS import)
    try:
        foreign_monthly = pd.read_sql_query('SELECT * FROM foreign_flows_monthly', conn)
        if len(foreign_monthly) > 0:
            df_clean['_month'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m')
            foreign_monthly_renamed = foreign_monthly.rename(columns={
                'net_value': 'foreign_net_monthly',
                'buy_value': 'foreign_buy_monthly',
                'sell_value': 'foreign_sell_monthly',
            })
            df_clean = df_clean.merge(
                foreign_monthly_renamed[['month', 'foreign_net_monthly',
                                          'foreign_buy_monthly', 'foreign_sell_monthly']],
                left_on='_month', right_on='month', how='left'
            )
            df_clean.drop(columns=['_month', 'month'], inplace=True)
            has_foreign_monthly = df_clean['foreign_net_monthly'].notna().sum()
            print(f"Merged monthly foreign flow features: {has_foreign_monthly} rows")
    except Exception:
        pass  # table may not exist yet

    # Historical trading features (PE, PBV, dividend yield)
    try:
        hist = pd.read_sql_query('SELECT * FROM historical_trading', conn)
        if len(hist) > 0:
            hist_cols = ['date', 'symbol', 'pe', 'pbv', 'dividend_yield', 'market_percent_change']
            df_clean = df_clean.merge(hist[hist_cols], on=['date', 'symbol'], how='left')
            has_hist = df_clean['pe'].notna().sum()
            print(f"Merged historical trading features: {has_hist} rows")
    except Exception:
        pass  # table may not exist yet

    if skipped_short:
        print(f"Skipped {len(skipped_short)} symbols with <30 rows: "
              f"{', '.join(f'{s}({n})' for s, n in skipped_short[:10])}"
              f"{' ...' if len(skipped_short) > 10 else ''}")

    # Write to candles table
    df_clean = df_clean.drop_duplicates(subset=['symbol', 'date'], keep='last')
    df_clean.to_sql('candles', conn, if_exists='replace', index=False)
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_symbol_date ON candles(symbol, date)')
    conn.commit()

    print(f"Computed indicators for {len(df_clean)} rows")
    conn.close()

if __name__ == '__main__':
    main()