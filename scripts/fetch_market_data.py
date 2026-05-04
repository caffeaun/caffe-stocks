#!/usr/bin/env python3
"""Fetch supplemental market data for LSTM model improvement.

Three data sources:
1. Intraday first-hour volume (yfinance 5-min) — detect fake breakouts
2. SET sector indices (SET API) — sector-relative strength
3. Foreign investor flows (SET API) — macro signal

Stores data in candles.db in separate tables, merged by compute_indicators.py.

Run: ~/projects/caffe-stocks/venv/bin/python ~/projects/caffe-stocks/scripts/fetch_market_data.py [--backfill]
"""
import os
import sys
import json
import sqlite3
import argparse
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.symbols import SYMBOLS, STOCK_SECTOR

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'

# All sector indices to fetch
SECTOR_INDICES = list(set(STOCK_SECTOR.values()))

# SET API config
SET_API_BASE = 'https://www.set.or.th'
SET_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.set.or.th/en/market/index/set/overview',
}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def init_tables(conn):
    """Create tables if they don't exist."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS intraday_features (
            date TEXT,
            symbol TEXT,
            first_hour_volume REAL,
            first_hour_range REAL,
            first_hour_vwap REAL,
            full_day_volume REAL,
            first_hour_vol_ratio REAL,
            PRIMARY KEY (date, symbol)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sector_indices (
            date TEXT,
            sector TEXT,
            last REAL,
            change_pct REAL,
            volume REAL,
            value REAL,
            PRIMARY KEY (date, sector)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS foreign_flows (
            date TEXT,
            buy_value REAL,
            sell_value REAL,
            net_value REAL,
            buy_pct REAL,
            sell_pct REAL,
            PRIMARY KEY (date)
        )
    ''')
    conn.commit()


# ============================================================
# 1. INTRADAY FIRST-HOUR VOLUME (yfinance)
# ============================================================

def fetch_intraday(symbols, period='5d', backfill=False):
    """Fetch intraday data, compute first-hour features.
    Uses 1h bars for backfill (2 years depth) and 5-min for recent data."""
    if backfill:
        # 1h bars go back ~2 years vs 60 days for 5-min
        return fetch_intraday_1h(symbols)
    # Recent data: use 5-min for finer granularity
    period = '5d'

    results = []
    for symbol in symbols:
        try:
            data = yf.download(symbol, period=period, interval='5m', progress=False)
            if data.empty:
                print(f'  {symbol}: no intraday data')
                continue

            # Flatten multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.index = data.index.tz_convert('Asia/Bangkok')

            # Group by date
            for date, day_data in data.groupby(data.index.date):
                # First hour: 10:00-11:00 ICT (market opens 10:00)
                first_hour = day_data.between_time('10:00', '11:00')
                if len(first_hour) == 0:
                    continue

                fh_volume = float(first_hour['Volume'].sum())
                full_volume = float(day_data['Volume'].sum())
                fh_high = float(first_hour['High'].max())
                fh_low = float(first_hour['Low'].min())
                fh_range = (fh_high - fh_low) / fh_low if fh_low > 0 else 0

                # VWAP approximation for first hour
                if fh_volume > 0:
                    fh_vwap = float((first_hour['Close'] * first_hour['Volume']).sum() / fh_volume)
                else:
                    fh_vwap = float(first_hour['Close'].mean())

                # First hour volume as fraction of full day
                fh_vol_ratio = fh_volume / full_volume if full_volume > 0 else 0

                results.append({
                    'date': str(date),
                    'symbol': symbol,
                    'first_hour_volume': fh_volume,
                    'first_hour_range': round(fh_range, 6),
                    'first_hour_vwap': round(fh_vwap, 2),
                    'full_day_volume': full_volume,
                    'first_hour_vol_ratio': round(fh_vol_ratio, 4),
                })

            print(f'  {symbol}: {len([r for r in results if r["symbol"] == symbol])} days')
        except Exception as e:
            print(f'  {symbol}: ERROR - {e}')

    return results


def store_intraday(conn, results):
    """Store intraday features, upsert on conflict."""
    for r in results:
        conn.execute('''
            INSERT OR REPLACE INTO intraday_features
            (date, symbol, first_hour_volume, first_hour_range, first_hour_vwap,
             full_day_volume, first_hour_vol_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (r['date'], r['symbol'], r['first_hour_volume'], r['first_hour_range'],
              r['first_hour_vwap'], r['full_day_volume'], r['first_hour_vol_ratio']))
    conn.commit()


# ============================================================
# 2. SECTOR INDICES (SET API)
# ============================================================

def get_set_session():
    """Get a session with WAF cookies from SET website."""
    session = requests.Session()
    session.headers.update(SET_HEADERS)
    # Hit the main page to get WAF cookies
    try:
        resp = session.get(f'{SET_API_BASE}/en/market/index/set/overview', timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f'  Warning: could not init SET session: {e}')
    return session


def fetch_sector_indices(session):
    """Fetch current sector index data from SET API."""
    results = []
    today = datetime.now().strftime('%Y-%m-%d')

    for sector in SECTOR_INDICES:
        try:
            url = f'{SET_API_BASE}/api/set/index/{sector}/info'
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                print(f'  {sector}: HTTP {resp.status_code}')
                continue

            data = resp.json()
            results.append({
                'date': today,
                'sector': sector,
                'last': data.get('last') or data.get('prior', 0),
                'change_pct': data.get('percentChange', 0),
                'volume': data.get('totalVolume', 0),
                'value': data.get('totalValue', 0),
            })
            print(f'  {sector}: {data.get("last", "?")} ({data.get("percentChange", 0):+.2f}%)')
            time.sleep(0.3)  # rate limit
        except Exception as e:
            print(f'  {sector}: ERROR - {e}')

    return results


def fetch_all_sector_indices(session):
    """Fetch all industry group indices with sub-sectors in one call."""
    results = []
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        url = f'{SET_API_BASE}/api/set/index/info/list?type=industry'
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            print(f'  Industry list: HTTP {resp.status_code}')
            return results

        data = resp.json()
        for industry in data:
            # Store industry group
            symbol = industry.get('symbol', '')
            results.append({
                'date': today,
                'sector': symbol,
                'last': industry.get('last') or industry.get('prior', 0),
                'change_pct': industry.get('percentChange', 0),
                'volume': industry.get('totalVolume', 0),
                'value': industry.get('totalValue', 0),
            })

            # Store sub-sectors
            for sub in industry.get('subSector', []):
                sub_symbol = sub.get('symbol', '')
                results.append({
                    'date': today,
                    'sector': sub_symbol,
                    'last': sub.get('last') or sub.get('prior', 0),
                    'change_pct': sub.get('percentChange', 0),
                    'volume': sub.get('totalVolume', 0),
                    'value': sub.get('totalValue', 0),
                })

        print(f'  Fetched {len(results)} sector/industry indices')
    except Exception as e:
        print(f'  Industry list: ERROR - {e}')

    return results


def store_sector_indices(conn, results):
    """Store sector index data."""
    for r in results:
        conn.execute('''
            INSERT OR REPLACE INTO sector_indices
            (date, sector, last, change_pct, volume, value)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (r['date'], r['sector'], r['last'], r['change_pct'],
              r['volume'], r['value']))
    conn.commit()


# ============================================================
# 3. FOREIGN INVESTOR FLOWS (SET API)
# ============================================================

def fetch_foreign_flows(session):
    """Fetch foreign investor net buy/sell from SET API."""
    today = datetime.now().strftime('%Y-%m-%d')

    try:
        url = f'{SET_API_BASE}/api/set/market/SET/investor-type'
        resp = session.get(url, timeout=10)
        if resp.status_code != 200:
            print(f'  Foreign flows: HTTP {resp.status_code}')
            return None

        data = resp.json()

        # Response structure: {investors: [{type: "foreign", buyValue, ...}, ...]}
        foreign = None
        investors = data.get('investors', [])
        for inv in investors:
            if inv.get('type', '').lower() == 'foreign':
                foreign = inv
                break

        if foreign is None:
            print(f'  Foreign flows: no "foreign" type in investors list')
            return None

        # Use asOfDate from response if available
        as_of = data.get('asOfDate', '')
        if as_of:
            today = as_of[:10]  # "2026-03-09T00:00:00+07:00" → "2026-03-09"

        result = {
            'date': today,
            'buy_value': float(foreign.get('buyValue', 0)),
            'sell_value': float(foreign.get('sellValue', 0)),
            'net_value': float(foreign.get('netValue', 0)),
            'buy_pct': float(foreign.get('percentBuyValue', 0)),
            'sell_pct': float(foreign.get('percentSellValue', 0)),
        }
        print(f'  Foreign: net={result["net_value"]:,.0f} '
              f'(buy={result["buy_pct"]:.1f}% sell={result["sell_pct"]:.1f}%)')
        return result

    except Exception as e:
        print(f'  Foreign flows: ERROR - {e}')
        return None


def store_foreign_flows(conn, result):
    """Store foreign flow data."""
    if result is None:
        return
    conn.execute('''
        INSERT OR REPLACE INTO foreign_flows
        (date, buy_value, sell_value, net_value, buy_pct, sell_pct)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (result['date'], result['buy_value'], result['sell_value'],
          result['net_value'], result['buy_pct'], result['sell_pct']))
    conn.commit()


# ============================================================
# 4. INTRADAY VIA 1-HOUR BARS (yfinance, 2-year depth)
# ============================================================

def fetch_intraday_1h(symbols):
    """Fetch 1h bars (2-year depth) and extract first-hour features.
    The 10:00 ICT bar represents the first trading hour on SET."""
    results = []
    for symbol in symbols:
        try:
            data = yf.download(symbol, period='2y', interval='1h', progress=False)
            if data.empty:
                print(f'  {symbol}: no 1h data')
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.index = data.index.tz_convert('Asia/Bangkok')

            for date, day_data in data.groupby(data.index.date):
                # First hour bar: 10:00 ICT
                first_hour = day_data.between_time('10:00', '10:59')
                if len(first_hour) == 0:
                    continue

                fh_volume = float(first_hour['Volume'].sum())
                full_volume = float(day_data['Volume'].sum())
                fh_high = float(first_hour['High'].max())
                fh_low = float(first_hour['Low'].min())
                fh_range = (fh_high - fh_low) / fh_low if fh_low > 0 else 0

                if fh_volume > 0:
                    fh_vwap = float((first_hour['Close'] * first_hour['Volume']).sum() / fh_volume)
                else:
                    fh_vwap = float(first_hour['Close'].mean())

                fh_vol_ratio = fh_volume / full_volume if full_volume > 0 else 0

                results.append({
                    'date': str(date),
                    'symbol': symbol,
                    'first_hour_volume': fh_volume,
                    'first_hour_range': round(fh_range, 6),
                    'first_hour_vwap': round(fh_vwap, 2),
                    'full_day_volume': full_volume,
                    'first_hour_vol_ratio': round(fh_vol_ratio, 4),
                })

            sym_count = len([r for r in results if r['symbol'] == symbol])
            print(f'  {symbol}: {sym_count} days (1h)')
        except Exception as e:
            print(f'  {symbol}: ERROR - {e}')

    return results


# ============================================================
# 5. HISTORICAL TRADING (SET API - PE, PBV, dividend yield)
# ============================================================

def fetch_historical_trading(session, symbols, count=121):
    """Fetch historical trading data from SET API.
    Returns PE, PBV, dividend yield, market cap per stock (up to 121 days)."""
    results = []
    for symbol in symbols:
        sym_short = symbol.replace('.BK', '')
        url = f'{SET_API_BASE}/api/set/stock/{sym_short}/historical-trading?count={count}'
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                print(f'  {sym_short}: HTTP {resp.status_code}')
                continue
            data = resp.json()
            if not isinstance(data, list):
                print(f'  {sym_short}: unexpected response format')
                continue
            for row in data:
                date_str = row.get('date', '')[:10]  # '2026-03-09T00:00:00+07:00' -> '2026-03-09'
                results.append({
                    'date': date_str,
                    'symbol': symbol,
                    'pe': row.get('pe'),
                    'pbv': row.get('pbv'),
                    'dividend_yield': row.get('dividendYield'),
                    'market_cap': row.get('marketCap'),
                    'market_percent_change': row.get('marketPercentChange'),
                })
            print(f'  {sym_short}: {len(data)} days')
            time.sleep(0.3)  # Rate limit
        except Exception as e:
            print(f'  {sym_short}: ERROR - {e}')
    return results


def store_historical_trading(conn, results):
    """Store historical trading data."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS historical_trading (
            date TEXT,
            symbol TEXT,
            pe REAL,
            pbv REAL,
            dividend_yield REAL,
            market_cap REAL,
            market_percent_change REAL,
            PRIMARY KEY (date, symbol)
        )
    ''')
    for r in results:
        conn.execute('''
            INSERT OR REPLACE INTO historical_trading
            (date, symbol, pe, pbv, dividend_yield, market_cap, market_percent_change)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (r['date'], r['symbol'], r['pe'], r['pbv'], r['dividend_yield'],
              r['market_cap'], r['market_percent_change']))
    conn.commit()


# ============================================================
# 6. SETTRADE FOREIGN FLOW HISTORICAL BACKFILL
# ============================================================

SETTRADE_API_BASE = 'https://www.settrade.com/api'


def get_settrade_session():
    """Get a session with cookies from SETTRADE website."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.settrade.com/th/equities/market-data/investor-type',
    })
    try:
        resp = session.get('https://www.settrade.com/th/equities/market-data/investor-type', timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f'  Warning: could not init SETTRADE session: {e}')
    return session


def fetch_foreign_flows_historical(session):
    """Try SETTRADE API for historical foreign investor flows.
    Attempts multiple API endpoints that may have multi-year history."""
    results = []

    # Attempt 1: SETTRADE investor type historical API
    endpoints = [
        f'{SETTRADE_API_BASE}/set/market/investortype/historical',
        f'{SETTRADE_API_BASE}/set/market/SET/investortype/historical',
        f'{SETTRADE_API_BASE}/set/market/investor-type/historical',
    ]

    for url in endpoints:
        try:
            print(f'  Trying: {url}')
            resp = session.get(url, timeout=15, params={
                'market': 'SET',
                'investorType': 'foreign',
                'limit': 1000,
            })
            if resp.status_code != 200:
                print(f'    HTTP {resp.status_code}')
                continue

            data = resp.json()
            if not data:
                print(f'    Empty response')
                continue

            # Handle different response formats
            records = data if isinstance(data, list) else data.get('data', data.get('records', []))
            if not records:
                print(f'    No records found')
                continue

            for row in records:
                date_str = row.get('date', row.get('tradingDate', ''))[:10]
                if not date_str:
                    continue

                # Try different field names
                buy_val = float(row.get('foreignBuyValue', row.get('buyValue', row.get('foreignBuy', 0))))
                sell_val = float(row.get('foreignSellValue', row.get('sellValue', row.get('foreignSell', 0))))
                net_val = float(row.get('foreignNetValue', row.get('netValue', row.get('foreignNet', buy_val - sell_val))))
                total = buy_val + sell_val
                buy_pct = (buy_val / total * 100) if total > 0 else 0
                sell_pct = (sell_val / total * 100) if total > 0 else 0

                results.append({
                    'date': date_str,
                    'buy_value': buy_val,
                    'sell_value': sell_val,
                    'net_value': net_val,
                    'buy_pct': round(buy_pct, 2),
                    'sell_pct': round(sell_pct, 2),
                })

            print(f'    Got {len(results)} records from SETTRADE')
            return results

        except Exception as e:
            print(f'    ERROR: {e}')

    # Attempt 2: SET API with date range parameters
    print('  Trying SET API with date range...')
    set_session = get_set_session()
    end_date = datetime.now()
    # Try fetching in monthly chunks going back 3 years
    for months_back in range(0, 36, 3):
        chunk_end = end_date - timedelta(days=months_back * 30)
        chunk_start = chunk_end - timedelta(days=90)

        for url_pattern in [
            f'{SET_API_BASE}/api/set/market/SET/investor-type?startDate={{start}}&endDate={{end}}',
            f'{SET_API_BASE}/api/set/market/investor-type/historical?startDate={{start}}&endDate={{end}}',
        ]:
            url = url_pattern.format(
                start=chunk_start.strftime('%Y-%m-%d'),
                end=chunk_end.strftime('%Y-%m-%d'),
            )
            try:
                resp = set_session.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                records = data if isinstance(data, list) else data.get('data', [])
                if not records:
                    continue
                for row in records:
                    investors = row.get('investors', [row])
                    for inv in investors:
                        if inv.get('type', '').lower() != 'foreign':
                            continue
                        date_str = row.get('asOfDate', row.get('date', ''))[:10]
                        if not date_str:
                            continue
                        results.append({
                            'date': date_str,
                            'buy_value': float(inv.get('buyValue', 0)),
                            'sell_value': float(inv.get('sellValue', 0)),
                            'net_value': float(inv.get('netValue', 0)),
                            'buy_pct': float(inv.get('percentBuyValue', 0)),
                            'sell_pct': float(inv.get('percentSellValue', 0)),
                        })
                if results:
                    print(f'    Got {len(results)} records via SET date range API')
                    break
            except Exception:
                continue
        if results and months_back > 0:
            break  # found a working endpoint, will continue chunking
        time.sleep(0.5)

    if results:
        # Deduplicate by date
        seen = set()
        unique = []
        for r in results:
            if r['date'] not in seen:
                seen.add(r['date'])
                unique.append(r)
        return unique

    print('  No historical foreign flow data available from APIs')
    return results


def fetch_historical_trading_extended(session, symbols, count=500):
    """Try SETTRADE API for deeper PE/PBV history beyond SET's 121-day limit."""
    results = []

    for symbol in symbols:
        sym_short = symbol.replace('.BK', '')

        # Try SETTRADE API endpoints for longer history
        endpoints = [
            f'{SETTRADE_API_BASE}/set/stock/{sym_short}/historical-trading?count={count}',
            f'https://www.settrade.com/api/set/stock/{sym_short}/historical-trading?limit={count}',
        ]

        for url in endpoints:
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if not isinstance(data, list) or len(data) == 0:
                    continue

                for row in data:
                    date_str = row.get('date', '')[:10]
                    if not date_str:
                        continue
                    results.append({
                        'date': date_str,
                        'symbol': symbol,
                        'pe': row.get('pe'),
                        'pbv': row.get('pbv'),
                        'dividend_yield': row.get('dividendYield'),
                        'market_cap': row.get('marketCap'),
                        'market_percent_change': row.get('marketPercentChange'),
                    })

                print(f'  {sym_short}: {len(data)} days (extended)')
                break  # Got data from this endpoint
            except Exception:
                continue

        time.sleep(0.3)

    return results


def import_csv_foreign_flows(csv_path):
    """Import foreign flow data from a downloaded CSV file.
    Expected columns: date, buy_value, sell_value, net_value (or similar).
    Flexible column matching for SET website CSVs."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f'  ERROR: {csv_path} is empty')
        return []
    print(f'  Read {len(df)} rows from {csv_path}')
    print(f'  Columns: {list(df.columns)}')

    results = []
    # Try to match columns flexibly
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if 'date' in cl or 'วันที่' in cl:
            col_map['date'] = col
        elif 'net' in cl and 'foreign' in cl:
            col_map['net_value'] = col
        elif 'buy' in cl and 'foreign' in cl:
            col_map['buy_value'] = col
        elif 'sell' in cl and 'foreign' in cl:
            col_map['sell_value'] = col
        elif cl == 'net' or cl == 'net_value':
            col_map.setdefault('net_value', col)
        elif cl == 'buy' or cl == 'buy_value':
            col_map.setdefault('buy_value', col)
        elif cl == 'sell' or cl == 'sell_value':
            col_map.setdefault('sell_value', col)

    if 'date' not in col_map:
        # Try first column as date
        col_map['date'] = df.columns[0]

    for _, row in df.iterrows():
        date_str = str(row[col_map['date']])[:10]
        buy_val = float(row.get(col_map.get('buy_value', ''), 0) or 0)
        sell_val = float(row.get(col_map.get('sell_value', ''), 0) or 0)
        net_val = float(row.get(col_map.get('net_value', ''), buy_val - sell_val) or 0)
        total = buy_val + sell_val
        buy_pct = (buy_val / total * 100) if total > 0 else 0
        sell_pct = (sell_val / total * 100) if total > 0 else 0

        results.append({
            'date': date_str,
            'buy_value': buy_val,
            'sell_value': sell_val,
            'net_value': net_val,
            'buy_pct': round(buy_pct, 2),
            'sell_pct': round(sell_pct, 2),
        })

    return results


def import_monthly_foreign_flows(xls_path):
    """Import monthly foreign flow data from SET XLS file (HTML format).
    Parses table 1 from the XLS, skips header rows, converts month names
    to YYYY-MM format, and stores in foreign_flows_monthly table.
    Values are in million baht."""
    tables = pd.read_html(xls_path)
    df = tables[1]

    # Skip header rows (row 0: NaN/SET/SET/mai/mai, row 1: Month-Year/BUY/SELL/BUY/SELL)
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = ['month_year', 'set_buy', 'set_sell', 'mai_buy', 'mai_sell']

    results = []
    for _, row in df.iterrows():
        month_str = str(row['month_year']).strip()
        if not month_str or month_str == 'nan':
            continue

        # Convert "Feb-2026" → "2026-02"
        try:
            dt = datetime.strptime(month_str, '%b-%Y')
            month_key = dt.strftime('%Y-%m')
        except ValueError:
            print(f'  Skipping unparseable month: {month_str}')
            continue

        try:
            buy_val = float(row['set_buy'])
        except (ValueError, TypeError):
            buy_val = 0.0
        try:
            sell_val = float(row['set_sell'])
        except (ValueError, TypeError):
            sell_val = 0.0

        net_val = buy_val - sell_val
        results.append({
            'month': month_key,
            'buy_value': buy_val,
            'sell_value': sell_val,
            'net_value': round(net_val, 2),
        })

    print(f'  Parsed {len(results)} monthly records from {xls_path}')
    if results:
        print(f'  Date range: {results[-1]["month"]} to {results[0]["month"]}')

    return results


def store_monthly_foreign_flows(conn, results):
    """Create table and store monthly foreign flow data."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS foreign_flows_monthly (
            month TEXT PRIMARY KEY,
            buy_value REAL,
            sell_value REAL,
            net_value REAL
        )
    ''')
    for r in results:
        conn.execute('''
            INSERT OR REPLACE INTO foreign_flows_monthly
            (month, buy_value, sell_value, net_value)
            VALUES (?, ?, ?, ?)
        ''', (r['month'], r['buy_value'], r['sell_value'], r['net_value']))
    conn.commit()
    count = conn.execute('SELECT COUNT(*) FROM foreign_flows_monthly').fetchone()[0]
    print(f'  Total monthly foreign flow rows: {count}')


def import_csv_historical_trading(csv_path):
    """Import historical trading data from a downloaded CSV file.
    Expected columns: date, symbol, pe, pbv, dividend_yield."""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f'  ERROR: {csv_path} is empty')
        return []
    print(f'  Read {len(df)} rows from {csv_path}')
    print(f'  Columns: {list(df.columns)}')

    results = []
    for _, row in df.iterrows():
        symbol = str(row.get('symbol', row.get('Symbol', '')))
        if not symbol.endswith('.BK'):
            symbol = symbol + '.BK'
        results.append({
            'date': str(row.get('date', row.get('Date', '')))[:10],
            'symbol': symbol,
            'pe': row.get('pe', row.get('PE', None)),
            'pbv': row.get('pbv', row.get('PBV', None)),
            'dividend_yield': row.get('dividend_yield', row.get('DividendYield', None)),
            'market_cap': row.get('market_cap', row.get('MarketCap', None)),
            'market_percent_change': row.get('market_percent_change', None),
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Fetch supplemental market data')
    parser.add_argument('--backfill', action='store_true',
                        help='Backfill 60 days of intraday data')
    parser.add_argument('--backfill-foreign', action='store_true',
                        help='Backfill historical foreign flow data via SETTRADE API')
    parser.add_argument('--backfill-historical', action='store_true',
                        help='Backfill extended PE/PBV history via SETTRADE API')
    parser.add_argument('--import-csv', type=str, metavar='FILE',
                        help='Import data from CSV file')
    parser.add_argument('--csv-type', choices=['foreign', 'historical'],
                        help='Type of CSV data to import (use with --import-csv)')
    parser.add_argument('--import-monthly-foreign', type=str, metavar='FILE',
                        help='Import monthly foreign flow data from SET XLS file')
    parser.add_argument('--only', choices=['intraday', 'sector', 'foreign', 'historical'],
                        help='Only fetch one data type')
    args = parser.parse_args()

    conn = get_conn()
    init_tables(conn)

    # 1. Intraday first-hour volume
    if args.only is None or args.only == 'intraday':
        print('\n=== Intraday First-Hour Volume ===')
        results = fetch_intraday(SYMBOLS, backfill=args.backfill)
        store_intraday(conn, results)
        count = conn.execute('SELECT COUNT(*) FROM intraday_features').fetchone()[0]
        print(f'  Total intraday rows: {count}')

    # 2. Sector indices
    if args.only is None or args.only == 'sector':
        print('\n=== Sector Indices ===')
        session = get_set_session()
        # Try bulk fetch first, fall back to individual
        results = fetch_all_sector_indices(session)
        if not results:
            results = fetch_sector_indices(session)
        store_sector_indices(conn, results)
        count = conn.execute('SELECT COUNT(*) FROM sector_indices').fetchone()[0]
        print(f'  Total sector rows: {count}')

    # 3. Foreign investor flows
    if args.only is None or args.only == 'foreign':
        print('\n=== Foreign Investor Flows ===')
        session = get_set_session() if 'session' not in dir() else session
        result = fetch_foreign_flows(session)
        store_foreign_flows(conn, result)
        count = conn.execute('SELECT COUNT(*) FROM foreign_flows').fetchone()[0]
        print(f'  Total foreign flow rows: {count}')

    # 4. Historical trading (PE, PBV, dividend yield)
    if args.only is None or args.only == 'historical':
        print('\n=== Historical Trading (PE/PBV) ===')
        session = get_set_session()
        results = fetch_historical_trading(session, SYMBOLS)
        store_historical_trading(conn, results)
        count = conn.execute('SELECT COUNT(*) FROM historical_trading').fetchone()[0]
        print(f'  Total historical trading rows: {count}')

    # 5. Backfill foreign flows (SETTRADE API)
    if args.backfill_foreign:
        print('\n=== Backfill Foreign Flows (SETTRADE) ===')
        settrade_session = get_settrade_session()
        results = fetch_foreign_flows_historical(settrade_session)
        if results:
            for r in results:
                conn.execute('''
                    INSERT OR REPLACE INTO foreign_flows
                    (date, buy_value, sell_value, net_value, buy_pct, sell_pct)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (r['date'], r['buy_value'], r['sell_value'],
                      r['net_value'], r['buy_pct'], r['sell_pct']))
            conn.commit()
        count = conn.execute('SELECT COUNT(*) FROM foreign_flows').fetchone()[0]
        date_range = conn.execute('SELECT MIN(date), MAX(date) FROM foreign_flows').fetchone()
        print(f'  Total foreign flow rows: {count} ({date_range[0]} to {date_range[1]})')

    # 6. Backfill historical trading (SETTRADE API)
    if args.backfill_historical:
        print('\n=== Backfill Historical Trading (SETTRADE) ===')
        settrade_session = get_settrade_session()
        results = fetch_historical_trading_extended(settrade_session, SYMBOLS)
        if results:
            store_historical_trading(conn, results)
        count = conn.execute('SELECT COUNT(*) FROM historical_trading').fetchone()[0]
        date_range = conn.execute('SELECT MIN(date), MAX(date) FROM historical_trading').fetchone()
        print(f'  Total historical trading rows: {count} ({date_range[0]} to {date_range[1]})')

    # 7. Monthly foreign flow import (SET XLS)
    if args.import_monthly_foreign:
        print(f'\n=== Import Monthly Foreign Flows ===')
        results = import_monthly_foreign_flows(args.import_monthly_foreign)
        if results:
            store_monthly_foreign_flows(conn, results)
            date_range = conn.execute(
                'SELECT MIN(month), MAX(month) FROM foreign_flows_monthly'
            ).fetchone()
            print(f'  Range: {date_range[0]} to {date_range[1]}')

    # 8. CSV import
    if args.import_csv:
        csv_type = args.csv_type
        if not csv_type:
            print('ERROR: --csv-type required with --import-csv')
            sys.exit(1)

        print(f'\n=== Import CSV ({csv_type}) ===')
        if csv_type == 'foreign':
            results = import_csv_foreign_flows(args.import_csv)
            for r in results:
                conn.execute('''
                    INSERT OR REPLACE INTO foreign_flows
                    (date, buy_value, sell_value, net_value, buy_pct, sell_pct)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (r['date'], r['buy_value'], r['sell_value'],
                      r['net_value'], r['buy_pct'], r['sell_pct']))
            conn.commit()
            count = conn.execute('SELECT COUNT(*) FROM foreign_flows').fetchone()[0]
            print(f'  Total foreign flow rows after import: {count}')
        elif csv_type == 'historical':
            results = import_csv_historical_trading(args.import_csv)
            store_historical_trading(conn, results)
            count = conn.execute('SELECT COUNT(*) FROM historical_trading').fetchone()[0]
            print(f'  Total historical trading rows after import: {count}')

    conn.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
