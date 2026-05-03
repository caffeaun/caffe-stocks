import os
import sys
import sqlite3
import argparse
import json
import logging
from datetime import datetime
from playwright.sync_api import sync_playwright
from scrape_alert import alert_scrape_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# List of SET mid-cap symbols to scrape
SYMBOLS = ['COM7', 'JMART', 'SAWAD', 'MTC', 'JMT']

# Settrade API endpoints (accessible via browser context with session cookies)
# /api/set/stock/{symbol}/info        -> intraday OHLCV (open, high, low, last, totalVolume)
# /api/set/stock/{symbol}/historical-trading -> EOD OHLCV array


def fetch_stock_data(page, symbol):
    """
    Fetch current-day OHLCV data for a symbol via the Settrade internal API.
    Uses page.evaluate so the fetch runs inside the browser's cookie context.
    Returns a dict with keys: open, high, low, close, volume, or None on failure.
    """
    result = page.evaluate(f'''async () => {{
        try {{
            const r = await fetch('/api/set/stock/{symbol}/info', {{
                headers: {{
                    'Accept': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }}
            }});
            if (!r.ok) return {{error: r.status}};
            return await r.json();
        }} catch (e) {{
            return {{error: e.toString()}};
        }}
    }}''')

    if not result or 'error' in result:
        logging.warning(f'{symbol}: API error: {result}')
        alert_scrape_error("scrape_settrade.py", f"{symbol}: API error — {result}")
        return None

    return {
        'open': result.get('open') or 0.0,
        'high': result.get('high') or 0.0,
        'low': result.get('low') or 0.0,
        'close': result.get('last') or 0.0,  # 'last' is the current/close price
        'volume': int(result.get('totalVolume') or 0),
    }


def scrape_settrade(dry_run=False):
    """Scrape Settrade for current-day OHLCV data using the internal JSON API."""
    conn = sqlite3.connect('/home/kanoonth-ai/projects/caffe-stocks/data/candles.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settrade_eod (
            timestamp TEXT,
            symbol TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    ''')

    today = datetime.now().strftime('%Y-%m-%d')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to settrade.com to establish a browser session with cookies
        page.goto('https://www.settrade.com/', timeout=30000)
        page.wait_for_load_state('domcontentloaded', timeout=15000)
        # Brief pause to allow session/cookie initialization
        page.wait_for_timeout(2000)

        all_data = []
        for symbol in SYMBOLS:
            try:
                data = fetch_stock_data(page, symbol)
                if data is None:
                    logging.warning(f'Skipping {symbol}: no data returned')
                    continue

                row = (today, symbol, data['open'], data['high'], data['low'], data['close'], data['volume'])
                all_data.append(row)
                logging.info(
                    f'Fetched {symbol}: open={data["open"]:.2f} high={data["high"]:.2f} '
                    f'low={data["low"]:.2f} close={data["close"]:.2f} vol={data["volume"]}'
                )
            except Exception as e:
                logging.warning(f'Failed to fetch {symbol}: {e}')
                alert_scrape_error("scrape_settrade.py", f"{symbol}: {e}")

        browser.close()

    for row in all_data:
        if dry_run:
            print(f'DRY RUN: {row}')
        else:
            cursor.execute('''
                INSERT OR REPLACE INTO settrade_eod VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', row)
            logging.info(f'Inserted: {row}')

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    args = parser.parse_args()

    scrape_settrade(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
