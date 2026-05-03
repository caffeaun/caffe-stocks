#!/usr/bin/env python3
"""Supplemental data scraper for Investing.com — fetches VIX historical data
and economic calendar events via Playwright headless browser.

Supplements yfinance (primary) with sentiment/macro data not available there:
- VIX (CBOE Volatility Index) — market fear gauge
- Economic calendar — upcoming TH/US macro events
- SET index via yfinance (^SET.BK) — more reliable than Investing.com scraping

Run: ~/shared-venv/bin/python ~/projects/caffe-stocks/scripts/scrape_investing.py
"""
import json
import os
import sqlite3
import sys
from datetime import datetime

import yfinance as yf
from playwright.sync_api import sync_playwright
from scrape_alert import alert_scrape_error

DB_PATH = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'
OUTPUT_DIR = '/home/kanoonth-ai/projects/caffe-stocks/data'

VIX_URL = 'https://www.investing.com/indices/volatility-s-p-500-historical-data'
ECON_CAL_URL = 'https://www.investing.com/economic-calendar/'


def fetch_set_index_yfinance():
    """Fetch SET index from yfinance (more reliable than scraping Investing.com)."""
    print('Fetching SET index via yfinance...')
    try:
        df = yf.download('^SET.BK', period='1mo', interval='1d', progress=False)
        if df.empty:
            print('  No data')
            return []
        data = []
        for date, row in df.iterrows():
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': float(row[('Open', '^SET.BK')]),
                'high': float(row[('High', '^SET.BK')]),
                'low': float(row[('Low', '^SET.BK')]),
                'close': float(row[('Close', '^SET.BK')]),
                'volume': int(row[('Volume', '^SET.BK')]),
            })
        print(f'  Got {len(data)} rows')
        return data
    except Exception as e:
        print(f'  Error: {e}')
        return []


def scrape_vix_table(page):
    """Scrape VIX historical data from Investing.com."""
    print('Fetching VIX historical data...')
    try:
        page.goto(VIX_URL, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(5000)  # let JS render tables

        # Find the historical data table (has data-test attr or is the largest table with dates)
        table = page.query_selector('table[data-test="historical-data-table"]')
        if not table:
            # Fallback: find table with >10 rows containing date-like first cells
            for t in page.query_selector_all('table'):
                trs = t.query_selector_all('tr')
                if len(trs) > 10:
                    table = t
                    break
        rows = table.query_selector_all('tr') if table else []
        data = []
        for row in rows:
            cells = row.query_selector_all('td')
            if len(cells) >= 5:
                try:
                    date_str = cells[0].inner_text().strip()
                    # Investing.com order: Date, Close, Open, High, Low, Vol, Change%
                    close = cells[1].inner_text().strip().replace(',', '')
                    open_ = cells[2].inner_text().strip().replace(',', '')
                    high = cells[3].inner_text().strip().replace(',', '')
                    low = cells[4].inner_text().strip().replace(',', '')
                    # Validate it looks like a date row
                    if not any(m in date_str for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                        continue
                    data.append({
                        'date': date_str,
                        'open': float(open_),
                        'high': float(high),
                        'low': float(low),
                        'close': float(close),
                    })
                except (ValueError, IndexError):
                    continue
        print(f'  Got {len(data)} rows')
        return data
    except Exception as e:
        print(f'  Error: {e}')
        return []


def scrape_economic_calendar(page):
    """Scrape economic calendar — falls back to simple table parse if JS fails."""
    print('Fetching economic calendar...')
    try:
        page.goto(ECON_CAL_URL, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(5000)  # let JS render

        # Try multiple selectors
        events = []
        rows = page.query_selector_all('tr.js-event-item') or page.query_selector_all('[data-test="economic-calendar-row"]')
        if not rows:
            # Fallback: look for any table with event-like structure
            rows = page.query_selector_all('table tbody tr')

        for row in rows[:50]:  # limit to 50 events
            try:
                cells = row.query_selector_all('td')
                if len(cells) < 3:
                    continue
                text = ' | '.join([c.inner_text().strip()[:50] for c in cells if c.inner_text().strip()])
                if text and len(text) > 10:
                    events.append(text)
            except Exception:
                continue

        print(f'  Got {len(events)} events')
        return events
    except Exception as e:
        print(f'  Error: {e}')
        return []


def save_set_index(data):
    """Save SET index data to supplemental table."""
    if not data:
        return 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS investing_set_index (
            date TEXT PRIMARY KEY,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        )
    ''')
    count = 0
    for row in data:
        cursor.execute('''
            INSERT OR REPLACE INTO investing_set_index (date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (row['date'], row['open'], row['high'], row['low'], row['close'], row.get('volume', 0)))
        count += 1
    conn.commit()
    conn.close()
    return count


def main():
    dry_run = '--dry-run' in sys.argv
    print(f'Investing.com supplemental scraper — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print()

    # 1. SET index via yfinance (reliable)
    errors = []
    set_data = fetch_set_index_yfinance()
    if set_data:
        if dry_run:
            print(f'  [dry-run] Would save {len(set_data)} rows')
        else:
            count = save_set_index(set_data)
            print(f'  Saved {count} rows to investing_set_index table')
    else:
        errors.append('SET index (yfinance): no data returned')

    # 2-3. VIX + Economic calendar via Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1280, 'height': 720},
        )
        page = context.new_page()

        # 2. VIX
        vix_data = scrape_vix_table(page)
        if vix_data:
            path = os.path.join(OUTPUT_DIR, 'vix_sentiment.json')
            if dry_run:
                print(f'  [dry-run] Would save {len(vix_data)} rows')
                for row in vix_data[:3]:
                    print(f'    {row}')
            else:
                with open(path, 'w') as f:
                    json.dump({
                        'fetched': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'data': vix_data,
                    }, f, indent=2)
                print(f'  Saved to {path}')
        else:
            errors.append('VIX (Investing.com): no data — page may have changed')

        # 3. Economic calendar
        events = scrape_economic_calendar(page)
        if events:
            path = os.path.join(OUTPUT_DIR, 'economic_calendar.json')
            if dry_run:
                print(f'  [dry-run] Would save {len(events)} events')
                for e in events[:3]:
                    print(f'    {e}')
            else:
                with open(path, 'w') as f:
                    json.dump({
                        'fetched': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                        'events': events,
                    }, f, indent=2)
                print(f'  Saved to {path}')
        else:
            errors.append('Economic calendar: no events — page may have changed')

        browser.close()

    if errors and not dry_run:
        alert_scrape_error("scrape_investing.py", "\n".join(errors))

    print('\nDone.')


if __name__ == '__main__':
    main()
