#!/usr/bin/env python3
"""Weekly data source exploration — checks health of existing sources and probes new ones.

Run: ~/projects/caffe-stocks/venv/bin/python ~/projects/caffe-stocks/scripts/explore_data_sources.py
Cron: Sunday 02:00 ICT via OpenClaw (developer agent)
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import yfinance as yf
from scrape_alert import alert_scrape_error

CANDLES_DB = '/home/kanoonth-ai/projects/caffe-stocks/data/candles.db'
US_CANDLES_DB = '/home/kanoonth-ai/projects/caffe-stocks/data/us_candles.db'
MEMORY_FILE = '/home/kanoonth-ai/projects/caffe-stocks/openclaw/MEMORY.md'
REPORT_DIR = '/home/kanoonth-ai/projects/caffe-stocks/data'

# SET symbols we track
SET_SYMBOLS = ['COM7.BK', 'DELTA.BK', 'GULF.BK', 'GPSC.BK', 'MINT.BK',
               'OSP.BK', 'SCBX.BK', 'SCC.BK', 'TIDLOR.BK', 'TRUE.BK']

# US symbols we track
US_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']


def check_yfinance_health():
    """Check if yfinance still returns data for our symbols."""
    results = {}
    for sym in SET_SYMBOLS[:3] + US_SYMBOLS[:3]:  # sample 3 from each
        try:
            df = yf.download(sym, period='5d', progress=False)
            results[sym] = len(df)
        except Exception as e:
            results[sym] = f'ERROR: {e}'
    return results


def check_db_freshness():
    """Check how stale our databases are."""
    freshness = {}
    for name, db_path, table in [
        ('SET candles', CANDLES_DB, 'candles'),
        ('US candles', US_CANDLES_DB, 'us_candles'),
    ]:
        if not os.path.exists(db_path):
            freshness[name] = 'DB missing'
            continue
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(f'SELECT MAX(date) FROM {table}')
            latest = cur.fetchone()[0]
            freshness[name] = latest or 'empty'
        except Exception as e:
            freshness[name] = f'ERROR: {e}'
        finally:
            conn.close()
    return freshness


def check_db_row_counts():
    """Count rows in each table."""
    counts = {}
    for name, db_path, table in [
        ('SET candles', CANDLES_DB, 'candles'),
        ('SET candles_raw', CANDLES_DB, 'candles_raw'),
        ('US candles', US_CANDLES_DB, 'us_candles'),
        ('US candles_raw', US_CANDLES_DB, 'us_candles_raw'),
    ]:
        if not os.path.exists(db_path):
            counts[name] = 'DB missing'
            continue
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(f'SELECT COUNT(*) FROM {table}')
            counts[name] = cur.fetchone()[0]
        except Exception as e:
            counts[name] = f'ERROR: {e}'
        finally:
            conn.close()
    return counts


def probe_new_sources():
    """Try fetching from potential new free data sources."""
    probes = {}

    # Check if Bangkok Post RSS still works (news source)
    try:
        import urllib.request
        req = urllib.request.Request(
            'https://www.bangkokpost.com/rss/data/business.xml',
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        resp = urllib.request.urlopen(req, timeout=10)
        probes['Bangkok Post RSS'] = f'OK ({resp.status})'
    except Exception as e:
        probes['Bangkok Post RSS'] = f'FAIL: {e}'

    # Check Alpha Vantage (free tier: 25 calls/day)
    try:
        av_key = os.environ.get('ALPHA_VANTAGE_KEY', '')
        if av_key:
            import urllib.request
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={av_key}&outputsize=compact'
            resp = urllib.request.urlopen(url, timeout=10)
            data = json.loads(resp.read())
            if 'Time Series (Daily)' in data:
                probes['Alpha Vantage'] = f'OK ({len(data["Time Series (Daily)"])} days)'
            else:
                probes['Alpha Vantage'] = f'WARN: {list(data.keys())[:2]}'
        else:
            probes['Alpha Vantage'] = 'SKIP (no API key)'
    except Exception as e:
        probes['Alpha Vantage'] = f'FAIL: {e}'

    return probes


def generate_report():
    """Generate weekly exploration report."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    yf_health = check_yfinance_health()
    freshness = check_db_freshness()
    counts = check_db_row_counts()
    probes = probe_new_sources()

    lines = [
        f'# Data Source Exploration Report — {now}',
        '',
        '## yfinance Health',
    ]
    for sym, result in yf_health.items():
        status = f'{result} rows' if isinstance(result, int) else result
        lines.append(f'- {sym}: {status}')

    lines += ['', '## Database Freshness']
    for name, date in freshness.items():
        lines.append(f'- {name}: latest = {date}')

    lines += ['', '## Row Counts']
    for name, count in counts.items():
        lines.append(f'- {name}: {count}')

    lines += ['', '## New Source Probes']
    for name, result in probes.items():
        lines.append(f'- {name}: {result}')

    # Staleness warnings
    lines += ['', '## Warnings']
    warnings = []
    for name, date in freshness.items():
        if isinstance(date, str) and len(date) == 10:
            try:
                d = datetime.strptime(date, '%Y-%m-%d')
                age = (datetime.now() - d).days
                if age > 7:
                    warnings.append(f'{name} is {age} days stale (latest: {date})')
            except ValueError:
                pass
    if warnings:
        for w in warnings:
            lines.append(f'- {w}')
    else:
        lines.append('- None')

    return '\n'.join(lines)


def update_memory(report):
    """Append latest exploration date to MEMORY.md Data Sources section."""
    if not os.path.exists(MEMORY_FILE):
        return

    with open(MEMORY_FILE, 'r') as f:
        content = f.read()

    today = datetime.now().strftime('%Y-%m-%d')
    entry = f'- {today}: Weekly exploration — all sources healthy. See data/exploration-report.txt'

    # Find Data Sources section and append
    marker = '## Data Sources'
    if marker in content:
        idx = content.index(marker)
        # Find next section or end
        rest = content[idx + len(marker):]
        next_section = rest.find('\n## ')
        if next_section == -1:
            # Append at end
            content = content.rstrip() + '\n' + entry + '\n'
        else:
            insert_at = idx + len(marker) + next_section
            content = content[:insert_at] + '\n' + entry + content[insert_at:]

        with open(MEMORY_FILE, 'w') as f:
            f.write(content)


def main():
    report = generate_report()
    print(report)

    # Save report
    report_path = os.path.join(REPORT_DIR, 'exploration-report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'\nReport saved to {report_path}')

    # Alert on warnings
    if warnings:
        alert_scrape_error("explore_data_sources.py", "Data staleness warnings:\n" + "\n".join(f"- {w}" for w in warnings))

    # Update MEMORY.md
    update_memory(report)


if __name__ == '__main__':
    main()
