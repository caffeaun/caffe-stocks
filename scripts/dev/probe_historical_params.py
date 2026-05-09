"""Probe param shapes for /api/set/stock/{SYM}/historical-trading.
Goal: figure out how to get >117 bars (full 3y) per symbol.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from scripts.settrade_driver import driver_session

CACHE = BASE / 'data' / 'settrade' / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)
SYM = 'KBANK'


def harvest_cookies():
    with driver_session(headless=False) as drv:
        drv.get('https://www.settrade.com/th')
        time.sleep(8)
        cookies = {c['name']: c['value'] for c in drv.get_cookies()}
        ua = drv.execute_script('return navigator.userAgent')
        return cookies, ua


def main():
    cookies, ua = harvest_cookies()
    s = requests.Session()
    s.cookies.update(cookies)
    s.headers.update({
        'User-Agent': ua,
        'Accept': 'application/json,text/plain,*/*',
        'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
        'Referer': f'https://www.settrade.com/th/equities/quote/{SYM}/historical-trading',
        'Origin': 'https://www.settrade.com',
    })

    base = f'https://www.settrade.com/api/set/stock/{SYM}/historical-trading'

    # Inspect the cached XHR network log for hints first — what params did the
    # page actually use? Print the recorded historical-trading request URL.
    log_path = CACHE / f'{SYM}_resource_log.json'
    if log_path.exists():
        log = json.load(open(log_path))
        hits = [e for e in log if 'historical-trading' in e['name'] and '/api/' in e['name']]
        print(f'\n--- prior page recorded {len(hits)} historical-trading API hits ---')
        for h in hits[:5]:
            print(f'  {h["name"]}')
        # No hits in earlier capture — page lazy-loads on interaction.
        # We'll discover the param shape by guessing.

    candidates = [
        ('no-param', {}),
        ('period=3Y', {'period': '3Y'}),
        ('period=ALL', {'period': 'ALL'}),
        ('period=1Y', {'period': '1Y'}),
        ('period=5Y', {'period': '5Y'}),
        ('period=MAX', {'period': 'MAX'}),
        ('period=6M', {'period': '6M'}),
        ('limit=1000', {'limit': 1000}),
        ('size=1000', {'size': 1000}),
        ('count=1000', {'count': 1000}),
        ('lang=en', {'lang': 'en'}),
        ('lang=en+period=3Y', {'lang': 'en', 'period': '3Y'}),
        ('from+to', {'from': '2023-05-01', 'to': '2026-05-06'}),
        ('fromDate+toDate', {'fromDate': '2023-05-01', 'toDate': '2026-05-06'}),
        ('startDate+endDate', {'startDate': '2023-05-01', 'endDate': '2026-05-06'}),
        ('start+end', {'start': '2023-05-01', 'end': '2026-05-06'}),
    ]

    print(f'\n--- probing {len(candidates)} param combos ---')
    for label, params in candidates:
        try:
            r = s.get(base, params=params, timeout=15)
            n = 'parse-fail'
            first = ''
            last = ''
            if r.ok:
                try:
                    data = r.json()
                    if isinstance(data, list):
                        n = len(data)
                        if data:
                            first = data[0].get('date', '')[:10]
                            last = data[-1].get('date', '')[:10]
                    else:
                        n = f'dict, keys={list(data.keys())[:3]}'
                except Exception:
                    n = 'non-json'
            print(f'  {label:30s}  status={r.status_code}  bars={n!s:10s}  range={first}..{last}  size={len(r.content):,}')
        except Exception as e:
            print(f'  {label:30s}  ERR: {type(e).__name__}: {e}')

    # Also try alternative endpoints
    print(f'\n--- alternative endpoint shapes ---')
    alt_endpoints = [
        f'https://www.settrade.com/api/set/stock/{SYM}/historical-trading-extended',
        f'https://www.settrade.com/api/set/stock/{SYM}/historical-trading-all',
        f'https://www.settrade.com/api/set/stock/{SYM}/historical-trading/extended',
        f'https://www.settrade.com/api/marketdata/stock/{SYM}/historical-trading',
        f'https://marketdata.set.or.th/mkt/stockquotation.do?symbol={SYM}&language=en&country=US',
        f'https://www.settrade.com/api/set/stock/{SYM}/historical-eod',
        f'https://www.settrade.com/api/set/stock/{SYM}/eod-historical',
        f'https://www.settrade.com/api/set/stock/{SYM}/historical-trading?period=3Y&limit=1000',
    ]
    for url in alt_endpoints:
        try:
            r = s.get(url, timeout=15)
            n = ''
            if r.ok:
                try:
                    data = r.json()
                    if isinstance(data, list):
                        n = f'list[{len(data)}]'
                    else:
                        n = f'dict[{",".join(list(data.keys())[:5])}]'
                except Exception:
                    n = 'non-json'
            print(f'  {r.status_code:>3}  {len(r.content):>8,}b  {n:25s}  {url}')
        except Exception as e:
            print(f'  ---  ERR  {url}  ({e})')


if __name__ == '__main__':
    main()
