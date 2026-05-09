"""Probe set.or.th JSON endpoints with cookies harvested from set.or.th itself
(NOT settrade.com — different Incapsula domain). Looking for an endpoint
returning more than the 117-bar settrade cap.
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
SYM = 'KBANK'
HOMEPAGE = 'https://www.set.or.th/en'


def harvest():
    print('opening set.or.th to harvest cookies...')
    with driver_session(headless=False) as drv:
        drv.get(HOMEPAGE)
        time.sleep(8)
        cookies = {c['name']: c['value'] for c in drv.get_cookies()}
        ua = drv.execute_script('return navigator.userAgent')
        return cookies, ua


def main():
    cookies, ua = harvest()
    s = requests.Session()
    s.cookies.update(cookies)
    s.headers.update({
        'User-Agent': ua,
        'Accept': 'application/json,text/plain,*/*',
        'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
        'Referer': f'https://www.set.or.th/en/market/product/stock/quote/{SYM}/historical-trading',
        'Origin': 'https://www.set.or.th',
    })

    endpoints = [
        f'https://www.set.or.th/api/set/stock/{SYM}/historical-trading',
        f'https://www.set.or.th/api/set/stock/{SYM}/historical-trading?period=3Y',
        f'https://www.set.or.th/api/set/stock/{SYM}/historical-trading?period=5Y',
        f'https://www.set.or.th/api/set/stock/{SYM}/historical-trading?lang=en',
        f'https://www.set.or.th/api/set/stock/{SYM}/historical-trading?from=2023-05-01&to=2026-05-06',
        f'https://www.set.or.th/api/set/stock/{SYM}/chart-quotation',
        f'https://www.set.or.th/api/set/stock/{SYM}/chart-quotation?period=3Y',
        f'https://www.set.or.th/api/set/stock/{SYM}/chart-quotation?period=5Y',
        f'https://www.set.or.th/api/set/stock/{SYM}/chart-quotation?period=ALL',
        f'https://www.set.or.th/api/set/stock/{SYM}/chart-quotation?from=2023-05-01&to=2026-05-06',
    ]

    for url in endpoints:
        try:
            r = s.get(url, timeout=15)
            sz = len(r.content)
            n_bars = ''
            first = last = ''
            if r.ok:
                try:
                    d = r.json()
                    if isinstance(d, list):
                        n_bars = f'list[{len(d)}]'
                        if d:
                            for k in ('date', 'timestamp', 'tradingDate', 'time', 'tradeDate'):
                                if k in d[0]:
                                    first = str(d[0][k])[:10]
                                    last = str(d[-1][k])[:10]
                                    break
                    elif isinstance(d, dict):
                        # For chart-quotation, it's typically a dict with arrays
                        keys = list(d.keys())[:5]
                        n_bars = f'dict[{",".join(keys)}]'
                        # Look for an OHLCV-style array nested in the dict
                        for k in ('data', 'rows', 'items', 'series', 'result', 'historicalData'):
                            if k in d and isinstance(d[k], list):
                                n_bars += f' .{k}=list[{len(d[k])}]'
                                if d[k]:
                                    s_keys = list(d[k][0].keys())[:6] if isinstance(d[k][0], dict) else []
                                    n_bars += f' keys={s_keys}'
                                break
                except ValueError:
                    n_bars = 'non-json'
            short = url.split('?', 1)[0]
            params = url.split('?', 1)[1][:50] if '?' in url else ''
            print(f'  {r.status_code:>3}  {sz:>8,}b  {n_bars:50s}  {first}..{last}  {params}')
            # Save full response of historical-trading (no params) for inspection
            if 'historical-trading' in url and '?' not in url and r.ok:
                (CACHE / 'set_or_th_historical_trading_kbank.json').write_bytes(r.content)
            if 'chart-quotation' in url and '?' not in url and r.ok:
                (CACHE / 'set_or_th_chart_quotation_kbank.json').write_bytes(r.content)
        except Exception as e:
            print(f'  ERR  {url}  ({type(e).__name__}: {e})')


if __name__ == '__main__':
    main()
