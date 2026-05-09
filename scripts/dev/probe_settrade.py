"""Phase 0 probe — verify Selenium + undetected-chromedriver can reach Settrade
and pull the historical bars table for one well-known stock (KBANK).

Tries headless first (cheapest); if that hits Incapsula or returns an empty
table, falls back to headed for the same URL and reports which path succeeded.

Outputs:
  - data/settrade/cache/{key}.html — raw HTML for later parser iteration
  - stdout — page title, body length, presence of expected DOM markers
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from scripts.settrade_driver import (
    driver_session, is_incapsula_wall, cache_raw_html, throttle
)

CACHE_DIR = BASE / 'data' / 'settrade' / 'cache'
TARGET_SYMBOL = 'KBANK'

# Two URL patterns to try. Modern English path first; fall back to the legacy
# Thai / generic path if that 404s or has different DOM structure.
URL_CANDIDATES = [
    f'https://www.settrade.com/en/equities/quote/{TARGET_SYMBOL}/historical-trading',
    f'https://www.settrade.com/th/equities/quote/{TARGET_SYMBOL}/historical-trading',
    f'https://www.settrade.com/en/equities/quote/{TARGET_SYMBOL}/price',
]


def probe_one(url: str, headless: bool, label: str) -> dict:
    """Open `url`, wait briefly, capture title + html. Cache the HTML.
    Returns a result dict with diagnostics."""
    print(f'\n--- {label} ---')
    print(f'URL: {url}')
    print(f'headless: {headless}')
    t0 = time.time()
    try:
        with driver_session(headless=headless) as drv:
            drv.get(url)
            time.sleep(5)  # let JS settle / Incapsula challenge resolve
            title = drv.title
            html = drv.page_source
            elapsed = time.time() - t0
            wall = is_incapsula_wall(html)
            key = f'{TARGET_SYMBOL}_{label.replace(" ", "_")}'
            cache = cache_raw_html(html, CACHE_DIR, key)
            print(f'title:        {title!r}')
            print(f'html size:    {len(html):,} chars')
            print(f'incapsula?:   {wall}')
            print(f'time:         {elapsed:.1f}s')
            print(f'cached HTML:  {cache}')

            markers = [
                ('historical table marker',
                 any(m in html for m in (
                     'historical-trading', 'history-table', 'StockHistorical'
                 ))),
                ('OHLCV row marker',
                 any(m in html.lower() for m in (
                     'open</th>', 'open</td>', '"open"', 'high</th>'
                 ))),
                ('quote header',
                 TARGET_SYMBOL in html),
            ]
            for name, present in markers:
                print(f'  marker {name:30s}: {present}')
            return {
                'url': url, 'headless': headless, 'incapsula': wall,
                'title': title, 'size': len(html), 'cache': str(cache),
                'markers': dict(markers),
            }
    except Exception as e:
        print(f'EXCEPTION: {type(e).__name__}: {e}')
        return {'url': url, 'headless': headless, 'error': str(e)}


def main():
    results = []
    # Try headless on first URL only — cheap test
    r = probe_one(URL_CANDIDATES[0], headless=True, label='headless_modern_en')
    results.append(r)

    # If headless got blocked or empty, retry headed
    blocked_or_empty = (
        r.get('incapsula') or
        r.get('error') or
        r.get('size', 0) < 50_000 or
        not r.get('markers', {}).get('quote header', False)
    )
    if blocked_or_empty:
        print('\nheadless looked blocked/empty — falling back to headed mode')
        throttle()
        r2 = probe_one(URL_CANDIDATES[0], headless=False, label='headed_modern_en')
        results.append(r2)

    # Try a couple of other URL shapes too if the modern one didn't work
    if not any(r.get('markers', {}).get('OHLCV row marker', False) for r in results):
        for i, url in enumerate(URL_CANDIDATES[1:], 1):
            throttle()
            r = probe_one(url, headless=False, label=f'headed_alt_{i}')
            results.append(r)
            if r.get('markers', {}).get('OHLCV row marker', False):
                break

    # Summary
    print('\n' + '=' * 60)
    print('Phase 0 summary')
    print('=' * 60)
    for r in results:
        url = r.get('url')
        ok = bool(r.get('markers', {}).get('OHLCV row marker'))
        wall = r.get('incapsula', False)
        size = r.get('size', 0)
        flag = '✅' if ok else ('🛑 incap' if wall else '⚠️ no OHLCV')
        print(f'  {flag}  hl={r.get("headless")}  {size:>8,}c  {url}')


if __name__ == '__main__':
    main()
