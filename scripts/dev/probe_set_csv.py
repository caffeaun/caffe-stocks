"""Probe set.or.th for downloadable historical CSV per stock.

We need to know:
  1. Can we reach a stock's historical-trading page on set.or.th?
  2. Does it expose a "Download CSV" button or a downloadable file URL?
  3. What history depth does it give us (3y? 5y? unlimited?)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from scripts.settrade_driver import driver_session, is_incapsula_wall

SYM = 'KBANK'
URL_CANDIDATES = [
    f'https://www.set.or.th/en/market/product/stock/quote/{SYM}/historical-trading',
    f'https://www.set.or.th/th/market/product/stock/quote/{SYM}/historical-trading',
    f'https://www.set.or.th/en/market/product/stock/quote/{SYM}/price',
]
CACHE = BASE / 'data' / 'settrade' / 'cache'


def probe(url: str, label: str):
    print(f'\n--- {label}: {url} ---')
    with driver_session(headless=False) as drv:
        drv.execute_cdp_cmd('Network.enable', {})
        drv.get(url)
        time.sleep(10)  # let SPA hydrate + any historical XHR fire
        title = drv.title
        html_size = len(drv.page_source)
        print(f'  title:     {title!r}')
        print(f'  html size: {html_size:,}')
        print(f'  incapsula: {is_incapsula_wall(drv.page_source)}')

        # Capture all XHR/fetch calls to a SET domain
        entries = drv.execute_script("""
            return performance.getEntriesByType('resource')
                .filter(e => e.initiatorType === 'xmlhttprequest' || e.initiatorType === 'fetch')
                .filter(e => e.name.includes('set.or.th') || e.name.includes('settrade.com'))
                .map(e => ({name: e.name, size: e.transferSize, init: e.initiatorType}));
        """)
        print(f'  XHR/fetch to SET domains: {len(entries)}')
        for e in entries:
            short = e['name'].split('?', 1)[0]
            print(f'    [{e["init"]:14s}] {e["size"]:>7}b  {short[:130]}')

        # Look for download buttons or CSV-related elements
        try:
            buttons = drv.find_elements('xpath',
                "//*[contains(translate(text(), 'CSVDOWNLOAD', 'csvdownload'), 'csv') "
                "or contains(translate(text(), 'CSVDOWNLOAD', 'csvdownload'), 'download') "
                "or contains(translate(text(), 'CSVDOWNLOAD', 'csvdownload'), 'export')]")
            print(f'  download/csv/export elements visible: {len(buttons)}')
            for b in buttons[:8]:
                txt = (b.text or b.get_attribute('aria-label') or
                       b.get_attribute('href') or b.get_attribute('value') or '').strip()[:80]
                tag = b.tag_name
                print(f'    <{tag}> {txt!r}')
        except Exception as e:
            print(f'  control discovery error: {e}')

        # Save a snapshot of the page for offline inspection
        snap = CACHE / f'set_or_th_{label}.html'
        snap.write_text(drv.page_source)
        print(f'  saved: {snap}')


def main():
    for i, url in enumerate(URL_CANDIDATES, 1):
        probe(url, f'attempt_{i}')


if __name__ == '__main__':
    main()
