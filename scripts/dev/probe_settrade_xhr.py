"""Phase 0b — Settrade is a Nuxt SPA. The static HTML doesn't contain the
OHLCV bars; they come from an XHR after page hydration. Use Chrome DevTools
Protocol (CDP) via Selenium to log network requests, find the JSON endpoint,
and dump a sample response.

If we can identify the JSON API, the rest of the project becomes a much
simpler HTTP-with-cookies job rather than a full DOM scrape.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from scripts.settrade_driver import driver_session, cache_raw_html

CACHE_DIR = BASE / 'data' / 'settrade' / 'cache'
TARGET_SYMBOL = 'KBANK'
URL = f'https://www.settrade.com/th/equities/quote/{TARGET_SYMBOL}/historical-trading'


def main():
    print(f'navigating: {URL}')
    captured = []  # list of (url, status, mime) seen via CDP

    with driver_session(headless=False) as drv:
        # Enable CDP Network domain so we can listen to requestWillBeSent etc.
        drv.execute_cdp_cmd('Network.enable', {})

        # Selenium doesn't expose a clean event stream, but we can poll the
        # browser's performance log if logging is enabled — easier path:
        # use `driver.execute_script` on `window.performance.getEntries()`.
        drv.get(URL)
        time.sleep(15)  # let the SPA hydrate + fire its XHRs

        # Grab all resource entries the page has fetched
        entries = drv.execute_script("""
            return performance.getEntriesByType('resource')
                .map(e => ({
                    name: e.name,
                    initiatorType: e.initiatorType,
                    duration: Math.round(e.duration),
                    transferSize: e.transferSize,
                }));
        """)

        # Filter to interesting ones (XHR / fetch / json paths)
        interesting = [
            e for e in entries
            if e['initiatorType'] in ('xmlhttprequest', 'fetch')
            or '/api/' in e['name']
            or '.json' in e['name']
        ]

        print(f'\n{len(interesting)} interesting requests:')
        for e in interesting[:60]:
            print(f"  [{e['initiatorType']:14s}] {e['transferSize']:>7}b  {e['name'][:140]}")

        # Cache the post-hydration DOM in case the data ended up there
        full_html = drv.page_source
        cache_raw_html(full_html, CACHE_DIR, f'{TARGET_SYMBOL}_hydrated')
        print(f'\npost-hydration HTML: {len(full_html):,} chars')

        # Save the request log for off-line inspection
        log_path = CACHE_DIR / f'{TARGET_SYMBOL}_resource_log.json'
        log_path.write_text(json.dumps(entries, indent=2))
        print(f'resource log:        {log_path}')

        # Try to look in the page for any inline JSON (Nuxt often embeds payload
        # in <script id="__NUXT_DATA__"> or window.__NUXT__)
        nuxt_script_present = '__NUXT_DATA__' in full_html or 'window.__NUXT__' in full_html
        print(f'\nNuxt inline payload present: {nuxt_script_present}')


if __name__ == '__main__':
    main()
