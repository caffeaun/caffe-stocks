"""Hunt for an endpoint that returns more than 117 bars.
Tries: SET's own API, settrade pagination, and a Selenium-driven page scroll
to capture any 'load more' XHR.
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

SYM = 'KBANK'
CACHE = BASE / 'data' / 'settrade' / 'cache'


def harvest():
    with driver_session(headless=False) as drv:
        drv.get('https://www.settrade.com/th')
        time.sleep(8)
        c = {x['name']: x['value'] for x in drv.get_cookies()}
        ua = drv.execute_script('return navigator.userAgent')
        return c, ua


def main():
    cookies, ua = harvest()
    s = requests.Session()
    s.cookies.update(cookies)
    s.headers.update({
        'User-Agent': ua, 'Accept': 'application/json,text/plain,*/*',
        'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
        'Referer': f'https://www.settrade.com/th/equities/quote/{SYM}/historical-trading',
        'Origin': 'https://www.settrade.com',
    })

    print('=== try SET official API (different host) ===')
    set_endpoints = [
        f'https://www.set.or.th/api/set/stock/{SYM}/historical-trading',
        f'https://www.set.or.th/api/set/stock/{SYM}/historical-trading?period=3Y',
        f'https://www.set.or.th/api/set/stock/{SYM}/info',
    ]
    for url in set_endpoints:
        try:
            r = s.get(url, timeout=15)
            n = ''
            if r.ok:
                try:
                    d = r.json()
                    n = f'list[{len(d)}]' if isinstance(d, list) else f'dict[{",".join(list(d.keys())[:3])}]'
                    if isinstance(d, list) and d:
                        n += f' range={d[0].get("date","")[:10]}..{d[-1].get("date","")[:10]}'
                except Exception:
                    n = 'non-json'
            print(f'  {r.status_code:>3}  {len(r.content):>8,}b  {n:50s}  {url}')
        except Exception as e:
            print(f'  ERR  {url}  ({type(e).__name__}: {e})')

    print('\n=== try paging on settrade endpoint ===')
    base_url = f'https://www.settrade.com/api/set/stock/{SYM}/historical-trading'
    for params in [{'page': 2}, {'page': 1, 'size': 1000}, {'offset': 117}, {'before': '2025-11-06'},
                    {'before': '2025-11-05'}, {'lang': 'en', 'before': '2025-11-06'}]:
        r = s.get(base_url, params=params, timeout=15)
        n = ''
        first = last = ''
        try:
            d = r.json()
            if isinstance(d, list):
                n = f'list[{len(d)}]'
                if d:
                    first = d[0].get('date', '')[:10]
                    last = d[-1].get('date', '')[:10]
        except Exception:
            n = 'non-json'
        print(f'  {str(params):60s}  status={r.status_code}  {n:15s}  range={first}..{last}')

    print('\n=== open page in headed browser, scroll, capture XHR ===')
    with driver_session(headless=False) as drv:
        drv.execute_cdp_cmd('Network.enable', {})
        drv.get(f'https://www.settrade.com/th/equities/quote/{SYM}/historical-trading')
        time.sleep(8)

        # Scroll down to bottom several times to trigger any infinite-scroll XHR
        for i in range(8):
            drv.execute_script('window.scrollTo(0, document.body.scrollHeight)')
            time.sleep(1.5)

        # Try to find an interactive control — buttons / radios / dropdowns
        try:
            controls = drv.find_elements('css selector',
                'button, .btn, [role="button"], input[type="radio"], select')
            print(f'  controls visible: {len(controls)}')
            # Look for date-range / period selectors specifically
            for c in controls[:40]:
                txt = (c.text or c.get_attribute('aria-label') or
                       c.get_attribute('value') or '')
                txt = txt.strip()[:30]
                if txt and any(kw in txt.lower() for kw in
                               ('1y', '3y', '5y', '6m', '1m', 'year', 'month',
                                'period', 'range', 'history', 'load')):
                    print(f'    candidate: {txt!r}  tag={c.tag_name}')
        except Exception as e:
            print(f'  control discovery failed: {e}')

        # Read all resources after scroll/interaction
        entries = drv.execute_script("""
            return performance.getEntriesByType('resource')
                .filter(e => e.initiatorType === 'xmlhttprequest' || e.initiatorType === 'fetch')
                .filter(e => e.name.includes('settrade.com') || e.name.includes('set.or.th'))
                .map(e => ({name: e.name, size: e.transferSize}));
        """)
        print(f'  XHR/fetch to set domains: {len(entries)}')
        for e in entries:
            short = e['name'].split('?', 1)[0]
            print(f'    {e["size"]:>7}b  {short}')


if __name__ == '__main__':
    main()
