"""Phase 0c — call the Settrade APIs directly using cookies harvested from a
Selenium session. If this works, we don't need a Selenium driver per page —
we just need to seed cookies once (Selenium passes the Incapsula challenge),
then use plain requests for everything.

Steps:
  1. Open Settrade homepage with Selenium → Incapsula challenge resolved → cookies set
  2. Extract cookies, build a `requests.Session`
  3. Hit /api/set/stock/list — confirm symbol-list works
  4. Try several plausible historical-bars endpoints for KBANK
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

CACHE_DIR = BASE / 'data' / 'settrade' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def harvest_cookies() -> tuple[dict, str]:
    """Open Settrade with Selenium; return (cookies, user_agent)."""
    print('opening settrade homepage to harvest cookies...')
    with driver_session(headless=False) as drv:
        drv.get('https://www.settrade.com/th')
        time.sleep(8)  # let the page load + Incapsula resolve
        cookies = {c['name']: c['value'] for c in drv.get_cookies()}
        ua = drv.execute_script('return navigator.userAgent')
        print(f'  harvested {len(cookies)} cookies; UA: {ua[:80]}...')
        return cookies, ua


def make_session(cookies: dict, ua: str) -> requests.Session:
    s = requests.Session()
    s.cookies.update(cookies)
    s.headers.update({
        'User-Agent': ua,
        'Accept': 'application/json,text/plain,*/*',
        'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
        'Referer': 'https://www.settrade.com/th',
        'Origin': 'https://www.settrade.com',
    })
    return s


def try_endpoint(s: requests.Session, url: str, label: str) -> dict:
    print(f'\n--- {label} ---\n  {url}')
    try:
        r = s.get(url, timeout=15)
        ctype = r.headers.get('Content-Type', '')
        size = len(r.content)
        print(f'  status: {r.status_code}  ctype: {ctype}  size: {size:,}')
        # Save body for inspection
        suffix = '.json' if 'json' in ctype else '.html'
        cache = CACHE_DIR / f'api_{label}{suffix}'
        cache.write_bytes(r.content)
        print(f'  saved: {cache}')
        if 'json' in ctype and r.ok:
            try:
                data = r.json()
                if isinstance(data, list):
                    print(f'  parsed: list[{len(data)}]')
                    if data:
                        print(f'  first: {json.dumps(data[0])[:200]}')
                elif isinstance(data, dict):
                    print(f'  parsed: dict keys={list(data.keys())[:10]}')
                    # Show a sample value if dict has data-like content
                    for k in ('data', 'rows', 'items', 'result'):
                        if k in data:
                            v = data[k]
                            if isinstance(v, list):
                                print(f'  data.{k}: list[{len(v)}], first={json.dumps(v[0])[:200] if v else "[]"}')
                                break
            except Exception as e:
                print(f'  json parse error: {e}')
        return {'url': url, 'status': r.status_code, 'size': size, 'ctype': ctype}
    except Exception as e:
        print(f'  EXCEPTION: {type(e).__name__}: {e}')
        return {'url': url, 'error': str(e)}


def main():
    cookies, ua = harvest_cookies()
    s = make_session(cookies, ua)

    results = []

    # 1) symbol list — confirmed working from XHR trace
    results.append(try_endpoint(s,
        'https://www.settrade.com/api/set/stock/list',
        'stock_list'))

    # 2) try historical-bars endpoints — guesses based on Settrade's REST shape.
    sym = 'KBANK'
    candidates = [
        f'https://www.settrade.com/api/set/stock/{sym}/historical-trading',
        f'https://www.settrade.com/api/set/stock/{sym}/historical-trading?period=3Y',
        f'https://www.settrade.com/api/set/stock/{sym}/historical',
        f'https://www.settrade.com/api/set/stock/{sym}/history',
        f'https://www.settrade.com/api/set/stock/{sym}/chart-historical',
        f'https://www.settrade.com/api/set/stock/{sym}/eod',
        f'https://www.settrade.com/api/set/stock/{sym}/info',
        f'https://www.settrade.com/api/set/stock/{sym}/quote',
        f'https://www.settrade.com/api/set/stock/{sym}',
    ]
    for url in candidates:
        results.append(try_endpoint(s, url, url.rsplit('/', 1)[-1].replace('?', '_').replace('=', '_')))

    # 3) related-product endpoints we saw — confirm they still work too
    results.append(try_endpoint(s,
        f'https://www.settrade.com/api/set/stock/{sym}/related-product/W',
        'related_W'))

    # Summary
    print('\n' + '=' * 60)
    print('Phase 0c summary')
    print('=' * 60)
    for r in results:
        ok = r.get('status') == 200 and r.get('size', 0) > 200
        flag = '✅' if ok else '❌'
        print(f'  {flag}  {r.get("status")}  {r.get("size", 0):>10,}b  {r.get("url")}')


if __name__ == '__main__':
    main()
