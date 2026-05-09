"""Settrade source adapter.

Settrade publishes a clean public JSON API behind Incapsula. We harvest
cookies once via Selenium (undetected-chromedriver bypasses the JS challenge),
then make plain `requests` calls for every symbol.

Hard cap: the historical-trading endpoint returns at most ~117 bars
(rolling 6 months). That's why this adapter is the **last** in the backfill
priority order — it's strictly a daily-refresh source with a 6mo safety net
when truly nothing else has the symbol. The orchestrator handles fallthrough.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Optional

import requests

from scripts.sources.base import Bar, normalize_symbol, to_bar, with_retry
from scripts.settrade_driver import driver_session, is_incapsula_wall

log = logging.getLogger(__name__)

API_BASE = 'https://www.settrade.com/api/set/stock'
HOMEPAGE = 'https://www.settrade.com/th'


def harvest_cookies(timeout_s: float = 8.0) -> tuple[dict, str]:
    """One-time browser visit to populate Incapsula cookies. Returns
    (cookies, user_agent) for use with a requests.Session."""
    with driver_session(headless=False) as drv:
        drv.get(HOMEPAGE)
        time.sleep(timeout_s)
        cookies = {c['name']: c['value'] for c in drv.get_cookies()}
        ua = drv.execute_script('return navigator.userAgent')
        return cookies, ua


class SettradeSource:
    name = 'settrade'

    def __init__(self, throttle_s: float = 0.3):
        self._throttle_s = throttle_s
        self._session: Optional[requests.Session] = None

    def _ensure_session(self, force: bool = False) -> requests.Session:
        if self._session is None or force:
            cookies, ua = harvest_cookies()
            s = requests.Session()
            s.cookies.update(cookies)
            s.headers.update({
                'User-Agent': ua,
                'Accept': 'application/json,text/plain,*/*',
                'Accept-Language': 'en-US,en;q=0.9,th;q=0.8',
                'Referer': HOMEPAGE,
                'Origin': 'https://www.settrade.com',
            })
            self._session = s
            log.info('settrade: harvested %d cookies', len(cookies))
        return self._session

    def _get_json(self, url: str) -> Optional[list | dict]:
        s = self._ensure_session()
        r = s.get(url, timeout=15)
        # Detect Incapsula challenge re-firing — harvest again, retry once
        if r.status_code in (403, 429) or is_incapsula_wall(r.text):
            log.warning('settrade: cookie expired (%d), re-harvesting', r.status_code)
            s = self._ensure_session(force=True)
            r = s.get(url, timeout=15)
        if not r.ok:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def fetch_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        ticker = normalize_symbol(symbol, self.name)
        url = f'{API_BASE}/{ticker}/historical-trading'
        try:
            data = with_retry(lambda: self._get_json(url))
        except Exception as e:
            log.warning('settrade fetch_history(%s) failed: %r', ticker, e)
            return []
        if not isinstance(data, list):
            return []
        bars: list[Bar] = []
        for row in data:
            try:
                ts = str(row['date'])[:10]
                if ts < start.isoformat() or ts > end.isoformat():
                    continue
                bars.append(to_bar(
                    timestamp=ts,
                    open=row['open'], high=row['high'], low=row['low'],
                    close=row['close'], volume=row.get('totalVolume', 0),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        # Settrade returns newest-first; sort ascending for consistency.
        bars.sort(key=lambda b: b['timestamp'])
        time.sleep(self._throttle_s)
        return bars

    def fetch_daily(self, symbol: str) -> Optional[Bar]:
        ticker = normalize_symbol(symbol, self.name)
        url = f'{API_BASE}/{ticker}/historical-trading'
        try:
            data = with_retry(lambda: self._get_json(url))
        except Exception as e:
            log.warning('settrade fetch_daily(%s) failed: %r', ticker, e)
            return None
        if not isinstance(data, list) or not data:
            return None
        # Newest-first — first row is today's (or last close's) bar.
        try:
            row = data[0]
            return to_bar(
                timestamp=str(row['date'])[:10],
                open=row['open'], high=row['high'], low=row['low'],
                close=row['close'], volume=row.get('totalVolume', 0),
            )
        except (KeyError, TypeError, ValueError):
            return None
