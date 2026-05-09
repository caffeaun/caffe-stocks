"""Settrade Selenium driver helpers.

Centralizes browser setup, anti-bot config, throttling, and retries so the
discovery and history scripts only worry about parsing, not infrastructure.

Why undetected-chromedriver: Settrade is behind Incapsula. Vanilla Selenium
trips it within seconds. undetected-chromedriver patches the standard webdriver
fingerprints so the JS challenge passes silently.

Default mode is **headed** for Phase 0 reliability. Switch to headless via
`HEADLESS = True` once we've confirmed the parser path works end-to-end.
"""
from __future__ import annotations

import random
import time
from contextlib import contextmanager
from pathlib import Path

import undetected_chromedriver as uc

DEFAULT_THROTTLE = (3.0, 5.0)  # uniform jitter in seconds between page loads
DEFAULT_TIMEOUT = 30           # implicit wait per element / page
HEADLESS = False               # Phase 0: headed for reliability + visual debug


def make_driver(headless: bool | None = None,
                user_data_dir: str | None = None) -> uc.Chrome:
    """Construct a configured Chrome instance. Caller owns the lifecycle.

    Use `with driver_session()` if you don't want to manage `.quit()` manually.
    """
    headless = HEADLESS if headless is None else headless
    options = uc.ChromeOptions()
    options.add_argument('--window-size=1280,1024')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')
    if headless:
        # uc supports the new headless mode out of the box; flag is opt-in.
        options.add_argument('--headless=new')
    if user_data_dir:
        options.add_argument(f'--user-data-dir={user_data_dir}')
    driver = uc.Chrome(options=options, version_main=147)
    driver.set_page_load_timeout(DEFAULT_TIMEOUT)
    return driver


@contextmanager
def driver_session(**kwargs):
    """Context manager that ensures driver.quit() runs even on exception."""
    drv = make_driver(**kwargs)
    try:
        yield drv
    finally:
        try:
            drv.quit()
        except Exception:
            pass


def throttle(low: float = DEFAULT_THROTTLE[0],
             high: float = DEFAULT_THROTTLE[1]) -> None:
    time.sleep(random.uniform(low, high))


def is_incapsula_wall(html: str) -> bool:
    """Detect Incapsula JS-challenge / block pages."""
    markers = (
        '_Incapsula_Resource',
        'Incapsula incident',
        'Request unsuccessful. Incapsula',
    )
    return any(m in html for m in markers)


def cache_raw_html(html: str, cache_dir: Path, key: str) -> Path:
    """Persist raw page HTML for resilient re-parsing later."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f'{key}.html'
    out.write_text(html, encoding='utf-8')
    return out
