"""yfinance source adapter.

Wraps `yf.download(symbol, start, end, interval='1d')`. Date-range parameterized
so backfill and daily refresh use the same call shape. Returns [] on empty
response (yfinance silently returns empty DataFrames when rate-limited or for
unknown tickers — the orchestrator handles fallthrough).
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from scripts.sources.base import Bar, normalize_symbol, to_bar, with_retry

log = logging.getLogger(__name__)


class YFinanceSource:
    name = 'yfinance'

    def __init__(self, throttle_s: float = 0.3):
        self._throttle_s = throttle_s

    def _download(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        # `end` is exclusive in yfinance — bump by 1 day so the user's
        # "[start, end] inclusive" contract holds.
        return yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval='1d',
            progress=False,
            auto_adjust=False,
            threads=False,
        )

    def fetch_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        ticker = normalize_symbol(symbol, self.name)
        try:
            df = with_retry(lambda: self._download(ticker, start, end))
        except Exception as e:
            log.warning('yfinance fetch_history(%s) failed: %r', ticker, e)
            return []
        if df is None or df.empty:
            return []
        # Multi-index columns can appear with single-symbol calls in some
        # versions; flatten defensively.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        bars: list[Bar] = []
        for ts, row in df.iterrows():
            try:
                bars.append(to_bar(
                    timestamp=ts.strftime('%Y-%m-%d'),
                    open=row['Open'], high=row['High'], low=row['Low'],
                    close=row['Close'], volume=row['Volume'],
                ))
            except (KeyError, ValueError):
                continue
        time.sleep(self._throttle_s)
        return bars

    def fetch_daily(self, symbol: str) -> Optional[Bar]:
        # Use last 5d window so weekends / holidays still surface yesterday.
        today = datetime.now().date()
        bars = self.fetch_history(symbol, today - timedelta(days=5), today)
        return bars[-1] if bars else None
