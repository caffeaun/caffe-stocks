"""Source Protocol + shared utilities for the multi-source historical data layer.

A `Source` is anything that can answer two questions:
  - fetch_history(symbol, start, end) -> list[Bar]
  - fetch_daily(symbol)               -> Bar | None

Adapters return [] / None on failure (never raise). The orchestrator decides
fallthrough; the adapter only owns "did *I* succeed for this symbol?".
"""
from __future__ import annotations

import time
from datetime import date
from typing import Callable, Optional, Protocol, TypedDict, runtime_checkable


class Bar(TypedDict):
    """Canonical OHLCV row. Sources convert their native formats to this."""
    timestamp: str   # 'YYYY-MM-DD'
    open: float
    high: float
    low: float
    close: float
    volume: int


@runtime_checkable
class Source(Protocol):
    """A historical-data source. Implementations are classes — they hold
    per-source state (cookies, drivers, sessions) so the orchestrator can
    share them across many calls in one run.
    """
    name: str  # short slug stored in candles_raw.source

    def fetch_history(self, symbol: str, start: date, end: date) -> list[Bar]:
        """Return all bars in [start, end] inclusive. Empty list on failure."""
        ...

    def fetch_daily(self, symbol: str) -> Optional[Bar]:
        """Return today's (or last close's) bar. None on failure."""
        ...


# ----- shared helpers ---------------------------------------------------

def normalize_symbol(symbol: str, source_name: str) -> str:
    """Map our canonical `KBANK.BK` form into per-source ticker conventions.

    yfinance:  KBANK.BK
    settrade:  KBANK    (bare; the API path adds context)
    stooq:     kbank.th (lowercase, .th suffix)
    set_csv:   KBANK    (bare)
    investing: handled by slug lookup, not by this function
    """
    bare = symbol.replace('.BK', '').upper()
    if source_name == 'yfinance':
        return symbol if symbol.endswith('.BK') else f'{bare}.BK'
    if source_name in ('settrade', 'set_csv'):
        return bare
    if source_name == 'stooq':
        return f'{bare.lower()}.th'
    return symbol


def with_retry(fn: Callable, *, attempts: int = 3, base_delay: float = 1.0):
    """Call fn() with exponential backoff (1s, 2s, 4s by default).

    Returns whatever fn returns. Swallows exceptions on non-final attempts;
    re-raises only after all attempts exhausted. The empty-list/None
    "failure" sentinel is the adapter's contract — we don't retry on those,
    only on actual exceptions (transient network errors).
    """
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    if last_exc is not None:
        raise last_exc
    return None


def to_bar(*, timestamp, open, high, low, close, volume) -> Bar:
    """Coerce native types to the Bar TypedDict. Defensive on int/float
    boundary (volume sometimes arrives as float)."""
    return {
        'timestamp': str(timestamp)[:10],
        'open': float(open),
        'high': float(high),
        'low': float(low),
        'close': float(close),
        'volume': int(float(volume)),
    }
