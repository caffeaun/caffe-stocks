"""Source priority configuration for fetch_multi_source.py.

PRIORITY_BACKFILL — order tried for `--mode backfill` (3y history).
PRIORITY_DAILY    — order tried for `--mode daily` (one bar).
SYMBOL_OVERRIDES  — per-symbol custom priority. Used for symbols where the
                    default chain is known to be wrong (e.g. yfinance returning
                    Facebook for `META.BK`).

All source names must match the `name` attribute of an installed adapter in
scripts/sources/. Unknown names are skipped with a warning at runtime.

Adapters appear here in declaration order. Adapters not yet built are
included for documentation but stripped at runtime by build_sources().
"""
from __future__ import annotations

PRIORITY_BACKFILL: list[str] = [
    'yfinance',   # 3y free, fastest, covers ~267 Thai names
    'settrade',   # 6mo cap (server-side); fills the yfinance gap with whatever
                  # 6mo gives us — daily refresh later accumulates full history.
    # Sources not in this list — investigated 2026-05-06, no free path:
    #   stooq    — apikey + captcha now required for CSV downloads
    #   set_csv  — set.or.th has no CSV download button; the historical-trading
    #              endpoint shares the 117-bar cap with settrade (same backend)
    #   investing — exists but Cloudflare-protected + slugs aren't deterministic;
    #              high build cost, low yield. Defer until a specific symbol
    #              forces the work.
]

PRIORITY_DAILY: list[str] = [
    'yfinance',   # proven for the existing 267
    'settrade',   # clean public API, low latency, Thai-authoritative
]

# Per-symbol override of the default priority list. Most symbols don't need
# this. Populated mechanically as we discover yfinance gaps.
SYMBOL_OVERRIDES: dict[str, list[str]] = {
    # Examples (uncomment when discovered):
    # 'META.BK':   ['settrade', 'stooq', 'set_csv'],   # yfinance gives Facebook
}
