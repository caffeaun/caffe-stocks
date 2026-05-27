#!/usr/bin/env python3
"""Backfill foreign_flows_monthly from foreign_flows (daily aggregate).

The monthly XLS import (scripts/fetch_market_data.import_monthly_foreign_flows)
is a manual workflow keyed to the SET monthly publication; it lags by
~1 month minimum and was last refreshed at 2026-02. As a result,
foreign_net_monthly (and its 60-day pctrank in compute_percentile_ranks)
is NaN for every stock-date in 2026-03 onwards, which makes
prepare_data() drop ALL rows after the last monthly entry — silently
breaking sequence_loader's recent coverage.

This script synthesizes the monthly row from the daily foreign_flows
table (populated by daily-foreign-flow.sh cron, fresh through today).
Conservative: only writes a month's row when the daily table covers ≥10
trading days of that month, so we don't promote a half-empty month into
the rank window.

UNITS: daily foreign_flows.net_value is raw THB (~10^9); monthly
foreign_flows_monthly.net_value is millions of THB (~10^4). Conversion:
daily_sum / 1e6. Matches the XLS-derived rows for the months where both
sources have been spot-checked offline.

After this runs, scripts/compute_indicators.py must be re-run (or the
daily 06:00 daily-feature-refresh.sh cron will do it) to merge the
freshly-backfilled monthly rows back into candles.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date

DB_PATH = os.path.expanduser('~/projects/caffe-stocks/data/candles.db')

# Below this number of trading days in a month, we treat the month as
# incomplete and skip the synthesis (better NaN than a half-month value).
MIN_DAYS_PER_MONTH = 10

# THB → millions of THB.
UNIT_CONVERT = 1_000_000.0


def backfill(db_path: str = DB_PATH, dry_run: bool = False) -> int:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS foreign_flows_monthly ("
            "  month TEXT PRIMARY KEY,"
            "  buy_value REAL, sell_value REAL, net_value REAL"
            ")"
        )
        # Aggregate daily → monthly, only for months meeting the coverage bar.
        rows = conn.execute(f"""
            SELECT strftime('%Y-%m', date) AS month,
                   SUM(buy_value)  / {UNIT_CONVERT} AS buy_M,
                   SUM(sell_value) / {UNIT_CONVERT} AS sell_M,
                   SUM(net_value)  / {UNIT_CONVERT} AS net_M,
                   COUNT(*) AS n_days
            FROM foreign_flows
            GROUP BY month
            HAVING n_days >= ?
            ORDER BY month
        """, (MIN_DAYS_PER_MONTH,)).fetchall()

        if not rows:
            print(f'foreign_flows has no months with ≥{MIN_DAYS_PER_MONTH} '
                  f'trading days yet — nothing to backfill.')
            return 0

        # Check current state of monthly table to report what changes.
        existing = {r[0] for r in conn.execute(
            "SELECT month FROM foreign_flows_monthly"
        ).fetchall()}

        upserts = []
        for month, buy_m, sell_m, net_m, n_days in rows:
            status = 'UPDATE' if month in existing else 'INSERT'
            print(f'  {status} {month}: buy={buy_m:,.0f}M  '
                  f'sell={sell_m:,.0f}M  net={net_m:+,.0f}M  '
                  f'({n_days} days)')
            upserts.append((month, buy_m, sell_m, net_m))

        if dry_run:
            print('\n[dry-run] skipping commit')
            return 0

        conn.executemany("""
            INSERT INTO foreign_flows_monthly (month, buy_value, sell_value, net_value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(month) DO UPDATE SET
              buy_value  = excluded.buy_value,
              sell_value = excluded.sell_value,
              net_value  = excluded.net_value
        """, upserts)
        conn.commit()

        latest_after = conn.execute(
            "SELECT month FROM foreign_flows_monthly ORDER BY month DESC LIMIT 1"
        ).fetchone()
        print(f'\nWrote {len(upserts)} row(s). Latest monthly row now: '
              f'{latest_after[0] if latest_after else "(none)"}')
        return len(upserts)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print upserts without writing.')
    args = parser.parse_args()
    n = backfill(dry_run=args.dry_run)
    return 0 if (args.dry_run or n >= 0) else 1


if __name__ == '__main__':
    raise SystemExit(main())
