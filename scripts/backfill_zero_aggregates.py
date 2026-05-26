"""One-shot backfill: recompute avg_annualized_return / avg_win_rate /
avg_max_dd / total_trades on iteration rows that were recorded with
zeros due to the pre-fix _aggregate() schema mismatch.

Targets rows where:
  - windows_passed > 0 (the WF gate actually ran)
  - total_trades = 0 AND avg_win_rate = 0 AND avg_annualized_return = 0
  - full_result JSON is present and parseable

For each such row, re-runs _aggregate() against the stored gate_result
JSON in `full_result` and UPDATEs the four aggregate columns. The
iteration_windows table is left untouched — it's keyed off the same
JSON but rebuilding it would require a DELETE/INSERT pair per affected
iter, which isn't justified for leaderboard display.

Dry-run by default; pass --apply to commit.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.feedback import _aggregate  # noqa: E402

DB_PATH = BASE / 'data' / 'ml-feedback.db'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='Commit the updates (default: dry-run)')
    ap.add_argument('--limit', type=int, default=0,
                    help='Only process the first N candidate rows')
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, mode, trainer, windows_passed, full_result
        FROM iterations
        WHERE windows_passed > 0
          AND COALESCE(total_trades, 0) = 0
          AND COALESCE(avg_win_rate, 0) = 0
          AND COALESCE(avg_annualized_return, 0) = 0
          AND full_result IS NOT NULL
        ORDER BY id
    """).fetchall()

    print(f'candidates: {len(rows)}')

    fixed = 0
    skipped_parse = 0
    skipped_no_change = 0
    if args.limit:
        rows = rows[:args.limit]

    for r in rows:
        try:
            gr = json.loads(r['full_result'])
        except (json.JSONDecodeError, TypeError):
            skipped_parse += 1
            continue
        agg = _aggregate(gr)
        if (agg['total_trades'] == 0
                and agg['avg_win_rate'] == 0.0
                and agg['avg_annualized_return'] == 0.0):
            skipped_no_change += 1
            continue
        if args.apply:
            conn.execute("""
                UPDATE iterations
                SET avg_annualized_return = ?, avg_win_rate = ?,
                    avg_max_dd = ?, total_trades = ?
                WHERE id = ?
            """, (
                agg['avg_annualized_return'], agg['avg_win_rate'],
                agg['avg_max_dd'], agg['total_trades'], r['id'],
            ))
        print(f'  #{r["id"]:<5} {r["mode"]:6} {r["trainer"][:24]:24} '
              f'wp={r["windows_passed"]}/7  '
              f'wr={agg["avg_win_rate"]*100:5.1f}%  '
              f'ann={agg["avg_annualized_return"]*100:+6.1f}%  '
              f'dd={agg["avg_max_dd"]*100:4.1f}%  '
              f'n={agg["total_trades"]}')
        fixed += 1

    if args.apply:
        conn.commit()
        print(f'\nApplied: {fixed} rows updated')
    else:
        print(f'\nDry-run: would update {fixed} rows; '
              f'{skipped_parse} skipped (parse), '
              f'{skipped_no_change} skipped (no change)')
        print('Re-run with --apply to commit.')


if __name__ == '__main__':
    main()
