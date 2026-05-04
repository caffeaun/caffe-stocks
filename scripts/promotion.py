#!/usr/bin/env python3
"""Refresh the top-3 paper-trading panel from recent iterations.

Cron-triggered monthly (1st of month, 00:01 ICT). Can also be run by hand
or via Telegram (`t panel refresh`).
"""
import argparse
import os
import sys
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import feedback as fb


def show_panel(panel):
    if not panel:
        print('Panel is empty.')
        return
    print('\nCurrent paper-trade panel:')
    print('-' * 80)
    for row in panel:
        ann = row['avg_annualized_return']
        wr = row['avg_win_rate']
        dd = row['avg_max_dd']
        n = row['total_trades']
        print(f"  rank {row['rank']}  iter#{row['iteration_id']}  {row['trainer']:10s}"
              f"  ann={ann:+.1%}  wr={wr:.1%}  dd={dd:.1%}  trades={n}")
        print(f"             promoted {row['promoted_at'][:16]}  expires {row['expires_at'][:10]}")
    print()


def show_top_recent(days=30, n=10):
    rows = fb.top_n_recent(n=n, days=days, only_passed=False)
    if not rows:
        print(f'No iterations in last {days} days.')
        return
    print(f'\nTop {len(rows)} recent iterations (last {days} days):')
    print('-' * 80)
    for r in rows:
        ann = r['avg_annualized_return']
        wr = r['avg_win_rate']
        dd = r['avg_max_dd']
        passed = '✓' if r['gate_passed'] else '✗'
        print(f"  iter#{r['id']:4d} {r['mode']:6s} {r['trainer']:10s} {passed} "
              f"ann={ann:+.1%}  wr={wr:.1%}  dd={dd:.1%}  trades={r['total_trades']:3d}  "
              f"({r['finished_at'][:16]})")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30,
                        help='How many days back to consider for panel selection')
    parser.add_argument('--show-only', action='store_true',
                        help='Just print the current panel; do not refresh')
    args = parser.parse_args()

    if args.show_only:
        show_panel(fb.get_promotion_panel())
        show_top_recent(days=args.days)
        return

    print('=== Refreshing paper-trade panel ===')
    show_top_recent(days=args.days)

    new_panel = fb.refresh_promotion_panel(days=args.days)
    if not new_panel:
        print(f'\nNo eligible iterations in last {args.days} days — panel cleared.')
        return

    print('\n=== New panel ===')
    show_panel(new_panel)


if __name__ == '__main__':
    main()
