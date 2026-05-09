#!/usr/bin/env python3
"""Refresh the top-3 paper-trading panel from recent iterations.

Cron-triggered monthly (1st of month, 00:01 ICT). Can also be run by hand
or via Telegram (`t panel refresh`).

Eligibility (per docs/ml-loop.md §promotion):
  - gate_passed = 1
  - finished_at within --days window
Then dedupe by (trainer, hyperparams), keep top 3 by ann_return DESC, dd ASC.
If no iteration is eligible, the panel is cleared and a Telegram alert fires.
"""
import argparse
import os
import sys
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import feedback as fb


def is_eligible(row: dict) -> bool:
    return bool(row.get('gate_passed'))


def select_panel(rows: list[dict], k: int = 3) -> list[dict]:
    """Apply panel eligibility + dedupe-by-(trainer, hyperparams). Input is
    expected pre-sorted (top_n_recent already orders by ann_return desc, dd asc)."""
    seen = set()
    out = []
    for r in rows:
        if not is_eligible(r):
            continue
        sig = (r['trainer'], r['hyperparams'])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(r)
        if len(out) == k:
            break
    return out


def telegram(text: str) -> None:
    """Best-effort Telegram alert (same conf lookup as claude_mode.py)."""
    conf_paths = [
        BASE.parent / 'ops' / 'telegram' / 'telegram.conf',
        Path.home() / 'kanoonth' / 'scripts' / 'telegram.conf',
    ]
    bot_token = chat_id = None
    for p in conf_paths:
        if p.exists():
            with open(p) as f:
                for line in f:
                    if line.strip().startswith('TELEGRAM_BOT_TOKEN='):
                        bot_token = line.split('=', 1)[1].strip().strip('"\'')
                    elif line.strip().startswith('TELEGRAM_CHAT_ID='):
                        chat_id = line.split('=', 1)[1].strip().strip('"\'')
            break
    if not (bot_token and chat_id):
        return
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': text[:4000],
            'disable_web_page_preview': 'true',
        }).encode()
        urllib.request.urlopen(
            f'https://api.telegram.org/bot{bot_token}/sendMessage',
            data=data, timeout=10)
    except Exception as e:
        print(f'Telegram alert failed: {e}', file=sys.stderr)


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
    rows = fb.top_n_recent(n=n, days=days)
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

    candidates = fb.top_n_recent(n=50, days=args.days)
    selected = select_panel(candidates, k=3)

    if not selected:
        # Panel left empty rather than promoting losers.
        fb.set_panel([])
        msg = (f'No gate-passing iterations in last {args.days} days — panel cleared. '
               f'Loop needs a passing iteration before paper-trade panel can repopulate.')
        print(f'\n{msg}')
        telegram(f'⚠️ Paper-trade panel cleared: {msg}')
        return

    new_panel = fb.set_panel([it['id'] for it in selected])

    print('\n=== New panel ===')
    show_panel(new_panel)


if __name__ == '__main__':
    main()
