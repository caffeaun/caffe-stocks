#!/usr/bin/env python3
"""Daily paper-trade summary — ONE Telegram message at 20:30 Mon-Fri.

Aggregates per-port stats (today's closes, 30-day cumulative pnl, open
positions) and posts a single block. No per-trade alerts (the user
explicitly opted out of those).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import paper_book


def _telegram(text: str) -> None:
    """Best-effort Telegram notify via the standard telegram.conf."""
    conf_paths = [
        BASE.parent / 'ops' / 'telegram' / 'telegram.conf',
        Path.home() / 'kanoonth' / 'scripts' / 'telegram.conf',
    ]
    bot_token = chat_id = None
    for p in conf_paths:
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('TELEGRAM_BOT_TOKEN='):
                        bot_token = line.split('=', 1)[1].strip().strip('"\'')
                    elif line.startswith('TELEGRAM_CHAT_ID='):
                        chat_id = line.split('=', 1)[1].strip().strip('"\'')
            break
    if not (bot_token and chat_id):
        print('Telegram config not found; printing only', file=sys.stderr)
        return
    try:
        import urllib.parse, urllib.request
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': text[:4000],
            'disable_web_page_preview': 'true',
        }).encode()
        urllib.request.urlopen(
            f'https://api.telegram.org/bot{bot_token}/sendMessage',
            data=data, timeout=10)
    except Exception as e:
        print(f'Telegram send failed: {e}', file=sys.stderr)


def format_message(summary: dict) -> str:
    lines = [f'📒 Paper trades — {summary["date"]}']
    for p in summary['portfolios']:
        trainer = p['trainer'] or '—'
        cap = p['current_capital']
        total_pct = p['pnl_pct_total']
        wr30 = p['wr_30d']
        n30 = p['recent_closed']
        lines.append('')
        lines.append(
            f"Port {p['rank']} ({trainer})  cap ฿{cap:,.0f}  "
            f"{total_pct:+.2%}  ({n30} trades 30d, WR {wr30:.0%})"
        )
        # Today's closes
        if p['today_closed']:
            lines.append(f"  closed today: {p['today_closed']} trades, "
                          f"pnl ฿{p['pnl_thb_today']:+,.0f}")
        # Open positions
        if p['opens']:
            for o in p['opens']:
                lines.append(
                    f"  open: {o['symbol'].replace('.BK','')}  "
                    f"entry ฿{o['entry_price']:.2f}  "
                    f"watermark ฿{o['high_watermark']:.2f}  "
                    f"(since {o['entry_date'][:10]})"
                )
        else:
            lines.append('  open: —')
    return '\n'.join(lines)


def main():
    today = datetime.now().date().isoformat()
    summary = paper_book.daily_summary(today)
    msg = format_message(summary)
    print(msg)
    _telegram(msg)


if __name__ == '__main__':
    main()
