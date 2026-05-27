#!/usr/bin/env python3
"""Model leaderboard — quick view of what's working in ml-feedback.db.

Four sections, ordered by what's actionable:
  1. CANDIDATES  — gate_passed = 1, sorted by avg_ann desc. These are the
     ones promote_panel will consider this Sunday.
  2. NEAR-PASSES — gate_passed = 0 but windows_passed >= 5, sorted by ann
     desc. One more window from the gate.
  3. BEST PER ENGINE FAMILY (last 30d) — single best iter per family.
     Shows whether each inductive bias is competitive at all.
  4. CURRENT PRODUCTION PANEL — the 3 trainers signaling live.

Usage:
  scripts/leaderboard.py                # print to stdout
  scripts/leaderboard.py --top 15       # bigger candidate list
  scripts/leaderboard.py --days 7       # tighter family lookback
  scripts/leaderboard.py --telegram     # also post to Telegram
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import feedback as fb
from scripts.promote_panel import _engine_family

BASELINES_PATH = BASE / 'data' / 'baselines.json'


def _load_baselines() -> dict:
    """Return the baselines dict, or {} if the file is missing/broken.
    The leaderboard renders whatever keys it finds (random_topk, rule_only,
    set_buy_hold, …) so future baselines need no code change here."""
    if not BASELINES_PATH.exists():
        return {}
    try:
        import json
        with open(BASELINES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _set_baseline_ann() -> float | None:
    """The SET buy-and-hold avg_ann across 7 windows; used as the floor
    for the ★ marker on trainer rows. None if unavailable."""
    b = _load_baselines().get('set_buy_hold') or {}
    if 'avg_annualized_return' not in b:
        return None
    try:
        return float(b['avg_annualized_return'])
    except (TypeError, ValueError):
        return None


def _fmt_pct(x: float | None, sign: bool = False) -> str:
    """Format a fractional value as a percentage string. Caps display at
    ±999.9% so historical pre-clamp ranker artifacts (10^12% ann_return)
    don't blow out the column width."""
    if x is None:
        return '   —   '
    pct = x * 100
    if pct > 999.9:
        return '>+999%' if sign else '>999%'
    if pct < -999.9:
        return '<-999%' if sign else '-999+'
    fmt = '{:+.1f}%' if sign else '{:.1f}%'
    return fmt.format(pct)


def _fmt_date(s: str | None) -> str:
    if not s:
        return '   —   '
    return str(s)[:10]


def _vault_status(iter_id: int) -> str:
    """Return vault status for an iteration: '✓' / '✗' / '—' (not evaluated)."""
    try:
        with fb.get_conn() as conn:
            row = conn.execute(
                "SELECT vault_passed FROM vault_results WHERE iteration_id = ?",
                (iter_id,),
            ).fetchone()
    except Exception:
        return '—'
    if not row:
        return '—'
    return '✓' if row[0] else '✗'


def _row(rank, iter_id, mode, trainer, wr, ann, dd, n_tr, wp, date,
         family=None, gate=False, show_vault=False) -> str:
    flag = '🏆' if gate else '  '
    fam = f'{family:16}' if family is not None else ''
    # ★ if avg_ann beats the SET buy-and-hold avg_ann floor; else blank.
    set_floor = _set_baseline_ann()
    if ann is not None and set_floor is not None and float(ann) > set_floor:
        beats = '★'
    else:
        beats = ' '
    vault_col = f'  {_vault_status(iter_id)}' if show_vault else ''
    return (f'{flag} {rank:>3}  #{iter_id:<4}  {mode:6}  '
            f'{trainer[:22]:22}  {fam}'
            f'{_fmt_pct(wr):>6}  {_fmt_pct(ann, sign=True):>8}{beats} '
            f'{_fmt_pct(dd):>6}  '
            f'{n_tr:>4}  {wp}/7  {_fmt_date(date)}{vault_col}')


def section_candidates(top: int) -> list[str]:
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute("""
            SELECT id, mode, trainer, avg_win_rate, avg_annualized_return,
                   avg_max_dd, total_trades, windows_passed, finished_at
            FROM iterations
            WHERE gate_passed = 1
            ORDER BY avg_annualized_return DESC
            LIMIT ?
        """, (top,)).fetchall()
    out = ['', '=' * 110,
           f'CANDIDATES  ({len(rows)} gate-passers, sorted by avg_ann desc)',
           '=' * 110]
    if not rows:
        out.append('   (none yet — no iteration has cleared the 6/7 gate)')
        return out
    out.append('     #     iter  mode    trainer                  '
               '   wr      ann       dd     n_tr  wp   date        vault')
    for i, r in enumerate(rows, 1):
        out.append(_row(i, r['id'], r['mode'], r['trainer'],
                        r['avg_win_rate'], r['avg_annualized_return'],
                        r['avg_max_dd'], r['total_trades'],
                        r['windows_passed'], r['finished_at'],
                        gate=True, show_vault=True))
    return out


def section_near_passes(top: int) -> list[str]:
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute("""
            SELECT id, mode, trainer, avg_win_rate, avg_annualized_return,
                   avg_max_dd, total_trades, windows_passed, finished_at
            FROM iterations
            WHERE gate_passed = 0 AND windows_passed >= 5
              AND total_trades > 50
            ORDER BY windows_passed DESC, avg_annualized_return DESC
            LIMIT ?
        """, (top,)).fetchall()
    out = ['', '=' * 110,
           f'NEAR-PASSES  (windows_passed >= 5, gate_passed = 0; one short of the gate)',
           '=' * 110]
    if not rows:
        out.append('   (none — no 5/7 or 6/7 misses on file)')
        return out
    out.append('     #     iter  mode    trainer                  '
               '   wr      ann       dd     n_tr  wp   date')
    for i, r in enumerate(rows, 1):
        out.append(_row(i, r['id'], r['mode'], r['trainer'],
                        r['avg_win_rate'], r['avg_annualized_return'],
                        r['avg_max_dd'], r['total_trades'],
                        r['windows_passed'], r['finished_at']))
    return out


def section_best_per_family(days: int) -> list[str]:
    fb.init_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with fb.get_conn() as conn:
        rows = conn.execute("""
            SELECT id, mode, trainer, avg_win_rate, avg_annualized_return,
                   avg_max_dd, total_trades, windows_passed, finished_at
            FROM iterations
            WHERE finished_at >= ? AND total_trades > 0
            ORDER BY windows_passed DESC, avg_annualized_return DESC
        """, (cutoff,)).fetchall()
    out = ['', '=' * 110,
           f'BEST PER ENGINE FAMILY  (last {days}d; one row per family)',
           '=' * 110]
    if not rows:
        out.append(f'   (no iterations in last {days} days)')
        return out
    out.append('     #     iter  mode    trainer                  '
               'family            wr      ann       dd     n_tr  wp   date')
    seen: set[str] = set()
    i = 0
    for r in rows:
        fam = _engine_family(r['trainer'])
        if fam in seen:
            continue
        seen.add(fam)
        i += 1
        out.append(_row(i, r['id'], r['mode'], r['trainer'],
                        r['avg_win_rate'], r['avg_annualized_return'],
                        r['avg_max_dd'], r['total_trades'],
                        r['windows_passed'], r['finished_at'],
                        family=fam))
    return out


def section_panel() -> list[str]:
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute(
            "SELECT rank, iteration_id, promoted_at "
            "FROM production_panel ORDER BY rank"
        ).fetchall()
    out = ['', '=' * 110,
           'CURRENT PRODUCTION PANEL  (the 3 trainers signaling live)',
           '=' * 110]
    if not rows:
        out.append('   (panel empty)')
        return out
    for row in rows:
        with fb.get_conn() as conn:
            it = conn.execute(
                "SELECT trainer, avg_win_rate, avg_annualized_return, "
                "avg_max_dd, windows_passed FROM iterations WHERE id = ?",
                (row['iteration_id'],)
            ).fetchone()
        if not it:
            out.append(f'   rank {row["rank"]}  iter#{row["iteration_id"]} '
                       f'(promoted {_fmt_date(row["promoted_at"])}) — '
                       f'iteration row missing')
            continue
        out.append(
            f'   rank {row["rank"]}  iter#{row["iteration_id"]}  '
            f'{it["trainer"]:22}  '
            f'family={_engine_family(it["trainer"]):16}  '
            f'wr={_fmt_pct(it["avg_win_rate"]):>6}  '
            f'ann={_fmt_pct(it["avg_annualized_return"], sign=True):>8}  '
            f'wp={it["windows_passed"]}/7  '
            f'promoted {_fmt_date(row["promoted_at"])}')
    return out


def section_baselines() -> list[str]:
    """Render the non-trainer baselines (SET buy-hold, random_topk, rule_only)
    as a floor block above the trainer sections. Anything that scores below
    these isn't beating dumb strategies."""
    b = _load_baselines()
    out = ['', '=' * 110,
           'BASELINES  (floors to beat — trainer ★ = avg_ann above SET buy-hold)',
           '=' * 110]
    rows = []
    for key, label in (
        ('set_buy_hold', 'SET buy & hold'),
        ('random_topk',  'random top-K'),
        ('rule_only',    'volume/RSI rule'),
    ):
        d = b.get(key)
        if not isinstance(d, dict):
            continue
        rows.append((label, d))
    if not rows:
        out.append('   (baselines file missing — '
                   'run scripts/compute_set_baseline.py)')
        return out
    out.append('     baseline                 wr      ann       dd     wp')
    for label, d in rows:
        wp = d.get('windows_passed', 0)
        wt = d.get('windows_total', 7)
        out.append(
            f'     {label:22}  '
            f'{_fmt_pct(d.get("avg_win_rate")):>6}  '
            f'{_fmt_pct(d.get("avg_annualized_return"), sign=True):>8}  '
            f'{_fmt_pct(d.get("avg_max_dd")):>6}  '
            f'{wp}/{wt}'
        )
    return out


def render(top: int, days: int) -> str:
    parts: list[str] = []
    parts += section_baselines()
    parts += section_candidates(top)
    parts += section_near_passes(top)
    parts += section_best_per_family(days)
    parts += section_panel()
    parts.append('')
    return '\n'.join(parts)


def telegram(text: str):
    """Best-effort Telegram delivery, plain text. Strips emojis from rank
    flag and ASCII-frames the table so Markdown doesn't try to parse it."""
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
        print('telegram config not found; skipping', file=sys.stderr)
        return
    try:
        import urllib.request, urllib.parse
        body = f'```\n{text[:3700]}\n```'
        data = urllib.parse.urlencode({
            'chat_id': chat_id,
            'text': body,
            'parse_mode': 'MarkdownV2',
            'disable_web_page_preview': 'true',
        }).encode()
        urllib.request.urlopen(
            f'https://api.telegram.org/bot{bot_token}/sendMessage',
            data=data, timeout=10)
    except Exception as e:
        print(f'Telegram delivery failed: {e}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=10,
                        help='How many rows in candidates/near-passes')
    parser.add_argument('--days', type=int, default=30,
                        help='Lookback window for per-family best')
    parser.add_argument('--telegram', action='store_true',
                        help='Also post to Telegram')
    args = parser.parse_args()
    text = render(top=args.top, days=args.days)
    print(text)
    if args.telegram:
        telegram(text)


if __name__ == '__main__':
    main()
