"""Compute the SET buy-and-hold baseline per walk-forward window.

For each window in scripts.return_gate.SPLIT_DEFS, this computes:
  - annualized_return    test-period close-to-close return, annualised
  - max_drawdown         peak-to-trough drawdown during the test period
  - win_rate             fraction of trading days with positive return
  - n_trades             1 (one position held the whole window)
  - passed               True if annualized_return > 0

Results land in data/baselines.json under key `set_buy_hold` in the same
shape as the existing `random_topk` / `rule_only` entries, so the
leaderboard can render them as a fourth section without special-casing.

Data source: yfinance `^SET.BK` (Thai SET index closing prices). The
result is cached in baselines.json; rerun when the W7 window has
rolled forward enough that the cached value is stale.

Usage:
    venv/bin/python scripts/compute_set_baseline.py            # update
    venv/bin/python scripts/compute_set_baseline.py --print    # show only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.return_gate import SPLIT_DEFS  # noqa: E402

BASELINES_PATH = BASE / 'data' / 'baselines.json'
SET_TICKER = '^SET.BK'


def _annualize(period_return: float, n_trading_days: int) -> float:
    """Annualise a cumulative period return given the number of trading
    days the position was held. 252 trading days/year for SET."""
    if n_trading_days <= 0:
        return 0.0
    years = n_trading_days / 252.0
    if years <= 0 or (1 + period_return) <= 0:
        return 0.0
    return float((1 + period_return) ** (1 / years) - 1)


def _max_drawdown(closes: pd.Series) -> float:
    """Peak-to-trough drawdown across the period."""
    if len(closes) < 2:
        return 0.0
    running_max = closes.cummax()
    dd = (closes - running_max) / running_max
    return float(abs(dd.min()))


def compute() -> dict:
    """Pull SET history and compute per-window stats."""
    # Fetch a single time series covering all windows (saves API calls).
    start = min(s for *_, s, _ in [(*x,) for x in SPLIT_DEFS]) if False else SPLIT_DEFS[0][0]
    end = SPLIT_DEFS[-1][3]
    # Pad end by 5 days in case yfinance has a settlement lag.
    end_padded = (datetime.fromisoformat(end[:10]) + pd.Timedelta(days=5)).date().isoformat()
    print(f'Fetching {SET_TICKER} {start} → {end_padded} ...', file=sys.stderr)
    df = yf.download(SET_TICKER, start=start, end=end_padded,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f'yfinance returned no data for {SET_TICKER}')
    # yfinance returns a MultiIndex column ('Close', '^SET.BK'); flatten.
    if isinstance(df.columns, pd.MultiIndex):
        closes = df[('Close', SET_TICKER)]
    else:
        closes = df['Close']
    closes = closes.dropna()
    closes.index = pd.to_datetime(closes.index)

    per_window = []
    anns: list[float] = []
    wrs: list[float] = []
    dds: list[float] = []
    n_passed = 0

    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(SPLIT_DEFS, start=1):
        mask = (closes.index >= pd.Timestamp(te_s)) & (closes.index <= pd.Timestamp(te_e))
        window_closes = closes.loc[mask]
        n = len(window_closes)
        if n < 2:
            per_window.append({
                'window_idx': idx, 'test': f'{te_s}..{te_e}',
                'n_trades': 0, 'win_rate': 0.0,
                'annualized_return': 0.0, 'max_drawdown': 0.0, 'passed': False,
            })
            continue
        c0, c1 = float(window_closes.iloc[0]), float(window_closes.iloc[-1])
        cum_return = (c1 / c0) - 1.0
        ann = _annualize(cum_return, n_trading_days=n - 1)
        daily_returns = window_closes.pct_change().dropna()
        wr = float((daily_returns > 0).mean()) if len(daily_returns) else 0.0
        dd = _max_drawdown(window_closes)
        passed = ann > 0
        if passed:
            n_passed += 1
        anns.append(ann)
        wrs.append(wr)
        dds.append(dd)
        per_window.append({
            'window_idx': idx, 'test': f'{te_s}..{te_e}',
            'n_trading_days': n,
            'cum_return': round(cum_return, 4),
            'annualized_return': round(ann, 4),
            'win_rate': round(wr, 4),
            'max_drawdown': round(dd, 4),
            'n_trades': 1,
            'passed': bool(passed),
        })

    return {
        'windows_passed': n_passed,
        'windows_total': len(SPLIT_DEFS),
        'avg_annualized_return': round(float(np.mean(anns)) if anns else 0.0, 4),
        'avg_win_rate':          round(float(np.mean(wrs)) if wrs else 0.0, 4),
        'avg_max_dd':            round(float(np.mean(dds)) if dds else 0.0, 4),
        'total_trades':          len(SPLIT_DEFS),  # one position per window
        'per_window': per_window,
        'note': f'Buy-and-hold {SET_TICKER}, close-to-close per window. '
                f'win_rate = fraction of positive trading days. '
                f'n_trades = 1 (held the whole window).',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--print', action='store_true',
                    help='Print computed baseline without updating baselines.json')
    args = ap.parse_args()

    result = compute()

    if args.print:
        print(json.dumps(result, indent=2))
        return

    # Merge into existing baselines.json.
    if BASELINES_PATH.exists():
        with open(BASELINES_PATH) as f:
            doc = json.load(f)
    else:
        doc = {}
    doc['set_buy_hold'] = result
    doc.setdefault('_metadata', {})
    doc['_metadata']['set_buy_hold_updated'] = datetime.now().isoformat(
        timespec='seconds')
    with open(BASELINES_PATH, 'w') as f:
        json.dump(doc, f, indent=2)

    # Summary
    print(f'Updated {BASELINES_PATH}:')
    print(f'  windows_passed: {result["windows_passed"]}/{result["windows_total"]}')
    print(f'  avg_ann:        {result["avg_annualized_return"]*100:+.2f}%')
    print(f'  avg_wr:         {result["avg_win_rate"]*100:.2f}%')
    print(f'  avg_max_dd:     {result["avg_max_dd"]*100:.2f}%')
    print()
    print('  per-window:')
    for w in result['per_window']:
        print(f'    W{w["window_idx"]}  {w["test"]:<24}  '
              f'cum={w.get("cum_return", 0)*100:+6.2f}%  '
              f'ann={w["annualized_return"]*100:+7.2f}%  '
              f'wr={w["win_rate"]*100:5.1f}%  '
              f'dd={w["max_drawdown"]*100:5.1f}%')


if __name__ == '__main__':
    main()
