#!/usr/bin/env python3
"""Promote the top-3 candidates to the paper-trade panel.

The whitepaper-defined panel is "the 3 currently-paper-trading models"
(docs/ml-loop.md §promotion). Strict eligibility is `gate_passed = 1` —
but until that bar is met, we use the top-3 by `avg_win_rate` among
iterations whose trainer is still in the current registry as a working
proxy. This keeps the daily signal scan productive while the loop
iterates toward genuinely gate-passing candidates.

What this does:
  1. Selects 3 distinct (trainer, hyperparams) iterations, ranked by
     avg_win_rate desc, tiebroken by avg_max_dd asc.
  2. For each, re-fits the trainer on the FULL dataset (no held-out
     split — production model). Saves to models/panel/rank{N}/.
  3. Updates production_panel rows so signal_generator.py finds them.

Run by hand or weekly cron. Idempotent — re-running just rebuilds.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

import numpy as np
from sklearn.preprocessing import RobustScaler

from models.sequence_loader import load_sequences, aggregate_sequence, aggregated_feature_names
from models.trainers import get_trainer, TRAINERS
from scripts import feedback as fb

PANEL_DIR = BASE / 'models' / 'panel'


def select_top_candidates(k: int = 3, days: int = 30) -> list[dict]:
    """Top-k by WR, current-registry trainers only, gate_passed not required.

    Dedupes on (trainer, hyperparams) so the same model with the same HPs
    can't take two slots.
    """
    fb.init_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    valid_trainers = ','.join(f"'{t}'" for t in sorted(TRAINERS.keys()))
    with fb.get_conn() as conn:
        rows = conn.execute(f"""
            SELECT id, trainer, hyperparams, avg_win_rate,
                   avg_annualized_return, avg_max_dd, total_trades,
                   gate_passed
            FROM iterations
            WHERE finished_at >= ?
              AND trainer IN ({valid_trainers})
              AND total_trades > 0
            ORDER BY avg_win_rate DESC, avg_max_dd ASC
            LIMIT 50
        """, (cutoff,)).fetchall()
    seen = set()
    out = []
    for r in rows:
        sig = (r['trainer'], r['hyperparams'])
        if sig in seen:
            continue
        seen.add(sig)
        out.append(dict(r))
        if len(out) == k:
            break
    return out


def fit_and_save(candidate: dict, X_tab: np.ndarray, y: np.ndarray,
                  pnl: np.ndarray, dates: np.ndarray, agg_features: list[str],
                  out_dir: Path) -> None:
    """Train on FULL data; save artifact + scaler + metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)

    hp = json.loads(candidate['hyperparams'])
    trainer = get_trainer(candidate['trainer'], **hp)

    # Standardize features the same way the gate does — fit on full data
    # since this is a production model, not a walk-forward eval.
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_tab)

    # No held-out val split — fit on all data. Pass dummy val arrays so
    # trainers that expect early-stopping don't crash.
    n = len(X_scaled)
    val_size = max(64, n // 10)
    val_idx = np.random.RandomState(42).choice(n, size=val_size, replace=False)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True

    trainer.fit(
        X_scaled[~val_mask], y[~val_mask],
        X_scaled[val_mask],  y[val_mask],
        verbose=False,
        pnl_train=pnl[~val_mask] if pnl is not None else None,
        pnl_val=pnl[val_mask] if pnl is not None else None,
        dates_train=dates[~val_mask] if dates is not None else None,
        dates_val=dates[val_mask] if dates is not None else None,
    )

    trainer.save(str(out_dir), extra={
        'iteration_id': candidate['id'],
        'trainer': candidate['trainer'],
        'hyperparams': hp,
        'avg_win_rate': candidate['avg_win_rate'],
        'avg_annualized_return': candidate['avg_annualized_return'],
        'features': list(agg_features),
        'promoted_at': datetime.now().isoformat(),
    })
    # Save scaler alongside the trainer's own artifacts
    import joblib
    joblib.dump(scaler, out_dir / 'scaler.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    candidates = select_top_candidates(k=args.k, days=args.days)
    if not candidates:
        print(f'No eligible candidates in last {args.days} days. Panel unchanged.')
        return

    print(f'\n=== Top {len(candidates)} candidates (WR-ranked, current trainers, last {args.days}d) ===')
    for i, c in enumerate(candidates, 1):
        print(f"  rank {i}  iter#{c['id']:4d}  {c['trainer']:24s}  "
              f"WR={c['avg_win_rate']:.1%}  ann={c['avg_annualized_return']:+.1%}  "
              f"DD={c['avg_max_dd']:.1%}  gate={c['gate_passed']}")

    if args.dry_run:
        print('\n--dry-run set — not training/saving.')
        return

    print('\n=== Loading sequences ===')
    t0 = time.time()
    X, y, dates, symbols, pnl, hold_days, features = load_sequences(
        seq_len=20, verbose=False)
    X_tab = aggregate_sequence(X)
    agg_features = aggregated_feature_names(features)
    print(f'  X_tab shape: {X_tab.shape}  ({time.time()-t0:.1f}s)')

    # Wipe + rebuild the panel directory tree
    if PANEL_DIR.exists():
        shutil.rmtree(PANEL_DIR)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    iter_ids = []
    for i, c in enumerate(candidates, 1):
        out_dir = PANEL_DIR / f'rank{i}'
        print(f"\n=== Training rank {i}: {c['trainer']} (iter#{c['id']}) ===")
        t = time.time()
        try:
            fit_and_save(c, X_tab, y, pnl, dates, agg_features, out_dir)
            print(f'  saved -> {out_dir}  ({time.time()-t:.1f}s)')
            iter_ids.append(c['id'])
        except Exception as e:
            print(f'  FAILED: {e!r}')

    if iter_ids:
        new_panel = fb.set_panel(iter_ids)
        print(f'\n=== production_panel updated ({len(new_panel)} rows) ===')
        for row in new_panel:
            print(f"  rank {row['rank']}  iter#{row['iteration_id']}  {row['trainer']}")
    else:
        print('\nNo candidates trained successfully — panel unchanged.')


if __name__ == '__main__':
    main()
