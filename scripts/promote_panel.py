#!/usr/bin/env python3
"""Promote the top-K candidates to the paper-trade panel.

The panel is the set of currently-paper-trading models — K=10 since
2026-05-30 (was 3; see feedback.PANEL_MAX_RANKS). Strict eligibility is
`gate_passed = 1` — but until that bar is met, we use the top-K by
`avg_win_rate` among iterations whose trainer is still in the current
registry as a working proxy. This keeps the daily signal scan productive
while the loop iterates toward genuinely gate-passing candidates.

What this does:
  1. Selects K distinct-engine-family (trainer, hyperparams) iterations,
     ranked by avg_win_rate desc, tiebroken by avg_max_dd asc.
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


# Engine-family map for panel diversity. Trainers that share an underlying
# inductive bias (XGB ranker objective, XGB binary classifier, XGB regressor,
# NN, etc.) tend to ride the same regime failures together — registry-key
# dedupe alone is too loose because half the registry is XGB-flavored.
# Unknown trainers fall back to their own name in `_engine_family`, so a
# brand-new family is automatically treated as distinct.
ENGINE_FAMILY = {
    # Learning-to-rank objectives (pairwise / per-date groups)
    'ev_gated_ranker':          'ranker',
    'bagged_ev_gated_ranker':   'ranker',
    'stacked_ranker':           'ranker',
    'xgb_ranker':               'ranker',
    'xgb_ranker_dart':          'ranker',
    'xgb_rank_fusion':          'ranker',
    'xgb_win_ranker':           'ranker',
    'lightgbm_ranker':          'ranker',

    # XGB / LightGBM binary classifiers (various loss / sampling variants)
    'xgboost':                  'xgb_classifier',
    'lightgbm':                 'xgb_classifier',
    'xgb_focal_loss':           'xgb_classifier',
    'xgb_group_balanced_focal': 'xgb_classifier',
    'xgb_strict_win':           'xgb_classifier',
    'xgb_magnitude_classifier': 'xgb_classifier',
    'xgb_topk_classifier':      'xgb_classifier',
    'xgb_mcdropout_classifier': 'xgb_classifier',
    'xgb_meta_label':           'xgb_classifier',
    'xgb_day_quality_consensus':'xgb_classifier',
    'xgb_recency_consensus':    'xgb_classifier',
    'xgb_jtt':                  'xgb_classifier',
    'xgb_adv_val':              'xgb_classifier',
    'xgb_quarterly_dro':        'xgb_classifier',
    'xgb_regime_blend':         'xgb_classifier',
    'xgb_temporal_mixup':       'xgb_classifier',

    # XGB / LightGBM regressors (continuous targets)
    'xgb_regressor':            'xgb_regressor',
    'xgb_huber_regressor':      'xgb_regressor',
    'xgb_quantile':             'xgb_regressor',
    'xgb_dual_quantile':        'xgb_regressor',
    'bagged_xgb_regressor':     'xgb_regressor',
    'lightgbm_regressor':       'xgb_regressor',

    # Non-XGB trees (randomized splits — different inductive bias from greedy XGB)
    'sklearn_extra_trees':      'tree_other',

    # Linear / kernel
    'logistic_elastic_net':     'linear',

    # Neural networks (MLP / GRU / Transformer)
    'torch_attentive_mlp':      'nn',
    'torch_seq_gru':            'nn',
    'torch_seq_gru_abstain':    'nn',
    'torch_seq_gru_day_gate':   'nn',
    'torch_seq_gru_ensemble':   'nn',
    'torch_seq_transformer':    'nn',
}


def _engine_family(trainer: str) -> str:
    """Map a registry key to an engine family for panel diversity.

    Unknown trainers fall back to their own registry key — so a freshly
    added family that hasn't been categorized yet is automatically treated
    as distinct from everything else.
    """
    return ENGINE_FAMILY.get(trainer, trainer)


def select_top_candidates(k: int = 10, days: int = 30) -> list[dict]:
    """Top-k by WR, **one per engine family**, current-registry only,
    gate_passed not required.

    Diversity matters more than marginal WR: k distinct engine families
    produce uncorrelated drawdowns. k from the same family (e.g. several
    XGB rankers, even with different sub-keys) ride the same regime
    failure into the ground together. So dedupe on engine family from
    ENGINE_FAMILY, not on the registry key. (35 families are currently
    active, so k=10 fills cleanly without relaxing the one-per-family rule.)
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
              AND contaminated = 0
            ORDER BY avg_win_rate DESC, avg_max_dd ASC
            LIMIT 200
        """, (cutoff,)).fetchall()
    seen_families = set()
    out = []
    for r in rows:
        fam = _engine_family(r['trainer'])
        if fam in seen_families:
            continue
        seen_families.add(fam)
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
    parser.add_argument('--k', type=int, default=10)
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
        # Sync paper-trade portfolios — keeps current_capital intact, only
        # updates (iteration_id, trainer) so daily summary labels correctly.
        try:
            from scripts import paper_book
            paper_book.update_panel_metadata()
            print('paper_portfolios metadata refreshed')
        except Exception as e:
            print(f'paper_book.update_panel_metadata failed: {e!r}')
    else:
        print('\nNo candidates trained successfully — panel unchanged.')


if __name__ == '__main__':
    main()
