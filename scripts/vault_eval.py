#!/usr/bin/env python3
"""Vault evaluator — retrain each gate-passing iteration on VAULT_SPLIT
and store the held-out result in the vault_results table.

The 7-window gate in return_gate.py is the SEARCH surface — every iter
mutates it via HP / feature / trainer changes. VAULT_SPLIT is the
VALIDATION surface — never touched by claude_mode / train_mode. A passing
iter that survives the vault is the only thing the production signal
generator will deploy.

Run weekly via cron (~/projects/ops/cron/trading.cron, Mondays 05:00).
Idempotent: re-running overwrites the existing row for each iteration_id.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/projects/caffe-stocks'))

# Weekly vault retrain runs on GPU 1 (the inference GPU) — GPU 0 is the
# daytime-only training sweep. Set before importing return_gate/trainers so
# it wins over their GPU-0 default. setdefault → an explicit override wins.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

from models.sequence_loader import (
    load_sequences, aggregate_sequence, aggregated_feature_names,
)
from scripts.return_gate import (
    VAULT_SPLIT, evaluate_window, _min_wr_for_window,
    MAX_DD, MIN_TRADES_PER_WINDOW,
)

DB_PATH = os.path.expanduser('~/projects/caffe-stocks/data/ml-feedback.db')

# Trade cap for the vault (operator decision 2026-05-30): "20 is a max, not a
# min." The held-out window is shorter than a gate window so it can't reach
# 20 trades; we judge the first 20 chronologically instead of demanding ≥20.
VAULT_MAX_TRADES = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS vault_results (
  iteration_id INTEGER PRIMARY KEY,
  evaluated_at TEXT NOT NULL,
  vault_train_start TEXT NOT NULL,
  vault_train_end TEXT NOT NULL,
  vault_test_start TEXT NOT NULL,
  vault_test_end TEXT NOT NULL,
  n_trades INTEGER, win_rate REAL, ann_return REAL, max_dd REAL,
  min_wr_threshold REAL, min_wr_source TEXT,
  vault_passed INTEGER NOT NULL,
  fails TEXT,
  full_result TEXT,
  FOREIGN KEY (iteration_id) REFERENCES iterations(id)
);
"""


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def _passing_iters(conn):
    cur = conn.execute(
        "SELECT id, trainer, hyperparams, full_result "
        "FROM iterations WHERE gate_passed = 1 AND contaminated = 0 ORDER BY id"
    )
    rows = cur.fetchall()
    return [(rid, trainer, json.loads(hp or '{}'),
             json.loads(fr or '{}') if fr else {}) for rid, trainer, hp, fr in rows]


def _evaluate_one(iter_id, trainer, hyperparams, data):
    X, y, dates, symbols, pnl, hold_days, features, X_tab, agg_features = data
    tr_s, tr_e, te_s, te_e = VAULT_SPLIT
    print(f'[iter {iter_id}] {trainer}  train={tr_s}..{tr_e}  test={te_s}..{te_e}')
    t0 = time.time()
    # Vault uses 20 as a CAP, not a floor (operator decision 2026-05-30):
    # the held-out window is shorter than a gate window and physically can't
    # produce 20 trades, so requiring ≥20 rejected every model. Instead take
    # the first 20 trades chronologically and judge WR/DD/ann on those.
    # min_trades=1 is an evaluability check ("did it trade at all"), not a
    # statistical floor. The gate keeps MIN_TRADES_PER_WINDOW=20 unchanged.
    r = evaluate_window(
        X_tab, y, dates, symbols, pnl, hold_days, agg_features,
        tr_s, tr_e, te_s, te_e,
        trainer, hyperparams, verbose=False, X_seq=X,
        max_trades=VAULT_MAX_TRADES, min_trades=1,
    )
    elapsed = time.time() - t0
    if 'best' in r:
        b = r['best']
        flag = '✓' if r['passed'] else '✗'
        print(f'  {flag} thr={b["threshold"]} n={b["n_trades"]} '
              f'WR={b["win_rate"]:.1%} ann={b["annualized_return"]:+.1%} '
              f'DD={b["max_drawdown"]:.1%}  ({elapsed:.1f}s)')
        if not r['passed']:
            print(f'    fails: {", ".join(r.get("fails", []))}')
    else:
        print(f'  ✗ {r.get("reason", "no result")}  ({elapsed:.1f}s)')
    return r


def _write_row(conn, iter_id, result):
    tr_s, tr_e, te_s, te_e = VAULT_SPLIT
    min_wr, min_wr_src = _min_wr_for_window(te_s, te_e)
    if 'best' in result:
        b = result['best']
        n_trades = int(b['n_trades'])
        wr = float(b['win_rate'])
        ann = float(b['annualized_return'])
        dd = float(b['max_drawdown'])
        passed = 1 if result.get('passed') else 0
        fails = json.dumps(result.get('fails', []))
    else:
        n_trades = 0
        wr = 0.0
        ann = 0.0
        dd = 0.0
        passed = 0
        fails = json.dumps([result.get('reason', 'no result')])

    conn.execute(
        "INSERT OR REPLACE INTO vault_results "
        "(iteration_id, evaluated_at, vault_train_start, vault_train_end, "
        " vault_test_start, vault_test_end, n_trades, win_rate, ann_return, max_dd, "
        " min_wr_threshold, min_wr_source, vault_passed, fails, full_result) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (iter_id, datetime.utcnow().isoformat(),
         tr_s, tr_e, te_s, te_e,
         n_trades, wr, ann, dd,
         min_wr, min_wr_src, passed, fails, json.dumps(result)),
    )
    conn.commit()


def main():
    conn = _init_db()
    iters = _passing_iters(conn)
    if not iters:
        print('No gate-passing iterations to evaluate. Exiting.')
        return 0

    print(f'=== vault_eval: {len(iters)} iter(s), VAULT_SPLIT={VAULT_SPLIT} ===')
    print()
    print('Loading sequences (seq_len=20)…')
    t0 = time.time()
    X, y, dates, symbols, pnl, hold_days, features = load_sequences(seq_len=20, verbose=True)
    X_tab = aggregate_sequence(X)
    agg_features = aggregated_feature_names(features)
    print(f'Loaded: {X_tab.shape}  ({time.time()-t0:.1f}s)')
    print()

    data = (X, y, dates, symbols, pnl, hold_days, features, X_tab, agg_features)

    for iter_id, trainer, hyperparams, _full in iters:
        try:
            r = _evaluate_one(iter_id, trainer, hyperparams, data)
            _write_row(conn, iter_id, r)
        except Exception as e:
            print(f'[iter {iter_id}] ERROR: {e}')
            import traceback
            traceback.print_exc()
            conn.execute(
                "INSERT OR REPLACE INTO vault_results "
                "(iteration_id, evaluated_at, vault_train_start, vault_train_end, "
                " vault_test_start, vault_test_end, n_trades, win_rate, ann_return, max_dd, "
                " min_wr_threshold, min_wr_source, vault_passed, fails, full_result) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, 0, NULL, 'error', 0, ?, ?)",
                (iter_id, datetime.utcnow().isoformat(),
                 *VAULT_SPLIT, json.dumps([f'eval error: {e}']),
                 json.dumps({'error': str(e)})),
            )
            conn.commit()

    print()
    print('=== Summary ===')
    cur = conn.execute(
        "SELECT iteration_id, vault_passed, n_trades, win_rate, ann_return, max_dd "
        "FROM vault_results ORDER BY iteration_id"
    )
    for row in cur:
        rid, ok, n, wr, ann, dd = row
        flag = '✓' if ok else '✗'
        print(f'  iter {rid}: {flag} n={n} wr={wr:.1%} ann={ann:+.1%} dd={dd:.1%}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
