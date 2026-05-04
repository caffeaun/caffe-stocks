#!/usr/bin/env python3
"""Build dynamic ML improvement prompt from feedback history and current stats."""
import argparse
import json
import os
import sys
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

STRATEGY_CATEGORIES = [
    'label_definition', 'loss_function', 'architecture', 'training_procedure',
    'feature_engineering', 'threshold_tuning', 'sequence_length',
    'data_filtering', 'ensemble', 'other'
]

CATEGORY_DESCRIPTIONS = {
    'label_definition': 'Changing what constitutes a positive label (e.g., +5%/-3% thresholds, intraday checks)',
    'loss_function': 'Focal loss, asymmetric loss, class weighting changes',
    'architecture': 'Hidden size, layers, attention, dropout, model structure',
    'training_procedure': 'Learning rate, epochs, schedulers, data weighting, transfer learning',
    'feature_engineering': 'New features, feature interactions, feature selection/removal',
    'threshold_tuning': 'LSTM score threshold (currently 0.6), per-stock calibration',
    'sequence_length': 'Changing seq_len (currently 20)',
    'data_filtering': 'Stock filtering, time filtering, sector-specific training',
    'ensemble': 'Combining multiple models or approaches',
}


def load_feedback(feedback_path):
    if not os.path.isfile(feedback_path):
        return []
    with open(feedback_path) as f:
        return json.load(f)


def load_backtest(backtest_path):
    if not os.path.isfile(backtest_path):
        return None
    with open(backtest_path) as f:
        return json.load(f)


def load_model_info(model_path):
    if not os.path.isfile(model_path):
        return {}
    with h5py.File(model_path, 'r') as f:
        return {
            'input_size': int(f.attrs.get('input_size', 0)),
            'hidden_size': int(f.attrs.get('hidden_size', 64)),
            'num_layers': int(f.attrs.get('num_layers', 2)),
            'dropout': float(f.attrs.get('dropout', 0.3)),
            'seq_len': int(f.attrs.get('seq_len', 20)),
            'features': f.attrs.get('features', '').split(','),
            'precision': float(f.attrs.get('test_precision', 0)),
            'accuracy': float(f.attrs.get('test_accuracy', 0)),
            'recall': float(f.attrs.get('test_recall', 0)),
            'trained_at': f.attrs.get('trained_at', 'unknown'),
        }


def load_feature_coverage(db_path):
    """Check feature coverage from candles.db."""
    try:
        import sqlite3
        import pandas as pd
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query('SELECT * FROM candles LIMIT 1', conn)
        total = conn.execute('SELECT COUNT(*) FROM candles').fetchone()[0]
        coverage = {}
        for col in df.columns:
            filled = conn.execute(f'SELECT COUNT(*) FROM candles WHERE "{col}" IS NOT NULL').fetchone()[0]
            coverage[col] = filled / total if total > 0 else 0
        conn.close()
        return coverage, total
    except Exception:
        return {}, 0


def build_performance_section(backtest, model_info):
    lines = ['## Current State']
    lines.append('Best walk-forward avg WR so far: 37.5% (Entry 197). No model has passed yet.')
    lines.append('')
    lines.append('**Walk-forward: 7 fixed calendar splits** (6-month train, 4-6 month test each).')
    lines.append('Each split trains a fresh model with YOUR hyperparameters on unseen future data.')
    lines.append('Base rates vary 6-20% across splits. Alpha over base rate is the primary signal.')
    lines.append('')
    lines.append('**COST OF HIGH VOLUME — every trade costs 1.1% round-trip (฿100 per ฿10K trade):**')
    lines.append('Target: ~30-100 trades per split. Models firing 1000+ trades spray at sub-30% WR.')
    if backtest:
        lines.append(f"\nData: {backtest.get('data_range', 'unknown')}, {backtest.get('symbols', '?')} symbols, {backtest.get('total_rows', '?')} rows")
    if model_info:
        lines.append(f"\nModel arch: {model_info.get('input_size', '?')} features, seq_len={model_info.get('seq_len', '?')}, hidden={model_info.get('hidden_size', '?')}, layers={model_info.get('num_layers', '?')}, dropout={model_info.get('dropout', '?')}")
    return '\n'.join(lines)


def _summarize_feedback(feedback):
    """Use Claude to summarize feedback entries into actionable lessons.
    Cached in ~/projects/caffe-stocks/data/ml-feedback-summary.json — regenerated
    only when feedback count changes."""
    import subprocess
    HOME = os.path.expanduser('~')
    claude_bin = os.path.join(HOME, '.local/bin/claude')
    if not os.path.isfile(claude_bin):
        return None

    cache_path = os.path.join(HOME, 'trading-system/data/ml-feedback-summary.json')
    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            if cache.get('feedback_count') == len(feedback):
                return cache.get('summary', '')
        except Exception:
            pass

    # Build a compact representation of all entries with walk-forward data
    entries_with_wf = []
    for entry in feedback:
        wf = entry.get('walk_forward', {})
        if wf and wf.get('valid_splits'):
            entries_with_wf.append({
                'id': entry.get('id'),
                'category': entry.get('strategy_category', ''),
                'detail': (entry.get('strategy_detail') or '')[:120],
                'what_changed': (entry.get('what_changed') or '')[:150],
                'lessons': (entry.get('lessons') or '')[:200],
                'wf_avg_wr': wf.get('avg_win_rate', 0),
                'wf_std': wf.get('wr_std', 0),
                'wf_trades': wf.get('total_trades', 0),
                'valid_splits': wf.get('valid_splits', 0),
            })

    # Also include last 5 entries regardless (most recent context)
    for entry in feedback[-5:]:
        eid = entry.get('id')
        if not any(e['id'] == eid for e in entries_with_wf):
            entries_with_wf.append({
                'id': eid,
                'category': entry.get('strategy_category', ''),
                'detail': (entry.get('strategy_detail') or '')[:120],
                'lessons': (entry.get('lessons') or '')[:200],
                'wf_avg_wr': 0,
                'valid_splits': 0,
            })

    import json as _json
    compact = _json.dumps(entries_with_wf, indent=None)

    prompt = f"""Summarize these {len(feedback)} ML improvement feedback entries into exactly 4 sections.
Be concise (under 500 words total). No markdown headers — use **bold** for section labels.

Sections:
1. **What works** — strategies/configs that consistently get 7/7 valid splits
2. **What fails** — approaches that consistently produce 0 trades or sub-25% WR
3. **Persistent problems** — patterns that appear across many attempts
4. **Untried approaches** — reasonable ideas NOT yet attempted based on the failure patterns

Data (entries with walk-forward results + last 5):
{compact}

Output ONLY the 4 sections, no preamble."""

    try:
        env = os.environ.copy()
        env.pop('CLAUDECODE', None)
        env['PATH'] = f"{HOME}/.local/bin:{HOME}/projects/caffe-stocks/venv/bin:" + env.get('PATH', '')
        result = subprocess.run(
            [claude_bin, '-p', '--model', 'claude-opus-4-7'],
            input=prompt, capture_output=True, text=True, timeout=60, env=env, cwd=HOME,
        )
        if result.returncode == 0 and result.stdout.strip():
            summary = result.stdout.strip()
            # Cache the result
            try:
                with open(cache_path, 'w') as f:
                    json.dump({'feedback_count': len(feedback), 'summary': summary}, f)
            except Exception:
                pass
            return summary
    except Exception:
        pass
    return None


def build_history_section(feedback, max_entries=10):
    if not feedback:
        return "## Strategy History\nNo previous improvement attempts recorded."

    lines = [f'## Strategy History ({len(feedback)} total attempts)']

    # Find top entries with 7/7 valid splits (the gold standard)
    entries_7_7 = []
    for entry in feedback:
        wf = entry.get('walk_forward', {})
        if wf and wf.get('valid_splits') == 7:
            entries_7_7.append(entry)

    if entries_7_7:
        entries_7_7.sort(key=lambda e: e.get('walk_forward', {}).get('avg_win_rate', 0), reverse=True)
        lines.append('\n### Top Entries with ALL 7 Splits Valid (sorted by WF avg WR)')
        lines.append('| # | WF WR | WF std | Trades | Detail |')
        lines.append('|---|-------|--------|--------|--------|')
        for entry in entries_7_7[:8]:
            wf = entry.get('walk_forward', {})
            eid = entry.get('id', '?')
            detail = entry.get('strategy_detail', '')[:80]
            lines.append(f"| {eid} | {wf.get('avg_win_rate', 0):.1%} | {wf.get('wr_std', 0):.1%} | {wf.get('total_trades', 0)} | {detail} |")

    # Recent attempts table
    recent = feedback[-max_entries:]
    lines.append(f'\n### Last {len(recent)} Attempts')
    lines.append('| # | Category | Detail | WF avg WR | WF std | WF trades | Outcome |')
    lines.append('|---|----------|--------|-----------|--------|-----------|---------|')
    for entry in recent:
        cat = entry.get('strategy_category', 'unknown')
        detail = entry.get('strategy_detail', 'unknown')[:50]
        wf = entry.get('walk_forward', {})
        wf_avg = wf.get('avg_win_rate', 0) if wf else 0
        wf_trades = wf.get('total_trades', 0) if wf else 0
        decision = entry.get('decision', 'unknown')
        eid = entry.get('id', '?')
        wf_avg_str = f'{wf_avg:.1%}' if wf_avg else 'n/a'
        lines.append(f'| {eid} | {cat} | {detail} | {wf_avg_str} | {wf_trades} | {decision} |')

    # Summarize lessons — use Claude when feedback exceeds 20 entries
    lines.append('\n### Condensed Lessons')
    if len(feedback) > 20:
        summary = _summarize_feedback(feedback)
        if summary:
            lines.append(summary)
        else:
            # Fallback: just show last 3 lessons
            recent_lessons = [e.get('lessons', '') for e in feedback[-3:] if e.get('lessons')]
            for i, lesson in enumerate(recent_lessons, 1):
                lines.append(f"{i}. {lesson[:300]}")
    else:
        # Few entries — show raw lessons
        all_lessons = [e.get('lessons', '') for e in feedback if e.get('lessons')]
        for i, lesson in enumerate(all_lessons[-5:], 1):
            lines.append(f"{i}. {lesson[:300]}")

    return '\n'.join(lines)


def build_suggestions(feedback):
    """Guide Claude mode toward structural innovation, not parameter tuning."""
    lines = ["""## Your Role: Structural Innovation (NOT Parameter Tuning)

Sweep mode handles hyperparameter tuning (hidden size, lr, dropout, batch size).
Your job is the creative work sweep CANNOT do:

### 1. New Features — Create Information Advantage
Price/volume indicators are commoditized. Everyone sees the same RSI/MACD.
Think about data that predicts SET mid-cap moves that others don't price in:
- Foreign fund flow patterns (we have foreign_net — is 3-day consecutive buying predictive?)
- Cross-asset signals (THB/USD, gold, Thai bond yields)
- Calendar effects (month-end window dressing, dividend season, index rebalancing)
- Thai-language sentiment (Pantip, StockRadars — retail drives SET mid-caps)
- Intraday patterns (first-hour range as % of day range — institutional order flow signal)

### 2. Reformulate the Problem
The current framing (binary classification: "will stock hit +15%?") may be fundamentally wrong.
273 LSTM classification attempts peaked at 37.5% WR. Consider:
- **Ranking instead of classification** — "which stocks are most likely to win today?"
  Learning-to-rank (LambdaMART, ListNet) optimizes relative ordering, not absolute threshold.
  Even weak signal produces edge if it consistently ranks winners above losers.
- **Survival modeling** — model P(survives day k without stop-loss) × P(hits target by day k).
  Naturally encodes path dependency and asymmetric payoff.
- **Direct P&L optimization** — loss function = negative trading profit after commission.
  The model optimizes what we actually care about, not classification accuracy.
- **Regime-conditional trading** — don't predict which stock, predict WHEN to trade.
  Only enter during favorable regimes. Fewer trades, much higher WR.

### 3. Invent New Approaches
The problem has specific pathologies no standard algorithm was designed for:
- Path-dependent labels (+15% BEFORE -3% — order matters)
- Extreme class imbalance (8.8% base rate)
- Regime non-stationarity (bull/bear/sideways shift base rates 5-20%)
- Asymmetric cost (false positive = -3%, true positive = +15%)
- ฿50 minimum commission at ฿10K capital = 1.1% friction

Design something purpose-built. Examples:
- Contrastive learning for regime-invariant stock representations
- Multi-task: predict direction + magnitude + timing simultaneously
- Adversarial training: penalize features that distinguish train-era from test-era
- Online/contextual bandit: learn incrementally, no train/test split needed

### DO NOT:
- Tune hyperparameters (hidden_size, lr, dropout, batch_size) — sweep does this
- Make incremental tweaks to the existing LSTM classifier — 273 attempts prove diminishing returns
- Modify CURATED_FEATURES in feature_eng.py — breaks scaler alignment
- Modify lstm_model.py shared class — affects production"""]

    return '\n'.join(lines)


def build_feature_section(db_path, model_info):
    coverage, total = load_feature_coverage(db_path)
    if not coverage:
        return "## Feature Inventory\nCould not read feature coverage."

    active = model_info.get('features', [])
    lines = ['## Feature Inventory']
    lines.append(f"Active ({len(active)} features): {', '.join(active)}")

    # Approaching threshold
    from feature_eng import SUPPLEMENTAL_FEATURES, SECTOR_FEATURES, BREADTH_FEATURES, MIN_COVERAGE
    all_optional = list(SUPPLEMENTAL_FEATURES) + list(SECTOR_FEATURES) + list(BREADTH_FEATURES)

    # Available but not yet used by current model
    available_unused = []
    for f in all_optional:
        cov = coverage.get(f, 0)
        if cov >= MIN_COVERAGE and f not in active:
            available_unused.append(f"{f} ({cov:.0%})")
    if available_unused:
        lines.append(f"\n**NEW — available but not in current model:** {', '.join(available_unused)}")

    approaching = []
    for f in all_optional:
        cov = coverage.get(f, 0)
        if 0.1 < cov < MIN_COVERAGE and f not in active:
            approaching.append(f"{f} ({cov:.0%})")
    if approaching:
        lines.append(f"Approaching threshold ({MIN_COVERAGE:.0%}): {', '.join(approaching)}")

    below = [f for f in all_optional if coverage.get(f, 0) <= 0.1 and f not in active]
    if below:
        lines.append(f"Low coverage (not usable yet): {', '.join(below)}")

    return '\n'.join(lines)


def build_prompt(args):
    backtest = load_backtest(args.backtest_file)
    model_info = load_model_info(args.model_file)
    feedback = load_feedback(args.feedback_file)

    # Read trainer source
    trainer_source = ''
    if os.path.isfile(args.trainer_src):
        with open(args.trainer_src) as f:
            trainer_source = f.read()

    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'candles.db')

    sections = []

    # Header
    sections.append("""You are an ML engineer improving a trading LSTM model for the SET (Thai stock exchange).

## PRIMARY GOAL: Pass Walk-Forward Validation for Paper Trading

The model must beat the market base rate across ALL 7 fixed calendar splits.
This is a paper-trading gate — we need a directionally useful model, not a perfect one.

### Walk-Forward Architecture
7 fixed calendar splits (NOT expanding window). Each has 6-month train, 4-6 month test:
- Split 1: train 2023-05-01→2023-10-31, test 2023-11-01→2024-02-28
- Split 2: train 2023-07-01→2023-12-31, test 2024-01-01→2024-04-30
- Split 3: train 2023-11-01→2024-04-30, test 2024-05-01→2024-08-31
- Split 4: train 2024-03-01→2024-08-31, test 2024-09-01→2024-12-31
- Split 5: train 2024-07-01→2024-12-31, test 2025-01-01→2025-04-30
- Split 6: train 2024-11-01→2025-04-30, test 2025-05-01→2025-08-31
- Split 7: train 2025-03-01→2025-08-31, test 2025-09-01→2026-02-28
Each split trains a fresh model and evaluates on unseen future data.

### Walk-Forward Gate Thresholds (ALL 7 splits must pass)
Hard gates (block promotion):
- **WR must beat market base rate** (alpha >= 0) — no fixed WR floor
- **Avg win >= 3.0%** per split
- **Max drawdown <= 15%** per split
- **EV per trade >= -1%** per split (allows marginally negative)
- **WR std <= 15%** across splits
- **Splits with < 10 trades** are skipped (not enough signal)
- **All 7 splits must have >= 10 trades** — no free passes

Monitor-only (logged but don't block):
- Alpha >= 5% over base (⚠α if below)
- Date concentration <= 20% (⚠conc if above)
- Selectivity <= 30% (⚠sel if above)

### Best Baseline So Far (Entry 197 — only model to cover all 7 splits well)
Config: Mixup alpha=0.3, pure BCE (survival=1.0, fp=1.0), weight_decay=0,
        hidden=48, layers=1, dropout=0.4, batch=256, lr=5e-4
Result: avg WR 37.5%, std 12.2%, 5120 trades across 7 splits
Why it failed: WR below base rate in several splits, DD > 15% in some

### Trading System Rules (CANNOT be changed)
- Entry: close + 0.2% slippage
- Stop-loss: -3% from entry (hard, no exceptions)
- Target: +15% from entry
- Trailing stop: after +7% gain, trail at 50% of peak gain
- Max hold: 10 trading days
- SET friction: 0.1578% commission + 7% VAT + 0.2% slippage per side
- Commission: 1.1% round trip (฿50 min on ฿10K)
- Signal filter: LSTM score > 0.6 AND ATR > 3% AND volume_ratio > 1.5 AND RSI 30-65
- Position size: 1% risk / 3% SL = 33% of portfolio

### CRITICAL CONSTRAINTS:
- **DO NOT modify** feature_eng.py CURATED_FEATURES — changing features breaks scaler alignment
- **DO NOT modify** lstm_model.py shared class — changes affect ALL models including production
- **DO NOT tune hyperparameters** (hidden_size, lr, batch_size, dropout, etc.) — that is the sweep mode's job. Focus on structural changes: loss function, training procedure, data augmentation, architecture innovations""")

    # Current performance
    sections.append(build_performance_section(backtest, model_info))

    # Strategy history
    sections.append(build_history_section(feedback))

    # Suggestions
    sections.append(build_suggestions(feedback))

    # Feature inventory
    sections.append(build_feature_section(db_path, model_info))

    # Rules
    models_dir = os.path.dirname(args.model_file)
    feature_eng_path = os.path.join(models_dir, 'feature_eng.py')
    labels_path = os.path.join(models_dir, 'labels.py')
    lstm_model_path = os.path.join(models_dir, 'lstm_model.py')

    sections.append(f"""## Rules
1. You may create NEW scripts in {args.candidate_dir}/ (e.g., xgboost_trainer.py, ranking_model.py)
   OR edit the candidate trainer: {args.candidate_path}
2. You may ALSO edit these shared files if your strategy requires it:
   - {labels_path} — modify labeling logic (check_early_stop, label_trade)
   But do NOT modify:
   - {feature_eng_path} CURATED_FEATURES — breaks scaler alignment
   - {lstm_model_path} — affects production model
3. Make ONE structural change per run (but it may span multiple files)
4. The script must save the model to: {args.candidate_dir}/candidate_model.h5
5. The script must save the scaler to: {args.candidate_dir}/candidate_scaler.pkl
6. Use --output-dir {args.candidate_dir} to control save location
7. Do NOT touch production model at {args.model_file}
8. After editing, run your script: {args.python_path} <your_script> --output-dir {args.candidate_dir}
9. Report test metrics and explain WHY your approach should generalize across regimes
10. Use {args.python_path} (not bare python) for all execution
11. Available packages: torch, sklearn, pandas, numpy, h5py, xgboost, lightgbm, scipy
    If a package is missing, implement the core logic yourself or use an available alternative.
12. **ONE attempt per run.** Train once, report results. 30-minute time limit.
13. The gate runs walk-forward validation automatically after your training.
14. Data is in {os.path.dirname(args.model_file)}/../../data/candles.db (SQLite, table: candles)

## Current Trainer Source
```python
{trainer_source}
```

After training, report:
1. **strategy_category**: one of: {', '.join(STRATEGY_CATEGORIES)}
2. **strategy_detail**: one-line description of what you changed
3. **hypothesis**: what you expected to happen AND why you think it will generalize across time periods
4. **what_changed**: summary of code changes
5. **lessons**: what the results tell us (especially if rejected)
6. Model test metrics (precision, recall, accuracy)
7. **generalization_reasoning**: why this change should produce consistent WR across walk-forward splits

Format your report as a JSON block at the end:
```json
{{
  "strategy_category": "...",
  "strategy_detail": "...",
  "hypothesis": "...",
  "what_changed": "...",
  "lessons": "..."
}}
```

## Data/Resource Requests
If you need additional data, write to: {os.path.dirname(args.model_file)}/../data/ml-data-requests.txt
Only write this if you have a concrete, high-impact idea -- not generic wishes.""")

    return '\n\n'.join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest-file', required=True)
    parser.add_argument('--model-file', required=True)
    parser.add_argument('--feedback-file', required=True)
    parser.add_argument('--trainer-src', required=True)
    parser.add_argument('--candidate-path', required=True)
    parser.add_argument('--candidate-dir', required=True)
    parser.add_argument('--python-path', default=os.path.expanduser('~/projects/caffe-stocks/venv/bin/python'))
    parser.add_argument('--backtest-script', default=os.path.expanduser('~/projects/caffe-stocks/scripts/backtest_lstm.py'))
    parser.add_argument('--feature-eng', default=os.path.expanduser('~/projects/caffe-stocks/models/feature_eng.py'))
    args = parser.parse_args()
    print(build_prompt(args))


if __name__ == '__main__':
    main()
