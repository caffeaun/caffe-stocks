#!/usr/bin/env python3
"""Build the prompt that claude_mode hands to `claude -p`.

Stitches together: mission excerpt, strategy goal, current pipeline state,
recent iterations from the feedback DB, open data requests, and a
structured output spec. Writes to stdout.

Used by claude_mode.py — pipe its output into claude -p.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from textwrap import indent

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import feedback as fb


# ----- excerpt helpers --------------------------------------------------- #

def _read_section(path: Path, header: str, max_lines: int = 60) -> str:
    """Read a markdown section starting at `## header` until the next `## `.
    Truncates to max_lines."""
    if not path.exists():
        return f'(section not found: {path}::{header})'
    text = path.read_text()
    lines = text.split('\n')
    out = []
    inside = False
    h = f'## {header}'
    for line in lines:
        if line.strip().startswith(h):
            inside = True
            out.append(line)
            continue
        if inside and line.startswith('## ') and not line.strip().startswith(h):
            break
        if inside:
            out.append(line)
        if len(out) >= max_lines:
            break
    return '\n'.join(out).strip() or f'(empty section: {header})'


def _list_trainers() -> list[str]:
    from models.trainers import TRAINERS
    return list(TRAINERS.keys())


def _features() -> list[str]:
    try:
        from models.feature_eng import CURATED_FEATURES
        return list(CURATED_FEATURES)
    except Exception:
        return []


def _open_data_requests() -> list[str]:
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute(
            "SELECT request_text, requested_at FROM data_requests "
            "WHERE fulfilled = 0 ORDER BY requested_at DESC LIMIT 5"
        ).fetchall()
    return [f"{r['requested_at'][:10]} — {r['request_text']}" for r in rows]


def _recent_iterations(k: int = 8) -> list[dict]:
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, finished_at, mode, trainer, hyperparams, "
            "       gate_passed, windows_passed, windows_total, "
            "       avg_annualized_return, avg_win_rate, avg_max_dd, "
            "       total_trades, hypothesis, lessons "
            "FROM iterations ORDER BY id DESC LIMIT ?", (k,)
        ).fetchall()
    return [dict(r) for r in rows]


def _summarize_iteration(it: dict) -> str:
    hp = json.loads(it['hyperparams']) if it['hyperparams'] else {}
    hp_str = ', '.join(f'{k}={v}' for k, v in hp.items() if k in
                       ('num_leaves', 'max_depth', 'learning_rate',
                        'n_estimators', 'min_child_weight', 'pos_class_weight',
                        'rank', 'alpha', 'hidden_size', 'num_layers'))
    flag = '✓' if it['gate_passed'] else '✗'
    line1 = (f"#{it['id']} [{it['mode']:6}] {it['trainer']:10} {flag} "
             f"wp={it['windows_passed']}/{it['windows_total']} "
             f"ann={it['avg_annualized_return']:+.1%} "
             f"wr={it['avg_win_rate']:.1%} "
             f"dd={it['avg_max_dd']:.1%} "
             f"trades={it['total_trades']}")
    line2 = f"   hp: {hp_str[:120]}" if hp_str else ''
    line3 = f"   hypothesis: {it['hypothesis']}" if it.get('hypothesis') else ''
    line4 = f"   lessons:    {it['lessons']}" if it.get('lessons') else ''
    return '\n'.join(s for s in [line1, line2, line3, line4] if s)


# ----- main prompt ------------------------------------------------------- #

PROMPT_TEMPLATE = """You are running ONE iteration of the Caffe-Stocks ML improvement loop in CLAUDE MODE.

Your job is to make ONE structural change to the pipeline based on the feedback below, then validate it via the walk-forward gate. The cron schedule guarantees you have at most 30 minutes of wall-time. After 30 minutes the process is killed.

==================================================================
1. MISSION (read-only — what you own and what you don't)
==================================================================
{mission}

==================================================================
2. STRATEGY GOAL (read-only constraints — see ~/projects/caffe-stocks/docs/ml-training.md for full spec)
==================================================================
GOAL: 50% gross annual return on a 10k THB base, repeatable, withdrawing the gain to SCB at year-end.
- Stop -3%, target +15%, trailing trigger +7% / floor 50%, max-hold 10 days
- Friction model in models/labels.py (COMMISSION_PCT)
- Universe: any THB-denominated SET instrument (SET / mai / ETFs / DRs / REITs)
- Walk-forward gate: 7 calendar splits in scripts/return_gate.py
- Gate criteria per window: ann_return >= 50%, max_dd <= 25%, n_trades >= 5, wr >= 30%
- Pass overall: >= 80% of windows must meet criteria

==================================================================
3. PIPELINE STATE
==================================================================
Trainers currently registered (models/trainers.py): {trainers}
Curated features ({n_features} total): {features_preview}
Latest feedback DB stats: {db_stats}

==================================================================
4. RECENT ITERATIONS ({n_recent} latest, newest first)
==================================================================
{recent_iterations}

==================================================================
5. OPEN DATA REQUESTS
==================================================================
{open_requests}

==================================================================
6. YOUR JOB — pick ONE of these to do this iteration
==================================================================

A. Add a new trainer to models/trainers.py (any algorithm family — trees, LSTM/Transformer, LoRA/QLoRA on a foundation model, RL, ensemble, anything from a recent paper). Then add the search space in models/search_spaces.py and a sample HP config to test.
B. Modify models/feature_eng.py — add/drop features. Don't change CURATED_FEATURES casually; the comment block explains why each feature is there.
C. Modify models/labels.py — change the label definition or friction model. The current COMMISSION_PCT (1.1%) is conservative — real BLS rate is closer to 0.4%.
D. Modify models/sequence_loader.py — change data prep, add augmentation.
E. Modify scripts/return_gate.py gate criteria — only if you can defend the change against ml-training.md §9.
F. Submit a data request via scripts/feedback.py log_data_request — when no code change can help (e.g., need more historical data, need foreign-flow data, etc.).

You may use WebSearch / WebFetch to research recent ML techniques (post-2024 papers, GitHub repos). The list of "trainer families in scope" in ml-loop.md is a starting point, not a cage.

==================================================================
7. WORKFLOW (must follow)
==================================================================

1. THINK — read the recent iterations carefully. What pattern of failure is the loop hitting? Don't repeat what's already in the lessons.
2. RESEARCH (optional) — WebSearch / WebFetch for recent techniques relevant to your hypothesis.
3. CODE — make ONE coherent change. Don't change unrelated things.
4. RUN THE GATE — use the new trainer name if you added one:
   ```
   ~/projects/caffe-stocks/venv/bin/python scripts/return_gate.py --model-type <trainer>
   ```
   Or if you only changed labels/features, use any existing trainer to test impact.
5. RECORD — at the end of your output, emit a JSON report block (see format below).

If the gate fails by a wide margin (avg_ann < -30%), CONSIDER reverting via `git checkout HEAD -- <files>`. If your change is structural and you expect it needs HP tuning, you can leave it — train mode will tune the HPs over the next hours.

==================================================================
8. OUTPUT FORMAT — your final message MUST end with this JSON block
==================================================================
```json
{{
  "trainer": "<registered trainer name used for the gate>",
  "hyperparams": {{ ... }},
  "code_changes": [
    "models/trainers.py: added LSTMTrainer (PyTorch, 2-layer, dropout 0.3)",
    "models/search_spaces.py: lstm entry"
  ],
  "hypothesis": "<one sentence — what change, why you expect it to help>",
  "backbone": "<for PEFT only: which pre-trained model, e.g. 'lag-llama-1b'>",
  "gate_result": {{
    "gate_passed": false,
    "windows_passed": 1,
    "windows_total": 7,
    "avg_annualized_return": -0.05,
    "avg_win_rate": 0.32,
    "avg_max_dd": 0.12,
    "total_trades": 38,
    "results": [<full per-window list from return_gate output>]
  }},
  "lessons": "<1-2 sentences — what worked, what didn't, what to try next>",
  "data_request": "<empty string, or a free-form request to be sent to the operator>",
  "git_action": "kept" | "reverted"
}}
```

Be precise. The JSON is parsed by claude_mode.py and written to the SQLite feedback DB. Bad JSON = lost iteration.

==================================================================
9. NON-NEGOTIABLES
==================================================================
- Do NOT change ~/projects/caffe-stocks/docs/ml-training.md or docs/ml-loop.md (the constitution).
- Do NOT touch ~/projects/caffe-stocks/data/ (production runtime state).
- Do NOT delete iteration_windows or iterations rows from the DB.
- Do NOT install new Python packages without first checking if a comparable one exists.
- Do NOT skip the JSON output block.

Proceed.
"""


def build_prompt() -> str:
    mission = _read_section(BASE / 'docs' / 'ml-loop.md', 'Two modes — sharp boundary', max_lines=80)
    trainers = ', '.join(_list_trainers())
    features = _features()
    features_preview = ', '.join(features[:8]) + (f', ... +{len(features)-8} more' if len(features) > 8 else '')

    db_stats = fb.stats()
    db_stats_str = (f"total={db_stats['total']} "
                    f"passed={db_stats['passed']} "
                    f"by_trainer={db_stats['by_trainer']} "
                    f"streak={db_stats['consecutive_failures']}")

    iters = _recent_iterations(k=8)
    if iters:
        recent_block = '\n'.join(_summarize_iteration(it) for it in iters)
    else:
        recent_block = '(no prior iterations — this is the first claude-mode run)'

    requests = _open_data_requests()
    requests_block = '\n'.join(f'- {r}' for r in requests) if requests else '(none)'

    return PROMPT_TEMPLATE.format(
        mission=indent(mission, '  '),
        trainers=trainers,
        n_features=len(features),
        features_preview=features_preview,
        db_stats=db_stats_str,
        n_recent=len(iters),
        recent_iterations=recent_block,
        open_requests=requests_block,
    )


def main():
    print(build_prompt())


if __name__ == '__main__':
    main()
