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

from scripts import active_case, feedback as fb


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


def _brief_directive() -> str:
    """Compute the binding directive about which trainer claude must use.
    Reads latest.md to extract the brief's registry_key, queries the
    feedback DB to determine if the brief is exhausted (≥2 claude attempts
    with best wp < 3), and emits one of three blocks for §5b: MUST-EXECUTE,
    EXHAUSTED-MAY-PIVOT, or NO-BRIEF."""
    path = BASE / 'data' / 'research_briefs' / 'latest.md'
    if not path.exists():
        return (
            '==================================================================\n'
            'BRIEF DIRECTIVE — NO BRIEF YET\n'
            '==================================================================\n'
            'No research brief on disk. The daily 03:00 research_mode has '
            "not produced one yet. For THIS iteration you have full discretion "
            'per §6.B.1–B.4. Set `exhausted_brief: false` in the JSON output.\n\n'
        )

    text = path.read_text(errors='replace')
    # Pull registry_key from "### Registry registration" code block
    import re as _re
    m = _re.search(
        r'### Registry registration\s*\n```python\s*\n(.*?)\n\s*```',
        text, _re.DOTALL,
    )
    registry_key = None
    if m:
        m2 = _re.search(r"['\"](\w+)['\"]\s*:\s*\w+", m.group(1))
        if m2:
            registry_key = m2.group(1)
    technique_m = _re.search(r'^## Recommended technique:\s*(.+?)$', text, _re.MULTILINE)
    technique = technique_m.group(1).strip() if technique_m else 'UNKNOWN'

    if not registry_key:
        return (
            '==================================================================\n'
            'BRIEF DIRECTIVE — BRIEF UNPARSEABLE\n'
            '==================================================================\n'
            f'A brief exists at data/research_briefs/latest.md but its '
            f'registry registration code block could not be parsed. Falling '
            f'back to full discretion per §6.B.1–B.4. Set `exhausted_brief: '
            f'false` in the JSON output.\n\n'
        )

    # Exhaustion check: count claude-mode iters using this registry_key.
    # Bumped 2026-05-27 from (≥2 attempts, best wp <3) → (≥5 attempts,
    # best wp <5) so the brief gets a real depth-search before Claude is
    # allowed to pivot. Last-20-iter trace showed Claude pivoting in 2
    # iters, leaving 14 different trainers across the most recent 20 —
    # breadth-search by curiosity, not depth-search of a winning family.
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, windows_passed FROM iterations "
            "WHERE mode='claude' AND trainer = ? AND contaminated = 0 "
            "ORDER BY id DESC LIMIT 10",
            (registry_key,),
        ).fetchall()
    attempts = len(rows)
    best_wp = max((r['windows_passed'] for r in rows), default=0)
    EXHAUSTED_MIN_ATTEMPTS = 5
    EXHAUSTED_WP_THRESHOLD = 5
    # AMP (Ablation Must Precede Pivot): if any prior attempt of this
    # registry_key got close to passing, force HP ablation even if other
    # exhaustion conditions look exhausted. A 4/7 result is the strongest
    # signal of latent potential; lock the family until depth-searched.
    AMP_BEST_WP = 4

    if best_wp >= AMP_BEST_WP:
        return (
            '==================================================================\n'
            'BRIEF DIRECTIVE — ABLATION MUST PRECEDE PIVOT (AMP)\n'
            '==================================================================\n'
            f'The brief recommends `{registry_key}` ({technique}). A prior '
            f'attempt reached windows_passed={best_wp} ({attempts} attempts '
            f'total) — strong enough to demand depth-search before any pivot. '
            f'You MUST execute this trainer this iteration with an HP '
            f'ablation (one or two parameters changed from the best prior '
            f'config, NOT a new family). In your JSON output:\n'
            f'  - Set `trainer: "{registry_key}"`\n'
            f'  - Set `exhausted_brief: false`\n'
            f'  - Hypothesis must name the SPECIFIC HP being ablated and '
            f'cite the prior iter whose result you are extending.\n'
            f'  - Pivoting to a different trainer or family will be rejected '
            f'at parse time (git revert + deviation log).\n\n'
        )

    if attempts >= EXHAUSTED_MIN_ATTEMPTS and best_wp < EXHAUSTED_WP_THRESHOLD:
        return (
            '==================================================================\n'
            'BRIEF DIRECTIVE — EXHAUSTED, PIVOT ALLOWED\n'
            '==================================================================\n'
            f'The brief recommends `{registry_key}` ({technique}), but it has '
            f'been attempted {attempts} times in claude-mode with best '
            f'windows_passed={best_wp} (threshold: <{EXHAUSTED_WP_THRESHOLD}). '
            f'You MAY pivot to a structurally different trainer per §6.B.0(b). '
            f'In your JSON output:\n'
            f'  - Set `exhausted_brief: true`\n'
            f'  - Set `trainer` to a registry key from §3 that is NOT '
            f'`{registry_key}` and NOT in the same engine family (see '
            f'§6.B.2 family list)\n'
            f'  - Hypothesis must explain why the new family addresses the '
            f'failure mode that {registry_key} hit.\n\n'
        )

    return (
        '==================================================================\n'
        'BRIEF DIRECTIVE — MUST EXECUTE\n'
        '==================================================================\n'
        f'The brief recommends `{registry_key}` ({technique}). It has been '
        f'attempted {attempts} times in claude-mode (best wp={best_wp}; '
        f'threshold for exhaustion: ≥{EXHAUSTED_MIN_ATTEMPTS} attempts AND '
        f'best wp <{EXHAUSTED_WP_THRESHOLD}). You MUST execute it this '
        f'iteration. In your JSON output:\n'
        f'  - Set `trainer: "{registry_key}"`\n'
        f'  - Set `exhausted_brief: false`\n'
        f'  - Run the gate via `--model-type {registry_key}` with default HPs '
        f'(or minor nudges from search_spaces.py).\n'
        f'  - Any deviation (custom variant, audit-driven pivot, etc.) will '
        f'be rejected at parse time: your code changes are reverted via `git '
        f'checkout HEAD --`, the iteration is logged as a deviation, '
        f'active_case.json is NOT updated. If the trainer truly cannot be '
        f'executed (e.g., import error), emit `exhausted_brief: true` with '
        f'an explanation in `lessons` so research_mode picks a different '
        f'family tomorrow.\n\n'
    )


def _research_brief(max_lines: int = 80) -> str:
    """Return the truncated body of data/research_briefs/latest.md.

    Truncated to ``max_lines`` so the brief doesn't dominate the prompt
    (it's already self-contained and the scaffold's already in trainers.py
    by the time claude_mode sees this). Returns a sentinel string if no
    brief exists yet — the daily research run hasn't fired its first time.
    """
    path = BASE / 'data' / 'research_briefs' / 'latest.md'
    if not path.exists():
        return '(no research brief yet — daily research_mode at 03:00 will produce one)'
    text = path.read_text(errors='replace')
    lines = text.splitlines()
    if len(lines) > max_lines:
        head = '\n'.join(lines[:max_lines])
        return head + f'\n\n... [truncated, full brief at {path.relative_to(BASE)}]'
    return text


def _same_family_streak(k: int = 8) -> tuple[int, str | None]:
    """Count the run of consecutive failing claude-mode iterations of the
    same trainer at the head of the log (most recent first).

    Train-mode runs are ignored — train HP-sweeps the active case by
    construction, so any "streak" there is the cron schedule talking, not
    a deliberate Claude choice. Returns (count, trainer_name) or (0, None)
    when no streak exists.
    """
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute(
            "SELECT trainer, gate_passed FROM iterations "
            "WHERE mode='claude' ORDER BY id DESC LIMIT ?",
            (k,),
        ).fetchall()
    if not rows or rows[0]['gate_passed']:
        return 0, None
    head_trainer = rows[0]['trainer']
    count = 0
    for r in rows:
        if r['trainer'] == head_trainer and not r['gate_passed']:
            count += 1
        else:
            break
    return count, head_trainer


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
- Universe: any THB-denominated SET / mai / TDEX (ThaiDEX ETF) instrument tradable via BLS
- Walk-forward gate: 7 calendar splits in scripts/return_gate.py
- v1 per-window pass: max_dd <= 20%, n_trades >= 20, wr >= max(30%, SET_buy_hold_wr + 1pp)
  (regime-aware floor since 2026-05-26; per-window SET WR is in data/baselines.json
  `set_buy_hold.per_window`. Approximate per-window thresholds at the time of writing:
  W1≈49.7%, W2≈44.0%, W3≈54.8%, W4≈47.2%, W5≈47.2%, W6≈50.4%, W7≈59.5%. The bar moves
  with the market — target alpha above market WR, not the legacy 40% floor that SET
  routinely clears. Set RETURN_GATE_REGIME_AWARE=0 to fall back to the fixed 40% floor.)
- v1 model-level pass (becomes a candidate): AT LEAST 6 of 7 windows pass per-window
  AND avg annualized_return strictly beats the best prior candidate (or > 0 if none
  yet). The 6/7 bar replaces an earlier 7/7 bar that produced zero passes across 750
  iterations and 42 trainer families — one regime may be exempt, but the avg-ann
  competition keeps the panel honest.

==================================================================
3. PIPELINE STATE
==================================================================
Trainers currently registered (models/trainers.py): {trainers}
Curated features ({n_features} total): {features_preview}
Latest feedback DB stats: {db_stats}
Active experimental case (models/active_case.json): {active_case_summary}

Every prior claude iteration is committed to git. If a trainer / feature / label / sequence-loader change you read about in §4 RECENT ITERATIONS is not visible in the current source, it was overwritten by a later iteration — recover it via:
  - `git log --oneline -- models/trainers.py` (find the commit that added it)
  - `git show <sha>:models/trainers.py` (read the prior version)
  - `git show <sha> -- models/search_spaces.py` (matching HP space)
Do NOT submit a "decompile the .pyc" data request — git history is the canonical source.

==================================================================
4. RECENT ITERATIONS ({n_recent} latest, newest first)
==================================================================
{recent_iterations}

==================================================================
5. OPEN DATA REQUESTS
==================================================================
{open_requests}

==================================================================
5b. TODAY'S RESEARCH BRIEF (the order you execute)
==================================================================
Research mode runs daily and decides what the loop tries next. You are the EXECUTOR of that decision, not a parallel decision-maker. The brief below was committed within the last 24 hours and is the authoritative answer to "what trainer should claude_mode run this iteration."

{research_brief}

{pivot_directive}{brief_directive}==================================================================
6. YOUR JOB — execute the brief, then diagnose what happened
==================================================================

Order of operations is now FIXED:

1. **EXECUTE** — run the gate with the trainer named in §6.B.0 below. No alternatives, no "improvements", no audit-driven pivots. If §6.B.0 forbids pivoting, the `trainer` field in your JSON output MUST equal the brief's registry_key.
2. **DIAGNOSE** (Part A) — same audit as before (baselines + regime stats + failure pattern), but it runs AFTER your gate result is in. The diagnosis informs research_mode's NEXT brief, not your CURRENT pick.
3. **REPORT** — emit the JSON block with gate_result, diagnosis, AND the new `exhausted_brief` field (true/false).

Why this changed: the loop ran for 16 hours with research_mode active and claude_mode reverted to a tree variant (histgb_monotonic) at iter #951 after two scaffolded picks (torch_time_moe #892, tabm_classifier #907) failed. The audit-first protocol gave claude rhetorical cover to override the research order. That defeats the point of having a research stage. claude_mode's job is execution; research_mode's job is selection.

PART A — DIAGNOSE  (audit AFTER execution — feeds research_mode tomorrow)

A.1. **Baselines** — what does a non-model strategy score on this same gate?
   - `random_topk` baseline: for each test date, pick the top-K symbols by a uniformly random score; apply the gate. Run this once and report the 7-window result.
   - `rule_only` baseline: select symbols where `atr/close > 0.03 AND volume_ratio > 2.0 AND 30 <= RSI <= 65 AND close >= 1.0`; the highest-`volume_ratio` symbol per date is the trade. Apply the gate.
   - If your proposed model change can't clearly beat both baselines on at least 5/7 windows, it's noise and you should not propose it.
   - You may reuse the baseline numbers across iterations if the underlying data hasn't changed materially — cache via `data/baselines.json` if helpful. Re-run them weekly.

A.2. **Per-window regime stats** — for each of the 7 walk-forward TEST windows, compute on the test slice only:
   - SET index return (close-on-close). If `data/candles.db.investing_set_index` is empty for the window (it currently only covers 2026-02..2026-03), fall back to equal-weighted close-to-close return aggregated from `candles` — same shape, smaller bias.
   - SET volatility (20d realized std of returns)
   - Market breadth (mean % of symbols above their 20d SMA, across the window)
   - Foreign-flow direction (positive / negative aggregate net for the window). The daily table `data/candles.db.foreign_flows` is sparsely populated (~2026 only). The monthly table `data/candles.db.foreign_flows_monthly` (columns: month TEXT 'YYYY-MM', net_value REAL) holds 1995-01..2026-02 and IS sufficient for window-level direction — sum `net_value` across the months overlapping the test window. Do not submit a data request to backfill daily foreign_flows; the monthly table already meets Part A.2's requirement.
   Report a small table. Flag windows that look hostile (negative SET return + high vol + bearish foreign flow). This tells you whether failures are regime-correlated.

A.3. **Failure pattern across last 20 iterations** — query feedback DB. Cross-tab `trainer × window`: which windows consistently fail across trainers? If W1, W4, W5 fail in 18/20 recent runs regardless of trainer family, the bottleneck is in those regimes' data, not model architecture.

PART B — EXECUTE THE BRIEF

  B.0. **THE TRAINER YOU MUST USE THIS ITERATION** — read the §5b BRIEF DIRECTIVE block above. It tells you exactly one of:
       (a) The registry_key of the brief's pick. You MUST set `trainer = <that_key>` in your gate run AND in the JSON output's `trainer` field. Set `exhausted_brief: false`. Pick the default HPs from the brief or the search_spaces.py entry for that key — minor HP nudges OK; structural changes NOT OK.
       (b) "Brief exhausted, pivot allowed" — only when the brief's trainer has been attempted ≥2 times in claude-mode with best wp < 3. In this case set `exhausted_brief: true` and pick a STRUCTURALLY DIFFERENT trainer per B.1–B.4 below. The choice must be evidence-backed by Part A.
       (c) "No brief present" — you have full discretion; choose per B.1–B.4 below.
       Any other deviation (custom variant on the brief's trainer, audit-driven pivot to a tree family, etc.) is rejected at parse time by claude_mode.py: your changes are reverted via `git checkout HEAD --`, the iteration is recorded as a deviation, and active_case.json is NOT updated. Don't deviate — emit `exhausted_brief: true` if you genuinely believe the brief is dead.

  B.1. Add or modify a feature in models/feature_eng.py — cite a Part A finding that motivates it (e.g., "W1 has bearish foreign flow regime so I'm adding foreign_flow_5d_zscore").
  B.2. Add a new trainer in models/trainers.py — encouraged when the new family represents a **genuinely different inductive bias** not present in the registry. The current registry (see §3) is heavy on XGBoost / LightGBM loss variants. Under-represented families that the loop has barely explored or never tried (a sample, not a fixed list):
      - **NN families**: MLP / CNN / LSTM / Transformer / TabNet / FT-Transformer. The torch_* trainers from earlier iterations were lost and never rebuilt.
      - **Self-supervised / unsupervised pretraining**: masked-reconstruction encoders on the full SET universe, autoencoder + downstream linear head, contrastive (SimCLR-style on time-series windows).
      - **Foundation-model PEFT**: LoRA / QLoRA / IA3 over time-series foundation models (Chronos, Lag-Llama, Moirai, TimeGPT).
      - **RL**: PPO / DQN on the daily-close action space.
      - **Bayesian / probabilistic**: Gaussian Process classifiers, MC-Bayesian neural nets.
      "We haven't tried <novel inductive bias>" IS a valid hypothesis — those families may break the saturation pattern the XGB-loss variants are stuck in. "We haven't tried <yet another XGB loss tweak>" is not.
  B.3. Modify models/labels.py — propose label/friction changes only as a v2-whitepaper data request (see B.5). Do NOT change labels in code without explicit human sign-off — the label is the contract the gate evaluates against.
  B.4. Modify models/sequence_loader.py — data prep / augmentation; cite Part A.
  B.5. Submit a data request via scripts/feedback.py log_data_request when Part A reveals something no code change can address (e.g., "W1/W5 are bear-regime test slices following bull-train; need more 2022 bear data" → request additional historical ingest).

Evidence-free trainer additions will be reviewed harshly. The bar going forward: this run produced data and an insight, not just code. If after Part A you have no actionable insight, it is acceptable (and preferable) to skip Part B and emit a data request explaining what's blocking you.

You may use WebSearch / WebFetch to research recent ML techniques. The list of "trainer families in scope" in ml-loop.md is a starting point, not a cage.

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
8. ACTIVE CASE — designate which trainer family train mode should sweep next
==================================================================
The cron-driven train mode (every hour at :30) sweeps HPs WITHIN ONE trainer
family. Which family is "active" is decided by you, in this iteration's
report, via the ``active_case`` block. Train mode reads
``models/active_case.json`` at startup; whatever you designate here is what
gets HP-swept until the next claude run reassigns it.

Two valid choices each iteration:

- **Keep the same case** — copy the prior active_case forward (deepens the
  sweep on the same family). Pick this when train mode hasn't yet exhausted
  the current family's HP space.
- **Pivot to a new case** — designate a different trainer (e.g. one you just
  added in this run, or a previously-registered one you now want to sweep).

The block must specify ``trainer`` (a key from the registry above),
``feature_set``, a ``dataset`` config, and a one-line ``rationale``. Defaults
are fine for ``feature_set`` (``"default"``) and ``dataset``
(``{{"seq_len": 20, "lookahead": 10, "label": "v1"}}``) unless you changed
them this iteration.

==================================================================
9. OUTPUT FORMAT — your final message MUST end with this JSON block
==================================================================
```json
{{
  "trainer": "<registered trainer name used for the gate — MUST equal the brief's registry_key unless exhausted_brief=true>",
  "exhausted_brief": false,
  "hyperparams": {{ ... }},
  "code_changes": [
    "models/trainers.py: added LSTMTrainer (PyTorch, 2-layer, dropout 0.3)",
    "models/search_spaces.py: lstm entry"
  ],
  "hypothesis": "<one sentence — what change, why you expect it to help. MUST cite a Part A finding>",
  "backbone": "<for PEFT only: which pre-trained model, e.g. 'lag-llama-1b'>",
  "diagnosis": {{
    "baselines": {{
      "random_topk":  {{"windows_passed": 0, "avg_win_rate": 0.0, "avg_annualized_return": 0.0, "avg_max_dd": 0.0, "total_trades": 0}},
      "rule_only":    {{"windows_passed": 0, "avg_win_rate": 0.0, "avg_annualized_return": 0.0, "avg_max_dd": 0.0, "total_trades": 0}}
    }},
    "regime_stats": [
      {{"window": "2023-11..2024-02", "set_return": 0.0, "set_vol_20d": 0.0, "breadth_above_sma20": 0.0, "foreign_flow_net": 0.0, "hostile": false}},
      "... 6 more rows, one per walk-forward window ..."
    ],
    "failure_pattern": "<one short paragraph: which windows fail consistently across last 20 iterations, regardless of trainer>"
  }},
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
  "active_case": {{
    "trainer": "<registry key for train mode to sweep next>",
    "feature_set": "default",
    "dataset": {{"seq_len": 20, "lookahead": 10, "label": "v1"}},
    "rationale": "<one line — why this is the strongest base for HP sweep right now>"
  }},
  "lessons": "<1-2 sentences — what worked, what didn't, what to try next>",
  "data_request": "<empty string, or a free-form request to be sent to the operator>",
  "git_action": "kept" | "reverted"
}}
```

Be precise. The JSON is parsed by claude_mode.py and written to the SQLite
feedback DB. ``active_case`` is also written verbatim to
``models/active_case.json`` and gates train mode until you next override it.
Bad JSON = lost iteration.

==================================================================
10. NON-NEGOTIABLES
==================================================================
- Do NOT change ~/projects/caffe-stocks/docs/ml-training.md or docs/ml-loop.md (the constitution).
- Do NOT touch ~/projects/caffe-stocks/data/ (production runtime state).
- Do NOT delete iteration_windows or iterations rows from the DB.
- Do NOT install new Python packages without first checking if a comparable one exists.
- Do NOT skip the JSON output block.
- Do NOT skip the active_case block — train mode depends on it. If you have
  no opinion, copy the prior active_case forward unchanged.

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

    case = active_case.read()
    if case:
        active_case_summary = (
            f"trainer={case.get('trainer')!r} "
            f"feature_set={case.get('feature_set')!r} "
            f"dataset={case.get('dataset')!r} "
            f"claude_iter_id={case.get('claude_iter_id')!r} "
            f"rationale={(case.get('rationale') or '')[:120]!r}"
        )
    else:
        active_case_summary = '(none — first run, or file missing)'

    streak_count, streak_trainer = _same_family_streak(k=8)
    if streak_count >= 3:
        pivot_directive = (
            "==================================================================\n"
            "PIVOT DIRECTIVE — TRIGGERED (mandatory this iteration)\n"
            "==================================================================\n"
            f"Trainer family `{streak_trainer}` has failed {streak_count} "
            "consecutive claude-mode iterations. This iteration MUST switch "
            "to a structurally different trainer family — pick from §6.B.2 "
            "(NN, unsupervised, foundation-model PEFT, RL, Bayesian) or any "
            "other registered family that is NOT a hyperparameter or label "
            "tweak of the current one. The `trainer` field in your gate run "
            "and the `active_case` block in your JSON output MUST NOT be "
            f"`{streak_trainer}`. Cite Part A evidence justifying the new "
            "family's inductive bias.\n\n"
        )
    else:
        pivot_directive = ''

    return PROMPT_TEMPLATE.format(
        mission=indent(mission, '  '),
        trainers=trainers,
        n_features=len(features),
        features_preview=features_preview,
        db_stats=db_stats_str,
        n_recent=len(iters),
        recent_iterations=recent_block,
        open_requests=requests_block,
        active_case_summary=active_case_summary,
        pivot_directive=pivot_directive,
        research_brief=_research_brief(),
        brief_directive=_brief_directive(),
    )


def main():
    print(build_prompt())


if __name__ == '__main__':
    main()
