#!/usr/bin/env python3
"""Prompt builder for research_mode.

Research mode's sole job is to write today's `data/research_briefs/YYYY-MM-DD.md`
whitepaper. It does NOT run the gate, NOT pivot active_case, NOT install deps,
NOT touch trainers.py — that's ml_scaffold.py's job, which parses the brief
after research_mode completes.

The prompt is heavily WebSearch-biased: research mode reads only minimal local
state (one-line registry-bias summary + the just-prior brief's "Recommended
next action"); everything else should come from web sources dated within the
past 12 months. Old data and iteration archaeology belong to claude_mode.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(os.path.expanduser('~/projects/caffe-stocks'))
sys.path.insert(0, str(BASE))

from scripts import feedback as fb
from models.trainers import TRAINERS

BRIEFS_DIR = BASE / 'data' / 'research_briefs'


# Coarse family bucketing, kept locally so we don't depend on promote_panel.
# Any unknown registry key falls into 'other'.
def _family(trainer: str) -> str:
    if trainer.startswith('torch_') or trainer in ('mlp_classifier', 'tabresnet'):
        return 'nn'
    if 'ranker' in trainer or 'rank_' in trainer:
        return 'ranker'
    if trainer.startswith('xgb_') or trainer in ('xgboost', 'lightgbm', 'lightgbm_regressor'):
        if 'regress' in trainer or 'quantile' in trainer or 'huber' in trainer:
            return 'xgb_regressor'
        return 'xgb_classifier'
    if trainer in ('logistic_elastic_net', 'kernel_logreg', 'bagged_kernel_logreg'):
        return 'linear_kernel'
    if trainer in ('knn_classifier', 'gaussian_nb', 'qda_classifier',
                    'gaussian_process_classifier'):
        return 'non_parametric'
    if trainer in ('sklearn_extra_trees',):
        return 'tree_other'
    return 'other'


def _registry_bias() -> str:
    """One short paragraph summarizing 30-day iteration counts by family.
    Surfaces the imbalance research mode is trying to break."""
    fb.init_db()
    with fb.get_conn() as conn:
        rows = conn.execute(
            "SELECT trainer FROM iterations "
            "WHERE finished_at >= datetime('now','-30 days')"
        ).fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        counts[_family(r['trainer'])] = counts.get(_family(r['trainer']), 0) + 1
    if not counts:
        return '(no iterations in last 30 days)'
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    parts = [f'{fam}={n} ({100*n/total:.0f}%)' for fam, n in ordered]
    return f'last 30d total {total} iters: ' + ', '.join(parts)


def _latest_prior_brief() -> str:
    """One-paragraph context from the most recent brief, if any. Just the
    'Recommended next action' line — research mode shouldn't dwell on its
    own history; the goal is fresh external data."""
    if not BRIEFS_DIR.exists():
        return '(no prior briefs)'
    briefs = sorted(BRIEFS_DIR.glob('20*.md'))
    if not briefs:
        return '(no prior briefs)'
    text = briefs[-1].read_text(errors='replace')
    # Pull the line after "## Recommended next action" if present.
    marker = '## Recommended next action'
    if marker in text:
        tail = text.split(marker, 1)[1].strip()
        # Strip subsequent headers
        for line in tail.splitlines():
            if line.startswith('## '):
                break
            if line.strip():
                return f'{briefs[-1].name}: {line.strip()[:200]}'
    return f'{briefs[-1].name} exists ({len(text)} chars) — no parseable recommendation'


def _current_registry() -> list[str]:
    return sorted(TRAINERS.keys())


PROMPT_TEMPLATE = """You are running ONE iteration of the Caffe-Stocks ML loop in RESEARCH MODE.

Research mode runs ONCE PER DAY (03:00 local). Your job is to write today's research whitepaper to `data/research_briefs/{today}.md`. You do NOT run the gate. You do NOT install dependencies. You do NOT write trainer code. You do NOT pivot active_case. A SEPARATE post-script (`scripts/ml_scaffold.py`) reads your brief and does the mechanical work — installs, scaffolds, registry registration. So your output is a markdown file, nothing else.

Hard wall-time: 45 minutes.

==================================================================
1. THE PROBLEM YOU EXIST TO SOLVE
==================================================================
Claude_mode keeps grinding on XGBoost-loss tweaks. 750+ iterations across 42 trainer families have produced zero gate-passes. Web-published SOTA for time-series and tabular ML in 2025–2026 (TabPFN v2.5, TimesFM, Chronos-2, Moirai-2, Mamba/SSM, iTransformer, PatchTST, contrastive SSL) is **not in our registry**. The loop has the WebSearch tool but treats it as optional — so claude_mode never breaks out of its comfort zone.

You break the cycle. Every day you survey the latest web sources and produce a focused, actionable brief that claude_mode will then read in-context on its next run. Yesterday's brief is irrelevant if today's web search finds something better — refresh, don't repeat.

==================================================================
2. MINIMAL LOCAL STATE (do not over-read)
==================================================================
Registry bias (last 30d): {registry_bias}
Current registry ({n_trainers} trainers): {trainers}
Latest prior brief recommendation: {prior_brief}

That is ALL the local state you need. Do not query the iteration DB further; do not read trainers.py source. Old data is not the goal of research mode — it's claude_mode's job to consult history. Your job is to bring in fresh external knowledge.

==================================================================
3. WORKFLOW (must follow)
==================================================================

PHASE A — WEB RESEARCH (heaviest section of your run; budget ~25 of 45 min)

You MUST issue at least 5 WebSearch queries across these axes:
- Latest 2025/2026 SOTA on time-series foundation models (TimesFM, Chronos, Moirai, Lag-Llama, Timer, Toto — find new entrants, find updated benchmarks)
- Latest 2025/2026 SOTA on TABULAR foundation models (TabPFN v2.5/v3, AutoGluon, neural tabular)
- Latest 2025/2026 SOTA on stock prediction / quantitative finance ML (transformer for stocks, cross-sectional ranking models, financial SSL)
- Latest 2025/2026 SOTA on state-space / Mamba / linear-attention models in time-series
- One free-form query of your choice on a NICHE / EMERGING family (e.g., neural ODEs, equivariant networks, mixture of experts for time-series, retrieval-augmented forecasting)

Use WebFetch on at least 2 specific arxiv / blog / GitHub pages to get concrete implementation details — dep names, weight URLs, sample code patterns.

The CURRENT date is {today}. Anything dated more than 12 months old is "history" and should be cited as context only — your candidate must be something the field has converged on or pushed out in the last 6–12 months.

PHASE B — CANDIDATE SELECTION (~10 min)

From your research, pick exactly ONE technique that is:
1. Genuinely under-represented in our registry (cite the registry-bias summary above)
2. Implementable as a `BaseTrainer` subclass in `models/trainers.py` (i.e., supports `fit(X, y, X_val, y_val)` + `predict_proba(X)` on 2D numeric features — or a clean adapter exists)
3. Has Python deps installable via pip (no compiled-from-source, no manual model weight curation)
4. Practical wall-time: training on 190k rows × 96 features × 7 walk-forward windows MUST fit in claude_mode's 30-min budget. If it can't, propose a frozen-encoder + small-head variant.

Prefer something concrete and shipped over something theoretical. If two options compete, pick the one with a maintained HuggingFace / GitHub repo and fewest install hassles.

PHASE C — WRITE THE BRIEF (~10 min)

Write the brief to `data/research_briefs/{today}.md` using the EXACT structure below. The post-script parses it mechanically — section headers and code-block fences must match.

```markdown
# Research brief — {today}

## Registry-bias diagnosis
<one paragraph quoting the bias counts above + your read on what's missing>

## Recommended technique: <Human-readable name>

### Why now
<2-3 sentences citing what the web research shows about this technique's
maturity / recent results / fit to our problem. Reference at least 2 of
your search/fetch citations by URL.>

### Deps
```bash
pip install <package-1> <package-2>
```

### Weights
<Optional. If a pretrained checkpoint is needed, ONE of:
- `huggingface-cli download org/model --local-dir <path>`
- direct download URL
- "N/A — trained from scratch on our data"
Use the exact command on its own line — ml_scaffold parses this verbatim.>

### Scaffold code
```python
# models/trainers.py — append at end (before TRAINERS dict)
# Complete, ready-to-paste class definition. Must:
#   - subclass BaseTrainer
#   - implement fit(self, X_tr, y_tr, X_val, y_val, **kwargs)
#   - implement predict_proba(self, X) -> np.ndarray of shape (N,)
#   - implement save(self, dir, extra={{}}) — write artifact files
# Imports go at the top of the code block. The post-script appends this
# block verbatim above the `TRAINERS = {{...}}` line in trainers.py.

import numpy as np
# ... additional imports ...

class TorchNewTrainer(BaseTrainer):
    name = 'torch_new'
    consumes_sequences = False
    def __init__(self, **kwargs): ...
    def fit(self, X_tr, y_tr, X_val, y_val, **kwargs): ...
    def predict_proba(self, X) -> np.ndarray: ...
    def save(self, model_dir, extra=None): ...
```

### Registry registration
```python
# Single line to append to TRAINERS dict in models/trainers.py
'torch_new': TorchNewTrainer,
```

### HP search space
```python
# models/search_spaces.py — entry to add to SEARCH_SPACES dict
'torch_new': {{
    'hidden_dim':     [64, 128, 256],
    'learning_rate':  (1e-4, 1e-2),
    # ...
}},
```

### Integration notes (for claude_mode)
<2-4 sentences: what claude_mode needs to know to give this trainer a fair
shot on its next run. Pre-training quirks, GPU/CPU sensitivity, data
preprocessing assumptions, expected gate behavior on each window.>

## Recommended next action for claude_mode
<one paragraph: after the scaffold is in place (post-script will run within
the hour), claude_mode's next iteration should `--model-type torch_new`
with these default HPs and cite this brief. Mention any specific failure
window from registry-bias diagnosis that this technique is likely to help.>

## Sources
- [Title 1](URL1) — one-line note on what this source contributes
- [Title 2](URL2) — ...
- [Title 3](URL3) — ...
(minimum 4 sources, ideally arxiv + github + benchmark blog + maintainer
docs)
```

PHASE D — VERIFY AND EMIT JSON

After writing the .md file:
1. Re-read it with the Read tool and confirm every required section header is present
2. Confirm code blocks are syntactically plausible Python (will be exec'd by post-script)
3. Emit the JSON output block below

==================================================================
4. OUTPUT FORMAT — your final message MUST end with this JSON block
==================================================================
```json
{{
  "brief_path": "data/research_briefs/{today}.md",
  "technique": "<human-readable technique name>",
  "registry_key": "<torch_xxx or similar — must NOT collide with existing keys>",
  "deps": ["package-1", "package-2"],
  "needs_weights": true|false,
  "sources_count": 4,
  "summary": "<one sentence — what's getting scaffolded and why it should break the saturation>"
}}
```

==================================================================
5. NON-NEGOTIABLES
==================================================================
- Do NOT run `scripts/return_gate.py` or any python script that touches the gate. Research mode is research-only.
- Do NOT modify `models/active_case.json`, `models/trainers.py`, `models/search_spaces.py`, `models/labels.py`, `models/sequence_loader.py`, `scripts/prompt_builder.py`, or any file outside `data/research_briefs/`. The post-script does that, deterministically, from your brief.
- Do NOT pip install. The post-script does that.
- Do NOT propose RL / Bayesian / online learning families that require infra we don't have. Propose only what fits the existing 2D tabular fit/predict_proba contract or has a clean adapter.
- Do NOT pick a registry key that collides with an existing trainer. List of current keys is in §2.
- Do NOT write Python code as raw text outside a fenced ```python``` block — the post-script parses code blocks by fence.
- If you cannot identify a credible candidate after Phase A (rare — the field is full of options), write a brief that explicitly says so under "Recommended technique: NONE" and emit `"registry_key": null` in the JSON. The post-script will skip cleanly.

Proceed.
"""


def build_prompt() -> str:
    today = datetime.now().strftime('%Y-%m-%d')
    return PROMPT_TEMPLATE.format(
        today=today,
        registry_bias=_registry_bias(),
        n_trainers=len(_current_registry()),
        trainers=', '.join(_current_registry()),
        prior_brief=_latest_prior_brief(),
    )


def main():
    print(build_prompt())


if __name__ == '__main__':
    main()
