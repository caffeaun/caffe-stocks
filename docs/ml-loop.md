# ML Iteration Loop — design

Companion to `ml-training.md`. The whitepaper defines *what we are training*; this doc defines *how we keep training better versions of it*.

Status: **draft v1** — design only, no code yet. Sign off before implementation.

## Goal

A model-agnostic, self-iterating ML loop that:
- Treats algorithm choice (decision trees, NN, RL, ensemble) as a swappable plug-in
- Separates **strategic changes** (Claude proposes code-level moves) from **tactical search** (cheap automated HP tuning)
- Writes a structured feedback log every iteration so neither Claude nor the sweeper repeats failed experiments
- Pauses itself when stuck, instead of grinding 327 LSTM tweaks like the old loop did
- Stays callable from a single shell wrapper invokable by cron, Telegram, or by hand

## Two modes — sharp boundary

### Train mode (`scripts/train_mode.py`)

**What it owns:** **hyperparameter search** within search spaces that already exist in code.

- Trainer type chosen from the registry (`models/trainers.py`)
- HP ranges read from `models/search_spaces.py` (per-trainer dict of name → [low, high] or [choices])
- Random or low-discrepancy sampling of N configs per run
- Each config goes through `return_gate.py` → annualized return / DD / WR / trade count
- Best config gets saved as `models/{trainer}/candidates/`
- One feedback entry per config, tagged `mode=train`
- Cheap: 5s per config × 20 configs ≈ 2 min per run
- **Cannot change code.** If a search space is missing or a trainer doesn't exist, train mode skips, doesn't synthesize.

### Claude mode (`scripts/claude_mode.py`)

**What it owns:** **code changes** — anything HP search can't reach.

- Reads recent iteration history from `ml-feedback.db` (SQLite) + the whitepaper + current trainer source
- Has **WebSearch + WebFetch** allowed — can pull in new ML techniques from arXiv, papers, blog posts, GitHub. Not restricted to the families listed in this doc; if a 2026 paper proposes something better, Claude mode is allowed to bring it in.
- Calls `claude -p` with a structured prompt asking for ONE specific code change per run
- Allowed surface area:
  1. Add new trainer to `models/trainers.py` (any family — listed examples are not exhaustive)
  2. Add/modify search space in `models/search_spaces.py`
  3. Modify `models/feature_eng.py` (add/drop features) — only when train mode hits a feature-shaped wall
  4. Modify `models/labels.py` (label definition, friction model)
  5. Modify `models/sequence_loader.py` (data prep, augmentation)
  6. Modify `scripts/return_gate.py` gate criteria — only when whitepaper §9 needs to change
  7. Submit a *data request* (text file → Telegram) when no code change can help
- Output: a candidate `models/{trainer}/candidates/` from the new code, gate-evaluated, feedback entry tagged `mode=claude`
- Hard wall-time: **30 min per run** (timeout-killed if exceeded)
- **Cannot tune HPs.** That's train mode's job.

This boundary is intentional. The LSTM-era loop blurred them — Claude was tuning hyperparameters AND structural changes in the same run, which made it impossible to attribute wins. With the boundary:
- Train-mode wins → the model space is rich; sweep harder
- Claude-mode wins → the model space needs new dimensions

### Both modes share

- `return_gate.py` — the only scoreboard. Both modes pass/fail by the same criteria.
- `data/ml-feedback.json` — structured log; both write, both read.
- `docs/ml-training.md` — the whitepaper. Both treat it as constitutional. Claude mode can *propose* changes via Telegram, never edits unilaterally.

## File layout

```
caffe-stocks/
├── docs/
│   ├── ml-training.md       (whitepaper — strategy + gate definition)
│   └── ml-loop.md           (this file — iteration mechanics)
├── models/
│   ├── trainers.py          (registry: BaseTrainer + LightGBM + XGBoost + future)
│   ├── search_spaces.py     (NEW — per-trainer HP ranges)
│   ├── feature_eng.py       (curated features, mostly stable)
│   ├── sequence_loader.py   (data prep, with cache)
│   └── labels.py            (label definition, friction model)
├── scripts/
│   ├── trainer.py           (single-shot — used by humans for sanity)
│   ├── return_gate.py       (walk-forward — the only scoreboard)
│   ├── train_mode.py        (NEW — HP sweep, ~20 configs/run)
│   ├── claude_mode.py       (NEW — invokes claude -p with feedback)
│   ├── feedback.py          (NEW — structured read/append to ml-feedback.json)
│   ├── prompt_builder.py    (NEW — builds Claude prompt)
│   └── ml_loop.sh           (NEW — top-level wrapper, replaces ml-improve.sh)
└── data/
    └── ml-feedback.json     (append-only iteration log)
```

## Trainer plug-in protocol

The pluggable interface (already in `models/trainers.py`):

```python
class BaseTrainer:
    name: str                              # registry key
    def fit(X_train, y_train, X_val, y_val, verbose) -> self
    def predict_proba(X) -> np.ndarray     # shape (N,), values in [0, 1]
    def save(output_dir, extra) -> dict    # writes artifacts, returns paths
    def load(output_dir) -> Trainer        # round-trips
    def feature_importance() -> Optional[np.ndarray]
    @property best_iteration -> Optional[int]
    @property hyperparams -> dict
```

Adding a new model type means:

1. Subclass `BaseTrainer`
2. Register name in `TRAINERS` dict
3. Add an entry in `search_spaces.py` defining what HPs train mode is allowed to vary
4. Run `python scripts/trainer.py --model-type <name>` once for sanity
5. `train_mode.py` and `claude_mode.py` automatically pick it up

### Trainer families in scope

**Tree-based** — current default
- `LightGBMTrainer` ✓ (shipped)
- `XGBoostTrainer` ✓ (shipped)
- `CatBoostTrainer` — categorical feature handling for sector encoding

**From-scratch neural**
- `LSTMTrainer` — sequence model; same interface as the legacy LSTM but using the new gate
- `TransformerTrainer` — self-attention over the 20-day window
- `MLPTrainer` — feed-forward on aggregated features (a control for the trees)

**Transfer-learning / parameter-efficient fine-tuning (PEFT)**

These wrap a *pre-trained backbone* (foundation model or our own pre-trained encoder) and only train a small fraction of parameters. Useful when a small dataset (~44k sequences) can't support training a large model from scratch.

- `HeadFineTuner` — cheapest. Freezes the entire backbone, retrains only the classification head (a single Linear → Sigmoid). ~2k trainable params; trains in seconds even with a frozen 100M-param backbone.
- `LoRATrainer` — Low-Rank Adaptation (Hu et al. 2021). Injects rank-r adapters into chosen layers (typically attention Q/V projections), freezes the base. Tunable: rank `r`, scaling `alpha`, dropout, target_modules. Adds ~0.1-1% extra params, often matches or beats full fine-tuning on small data.
- `QLoRATrainer` — Quantized LoRA (Dettmers et al. 2023). Same as LoRA but the frozen base is quantized to 4-bit (NF4 + double-quant). Drops the memory footprint by ~4× — lets a 7B-param backbone fit in <8 GB VRAM. Tunable: same as LoRA plus quant config.
- `IA3Trainer` — even lighter than LoRA: scales activations of K, V, and FFN by learned vectors. ~10× fewer trainable params than LoRA. Worth trying if LoRA still overfits.
- `PromptTuningTrainer` — only learns soft prompts prepended to the input. The lightest option; if it works, the rest of the model has the signal already.

**Backbone options for PEFT** (Claude mode picks one when adding a PEFT trainer):
- Lag-Llama, Chronos-T5, Moirai, TimeGPT — public time-series foundation models
- Our own self-supervised encoder pre-trained on the full SET universe via next-step prediction or masked reconstruction (`scripts/pretrain.py` would be added by Claude mode when chosen)

**Ensembles**
- `EnsembleTrainer` — averages predictions of N sub-trainers (e.g., 3× XGBoost with different seeds + 1× LSTM)
- `StackingTrainer` — meta-learner trained on out-of-fold predictions of base trainers

**Reinforcement learning**
- `RLTrainer` — wraps a Stable-Baselines3 PPO/DQN agent. Action space = {hold, buy, sell}; observation = the same engineered features. Reward = realized P&L per closed trade. Outputs P(buy) by softmax over the action head.

**Interpretable baseline**
- `LinearTrainer` — sklearn LogisticRegression. If the trees and NNs can't beat a logistic regression on the engineered features, the engineered features are the bottleneck — not the model.

The gate doesn't care which one's running. If LoRA-tuned Lag-Llama beats trees, that ships. If logistic regression beats them all, that ships.

## Search space schema (new file)

```python
# models/search_spaces.py
SEARCH_SPACES = {
    'lightgbm': {
        'num_leaves': [15, 31, 63, 127],
        'max_depth': [4, 6, 8, -1],
        'learning_rate': (0.01, 0.10),       # tuple = continuous range
        'n_estimators': [200, 500, 1000],
        'min_child_samples': [20, 50, 100, 200],
        'pos_class_weight': (1.0, 8.0),
    },
    'xgboost': {
        'max_depth': [3, 4, 6, 8],
        'learning_rate': (0.01, 0.10),
        'n_estimators': [200, 400, 800],
        'min_child_weight': [1, 5, 10, 20],
        'gamma': (0.0, 0.5),
    },
    'lstm': {
        'hidden_size': [32, 48, 64, 128],
        'num_layers': [1, 2, 3],
        'dropout': (0.1, 0.5),
        'lr': (1e-4, 5e-3),
        'batch_size': [64, 128, 256],
    },
    'lora': {
        'rank': [4, 8, 16, 32],
        'alpha': [8, 16, 32, 64],
        'dropout': (0.0, 0.2),
        'target_modules': [['q_proj', 'v_proj'], ['q_proj', 'k_proj', 'v_proj']],
        'lr': (1e-5, 1e-3),
        'epochs': [3, 5, 10, 20],
    },
    'qlora': {
        'rank': [4, 8, 16, 32],
        'alpha': [8, 16, 32],
        'quant_bits': [4, 8],
        'compute_dtype': ['bfloat16', 'float16'],
        'lr': (1e-5, 1e-3),
    },
    'head_finetune': {
        'lr': (1e-4, 1e-2),
        'weight_decay': (0.0, 1e-2),
        'epochs': [5, 10, 20, 50],
        'hidden_layers': [[], [64], [128, 64]],
    },
    # Claude can append entries here when adding a new trainer.
}
```

Train mode samples N configs from these ranges, gates each, picks winners.

## Feedback store — SQLite

`data/ml-feedback.db`. JSON couldn't handle concurrent reads/writes once we have train mode running every hour while Claude mode reads it. SQLite gives us proper indexing for "top 3 in last 30 days" queries and append-safety with WAL mode.

```sql
-- One row per gate run, regardless of mode.
CREATE TABLE iterations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,                       -- ISO 8601
    finished_at     TEXT NOT NULL,
    mode            TEXT NOT NULL CHECK(mode IN ('train', 'claude')),
    trainer         TEXT NOT NULL,                        -- registry key
    hyperparams     TEXT NOT NULL,                        -- JSON blob
    code_changes    TEXT,                                 -- JSON list, claude mode only
    hypothesis      TEXT,                                 -- claude mode only
    backbone        TEXT,                                 -- for PEFT trainers
    git_sha         TEXT,                                 -- exact code revision
    -- Aggregate gate metrics
    gate_passed         INTEGER NOT NULL,                  -- 0/1
    windows_passed      INTEGER NOT NULL,
    windows_total       INTEGER NOT NULL,
    avg_annualized_return REAL,
    avg_win_rate          REAL,
    avg_max_dd            REAL,
    total_trades          INTEGER,
    -- Artifacts + provenance
    model_dir       TEXT NOT NULL,                        -- candidate dir path
    elapsed_seconds INTEGER,
    full_result     TEXT,                                  -- JSON of return_gate output
    lessons         TEXT                                   -- claude mode only
);

CREATE INDEX idx_iter_finished        ON iterations(finished_at);
CREATE INDEX idx_iter_passed_return   ON iterations(gate_passed, avg_annualized_return);
CREATE INDEX idx_iter_mode_finished   ON iterations(mode, finished_at);

-- Per-window detail for top runs (don't store for losers — keep db small).
CREATE TABLE iteration_windows (
    iteration_id  INTEGER NOT NULL REFERENCES iterations(id),
    window_idx    INTEGER NOT NULL,
    train_start   TEXT, train_end TEXT,
    test_start    TEXT, test_end  TEXT,
    threshold     REAL,
    n_trades      INTEGER,
    win_rate      REAL,
    avg_pnl       REAL,
    avg_win       REAL,
    avg_loss      REAL,
    annualized_return REAL,
    max_drawdown  REAL,
    final_equity  REAL,
    PRIMARY KEY (iteration_id, window_idx)
);

-- The 3 currently-paper-trading models. Refreshed monthly.
CREATE TABLE production_panel (
    rank          INTEGER PRIMARY KEY CHECK(rank IN (1, 2, 3)),
    iteration_id  INTEGER NOT NULL REFERENCES iterations(id),
    promoted_at   TEXT NOT NULL,
    expires_at    TEXT NOT NULL,                          -- promoted_at + 30 days
    UNIQUE(iteration_id)
);

CREATE TABLE data_requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_id  INTEGER NOT NULL REFERENCES iterations(id),
    requested_at  TEXT NOT NULL,
    request_text  TEXT NOT NULL,
    fulfilled     INTEGER DEFAULT 0
);

-- WAL for concurrent reads while writes are in flight.
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

`scripts/feedback.py` wraps it with a small API:

```python
record_iteration(...)              # called at end of every gate run
top_n_recent(n, days)              # used by promotion + claude-mode prompt
get_promotion_panel()              # current 3 + their iterations
refresh_promotion_panel()          # called by monthly cron
recent_lessons(k)                  # last K claude-mode lessons → prompt
log_data_request(text)             # claude mode → Telegram
```

The pre-existing 335 entries from `data/ml-feedback.json` (LSTM era) get migrated into this schema as a one-time seed step so historical lessons remain queryable.

## Schedule

Two cadences with deliberate non-overlap on the hour boundary.

| When (Asia/Bangkok) | Mode | Wall-time | Notes |
|---|---|---|---|
| `0 */3 * * *` (00:00, 03:00, 06:00, … every 3h, top of hour) | **Claude** | hard kill at 30 min | 1 attempt per slot |
| `30 * * * *` (every hour at :30) | **Train** | uncapped, but must finish before next :30 | 1 process at a time |
| `1 0 1 * *` (1st of month, 00:01) | **Promotion refresh** | seconds | Re-pick top-3 from last 30 days |

**Conflict and lock priority:**

- Both modes acquire `models/.ml-loop.lock` via `flock`.
- **Claude > Train.** If Claude's slot fires while Train is running, Train is killed (matches the old `ml-improve.sh` priority hierarchy). After Claude finishes (≤ 30 min), the next :30 trigger reruns Train.
- If Claude's run is still active when its next 3-hour slot fires, the new slot is skipped (single concurrent Claude).
- If Train's run is still active when the next :30 fires, the new fire is skipped — but the previous run gets a soft warning logged. If three :30 slots in a row are skipped, an alert goes to Telegram (Train is taking too long; Claude should reduce HP search scope).

**Cost:**

We're on the Claude Max 20× plan, so no per-run cost cap is needed. The 30-minute wall-time is the only hard limit per Claude run.

**Auto-pause:**

If 5 consecutive iterations across BOTH modes fail to produce a gate-passing model, both modes pause until manual reset (`t resume-ml` over Telegram). This is the lesson from 327 LSTM attempts — grinding without progress pollutes the feedback log.

## Promotion criteria — top-3 paper-trading panel

We don't promote to a single "production" model. Instead, we keep a **panel of 3 currently-paper-trading models**, refreshed monthly. The panel is a stable A/B/C set so the strategy team can compare.

**Panel selection rules:**

1. Eligible iterations: `gate_passed = 1` AND `finished_at > now - 30 days`
2. Sort by `avg_annualized_return DESC` (primary), `avg_max_dd ASC` (tiebreaker)
3. Pick top 3 distinct trainer/hyperparam combinations
4. Write rows 1, 2, 3 to `production_panel` table
5. Each panel slot expires 30 days after promotion → next monthly refresh re-picks

**Paper trading the panel:**

- All 3 models score signals concurrently each session
- Each Telegram signal includes which panel model fired it (`signal from rank-1 / rank-2 / rank-3`)
- After a month of live paper performance, the strategy team reviews and decides which (if any) graduates to half-size real money — that's whitepaper Phase C, not auto-decided

**Manual override:** the bot supports `t freeze` (block monthly refresh), `t panel show` (current 3), `t panel promote <iter_id>` (force a specific iteration into the panel).

The "1 monolithic production model" approach was wrong for the LSTM era — it created winner-take-all dynamics where any single bad month would destroy the only model. With 3, we get diversification of model risk for ~3× the inference cost (which is cheap).

## What this fixes vs the old `ml-improve.sh`

| Problem in v0 | Fix in v1 |
|---|---|
| LSTM hardcoded everywhere | Pluggable trainer registry |
| Claude tuned HPs *and* changed code in same run | Two modes with sharp boundary |
| Sweep was generic random search ignoring history | Sweep seeds around recent winners from feedback |
| No auto-pause — 327 attempts before someone noticed it was stuck | 5-fail auto-pause with Telegram alert |
| Gate criteria (per-split WR ≥ 50%) didn't match the actual goal (50% annual return) | Gate is whitepaper §9 — return-based |
| Claude prompt was 4000 lines of accumulated rationale | Prompt is feedback-summary + last 5 entries + open requests |
| Promote logic mixed with everything | Single `promote()` function, criteria explicit |

## What about the backbone for PEFT?

LoRA / QLoRA / head-tuning need *a pre-trained base*. Three sourcing options, in order of cheapness:

1. **Use a public time-series foundation model.** Chronos, Lag-Llama, Moirai, TimeGPT — all have open weights or HF Inference API. PEFT against one of them. Risk: their pretraining distribution differs from Thai daily bars; transfer may be weak.
2. **Pre-train our own encoder.** A `scripts/pretrain.py` task: self-supervised next-step prediction or masked-reconstruction on the full SET universe, no labels needed. Once pre-trained, fine-tune the head (`HeadFineTuner`) for our specific binary label. Cost: one-time ~hours of GPU time per pretraining run; much cheaper than training a fresh model per gate split.
3. **Use the most-recent winning XGBoost as a "soft teacher."** Distill its predictions into a small NN, then PEFT. Hybrid path.

Claude mode chooses which path when proposing a PEFT trainer; it cites the choice in its hypothesis field.

## Implementation order

If signed off, implement in this order so each step produces a runnable artifact:

1. **`models/search_spaces.py`** — define HP ranges for the 2 existing trainers. ~50 lines.
2. **`scripts/feedback.py`** — read/append helper, schema validation. ~80 lines.
3. **`scripts/train_mode.py`** — sample N configs, gate each, log to feedback. ~150 lines.
4. **`scripts/prompt_builder.py`** — build Claude prompt from feedback + whitepaper + trainer source. ~120 lines.
5. **`scripts/claude_mode.py`** — invoke `claude -p`, capture output, validate code changes, gate, log. ~180 lines.
6. **`scripts/ml_loop.sh`** — top-level wrapper with mode selection, lock file, Telegram alerts. ~100 lines.
7. **Add `LSTMTrainer`** to `trainers.py` to validate the abstraction with a non-tree model. ~150 lines.
8. **Add `EnsembleTrainer`** as another validation. ~80 lines.
9. **(Optional) `RLTrainer`** as a v1.5 stretch.

Each step ends with a runnable artifact that we can hand-execute and check. We don't wire cron until step 6 is verified working.

## Decisions locked (per 2026-05-04 review)

- ✓ Boundary: train mode = HP search, Claude mode = code changes
- ✓ Claude mode internet access: WebSearch + WebFetch enabled — not restricted to listed model families
- ✓ Feedback store: SQLite (`data/ml-feedback.db`) with WAL mode, not JSON
- ✓ Schedule: Claude every 3h on the hour (max 30 min); Train every hour at :30 (uncapped but must finish in <1h)
- ✓ Conflict priority: Claude preempts Train (kills running train if needed)
- ✓ Cost cap: none (Claude Max 20× plan)
- ✓ Promotion: top-3 paper-trade panel, monthly refresh
- ✓ Auto-pause: 5 consecutive failures across both modes → pause until `t resume-ml`

## Still open

1. **`base_dir`** — currently hardcoded `~/projects/caffe-stocks`. Native install may want this configurable. Defer to migration.
2. **First trainer family for Claude to add** — start with LSTM (validates the abstraction with a non-tree model), or jump to PEFT (LoRA on a public time-series foundation model)? Recommendation: LSTM first because it's a known quantity from the legacy era; PEFT second once the abstraction is proven.
3. **Top-3 panel mid-month behavior** — if a brand-new iteration has +30% better gate result mid-month, does it bump the worst panel member immediately or wait for the monthly refresh? Recommend monthly to reduce churn during paper trade evaluation.

These can be decided as we hit them; not blockers for starting implementation.

## Implementation gate

Implementation can start. Suggest steps 1-3 first (search_space + feedback.py SQLite + train_mode end-to-end with cron) so we have a closed loop running on its own before adding Claude mode complexity. Then 4-6 (prompt builder + Claude mode + ml_loop.sh wrapper). Then trainer families per #2 above.
