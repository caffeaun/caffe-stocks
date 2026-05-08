"""Per-trainer hyperparameter search spaces.

Train mode samples configurations from these definitions; Claude mode adds
new entries here when registering a new trainer.

Convention:
- list value → discrete choice (e.g. [3, 6, 9])
- tuple (low, high) → continuous range; ints if both endpoints are ints,
  floats otherwise.
"""
from __future__ import annotations
import numpy as np


SEARCH_SPACES: dict[str, dict] = {
    'lightgbm': {
        'num_leaves':         [15, 31, 63, 127],
        'max_depth':          [4, 6, 8, -1],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 500, 1000],
        'min_child_samples':  [20, 50, 100, 200],
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.6, 1.0),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.0, 1.0),
        'pos_class_weight':   (1.0, 8.0),
    },
    # LightGBM regression head: same EV-prediction objective as xgb_regressor
    # (the best structural change to date — #31, 5/7 positive windows) but with
    # leaf-wise tree growth + GOSS gradient sampling, which gives genuine
    # algorithmic diversity vs the three XGBoost-family heads. HP space mirrors
    # 'lightgbm' minus pos_class_weight (regression has no class imbalance).
    'lightgbm_regressor': {
        'num_leaves':         [15, 31, 63, 127],
        'max_depth':          [4, 6, 8, -1],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 500, 1000],
        'min_child_samples':  [20, 50, 100, 200],
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.6, 1.0),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.0, 1.0),
    },
    'xgboost': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
    },
    # Regression head: predicts per-trade pnl directly, optimises EV instead
    # of P(win). Wider depth/n_est range mirrors xgboost's space; same family,
    # so HP intuitions carry over.
    'xgb_regressor': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
    },
    # Bagged regression head: K=3..7 xgb_regressor learners with seed +
    # row-bootstrap diversity, scored by mean - conf_lambda*std across bags.
    # Same XGB-tree HP space as xgb_regressor (sub-bag HP intuitions carry
    # over) plus three ensemble knobs:
    #   n_bags ∈ {3, 5, 7} — variance reduction is ~1/K with diminishing
    #     returns past 5; 7 stays under wall-time budget (~7×5s/window×7
    #     windows ≈ 4 min training, well inside 30 min).
    #   conf_lambda ∈ [0.0, 3.0] — strength of std penalty. 0 = plain
    #     bagging (variance reduction without consensus filtering); ~1.0 =
    #     row-level prediction must clear ~1σ of bag-disagreement; >2.0 =
    #     defensive (often produces too few trades on noisy windows).
    #   bootstrap_frac ∈ [0.7, 1.0] — fraction of train rows sampled per
    #     bag (with replacement). 0.7 = aggressive decorrelation but small
    #     train-per-bag; 1.0 = classic bootstrap (~63% unique rows per bag,
    #     full sample size). The sweep should find the slope where consensus
    #     filtering peaks WR without starving trade count.
    'bagged_xgb_regressor': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'n_bags':             [3, 5, 7],
        'conf_lambda':        (0.0, 3.0),
        'bootstrap_frac':     (0.7, 1.0),
    },
    # Huber regression head: pseudo-Huber loss bounds gradient on outlier
    # pnl (target hits / stop-loss exits) so the model stops chasing rare big
    # wins the way reg:squarederror does. Same tree-family HP space as
    # xgb_regressor (HP intuitions carry over) plus huber_slope ∈ [0.02, 0.10]:
    #   0.02 ≈ MIN_PROFIT_PCT/2 — almost everything looks like a tail event,
    #     loss collapses toward L1 (median-like, robust but biased).
    #   0.10 ≈ trailing-trigger zone — quadratic in the dense middle, behaves
    #     like a slightly-clipped L2.
    # The sweep should find a slope where the gradient transition lands near
    # the WR=40% boundary the gate enforces.
    'xgb_huber_regressor': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'huber_slope':        (0.02, 0.10),
    },
    # Quantile-regression head: predicts P(quantile_alpha)(pnl|X). Same tree
    # family as xgb_regressor (HP intuitions carry over) plus the quantile
    # parameter itself. quantile_alpha ∈ [0.55, 0.85] keeps the prediction in
    # the upper-but-not-extreme range — alpha=0.5 collapses to median (often
    # negative when WR<50%), alpha>0.9 chases noise in the right tail.
    'xgb_quantile': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'quantile_alpha':     (0.55, 0.85),
    },
    # Dual-quantile head: trains TWO XGBRegressors at alpha_lower and
    # alpha_upper, ranks by upper - dd_penalty * max(0, -lower). Same tree-family
    # HP space as xgb_quantile plus three new dimensions:
    #   alpha_lower ∈ [0.15, 0.35] — left-tail estimator. Below 0.15 the
    #     quantile loss is dominated by sparse extreme losses (unstable on
    #     200-1000 row train chunks); above 0.35 the "downside" estimate
    #     starts overlapping the median and the penalty loses bite.
    #   alpha_upper ∈ [0.65, 0.85] — right-tail estimator. Mirror range of
    #     alpha_lower around the median; same xgb_quantile rationale.
    #   dd_penalty ∈ [1.0, 4.0] — strength of the downside subtraction.
    #     1.0 = symmetric weighting (upside and downside in same units);
    #     >2.0 = downside-averse selection. The whitepaper's MAX_DD=25%
    #     gate criterion suggests asymmetric aversion is justified.
    'xgb_dual_quantile': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'alpha_lower':        (0.15, 0.35),
        'alpha_upper':        (0.65, 0.85),
        'dd_penalty':         (1.0, 4.0),
    },
    # Ranker head: pairwise ranking loss with date-grouped training. Same XGB
    # tree-family HP space as xgb_quantile/xgb_dual_quantile (HP intuitions
    # carry over). No quantile_alpha or dd_penalty — the loss only sees
    # within-date pairwise pnl comparisons, no quantile threshold to tune.
    'xgb_ranker': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
    },
    # DART variant of xgb_ranker. Same tree-family HP space (HP intuitions
    # carry over from xgb_ranker) plus four DART-specific knobs:
    #   rate_drop ∈ [0.05, 0.25] — fraction of trees dropped per round.
    #     Below 0.05 collapses to plain GBT; above 0.25 destabilizes training
    #     on small (~1000-row) train chunks.
    #   skip_drop ∈ [0.3, 0.7] — probability a round skips dropping. Spans
    #     "mostly-DART" (0.3) to "mostly-GBT with occasional dropout" (0.7),
    #     letting the sweep find the right regularization-vs-progress mix.
    #   sample_type ∈ {'uniform', 'weighted'} — uniform draw vs residual-weighted.
    #   normalize_type ∈ {'tree', 'forest'} — rescaling rule for surviving trees.
    'xgb_ranker_dart': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'rate_drop':          (0.05, 0.25),
        'skip_drop':          (0.3, 0.7),
        'sample_type':        ['uniform', 'weighted'],
        'normalize_type':     ['tree', 'forest'],
    },
    # LightGBM lambdarank head: same date-grouped ranking task as xgb_ranker
    # but with leaf-wise tree growth + GOSS + delta-NDCG-weighted pairs that
    # emphasize top-of-list ordering. HP space mirrors 'lightgbm_regressor'
    # (the LGB-family analogue) — no quantile_alpha or dd_penalty since the
    # listwise loss only sees within-date pnl ordering, no threshold to tune.
    'lightgbm_ranker': {
        'num_leaves':         [15, 31, 63, 127],
        'max_depth':          [4, 6, 8, -1],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 500, 1000],
        'min_child_samples':  [20, 50, 100, 200],
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.6, 1.0),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.0, 1.0),
    },
    # Stacked-ranker ensemble: validation-fold-weighted blend of xgb_ranker,
    # lightgbm_ranker and xgb_dual_quantile with HARDCODED sub-trainer HPs (the
    # best-known configs from claude #80 / #96 / #58 respectively). Train mode
    # explores three knobs that change ensemble behavior without re-tuning the
    # frozen sub-models:
    #   weight_grid_step ∈ {0.05, 0.1, 0.2, 0.25} — granularity of the simplex
    #     search; 0.25 → ~15 combos (coarse), 0.05 → ~231 combos (fine but
    #     more prone to overfitting a small val fold)
    #   min_concordance ∈ [0.0, 0.15] — Spearman ρ floor below which the
    #     learned weights are rejected and the stacker falls back to uniform
    #     1/3 weights (defensive against noisy val folds)
    #   random_state — seed-stability sweep across the 3 sub-trainers
    'stacked_ranker': {
        'weight_grid_step':   [0.05, 0.1, 0.2, 0.25],
        'min_concordance':    (0.0, 0.15),
        'aggregation':        ['weighted_avg', 'weighted_max'],
        'random_state':       [1, 7, 17, 42, 100, 2026],
    },
    # When Claude mode adds new trainers (LSTM, LoRA, RL, ...), it appends here.
}


def sample(trainer: str, rng: np.random.RandomState | None = None) -> dict:
    """Draw one configuration for a given trainer."""
    if trainer not in SEARCH_SPACES:
        raise ValueError(f'No search space defined for {trainer!r}. '
                         f'Add one in models/search_spaces.py.')
    if rng is None:
        rng = np.random
    space = SEARCH_SPACES[trainer]
    cfg = {}
    for key, val in space.items():
        if isinstance(val, list):
            cfg[key] = type(val[0])(rng.choice(val))
        elif isinstance(val, tuple) and len(val) == 2:
            lo, hi = val
            if isinstance(lo, int) and isinstance(hi, int):
                cfg[key] = int(rng.randint(lo, hi + 1))
            else:
                cfg[key] = float(rng.uniform(lo, hi))
        else:
            raise ValueError(f'Bad search space value for {key}: {val!r}')
    return cfg


def list_trainers() -> list[str]:
    return sorted(SEARCH_SPACES.keys())
