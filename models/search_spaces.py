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
    # Win-ranker head: rank:ndcg with BINARY win label as relevance + ndcg@K
    # eval. Same XGB tree-family HP space as xgb_ranker (HP intuitions carry
    # over from the pairwise-ranking sibling) plus one new knob:
    #   ndcg_at ∈ {1, 2, 3} — top-K position the LambdaRank loss puts gradient
    #     weight on. K=2 matches MAX_OPEN_POSITIONS in return_gate.simulate_window
    #     (the gate picks top-K-per-date with K=2). K=1 = "single best of the
    #     day" (more aggressive top-of-list focus, but smaller effective gradient
    #     since only one row per date "matters"). K=3 = mild slack, useful if
    #     the pred top-2 sometimes shifts to top-3 on val. The sweep should
    #     find which K produces the cleanest gate alignment.
    'xgb_win_ranker': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'ndcg_at':            [1, 2, 3],
    },
    # EV-Gated Ranker: composes a binary-win XGBRanker (same tree-family HPs as
    # xgb_win_ranker) with a XGBRegressor predicting per-trade pnl, multiplied
    # at inference: final = ranker_prob × (ev_floor + (1-ev_floor) × σ(ev_scale ×
    # ev_pred)). The ranker contributes top-K-per-day ordering; the EV regressor
    # contributes regime abstention (negative-EV days squashed below the
    # threshold sweep). Sub-models share tree HPs for parsimony — joint sweep
    # first, split later if attribution is informative. Three new knobs:
    #   ev_scale ∈ [10, 100]: sigmoid steepness. ev_scale=30 means a +5% EV
    #     pred maps to gate≈0.82 and a -3% EV pred maps to gate≈0.29 — gentle
    #     enough to keep gradients meaningful, sharp enough to cleanly abstain.
    #     Below 10 the gate becomes too soft (negative-EV days still admit
    #     trades); above 100 the gate is essentially a hard step.
    #   ev_floor ∈ [0.0, 0.5]: minimum gate value. Lower floor = more
    #     aggressive abstention (negative-EV trades scored ≈ 0 × ranker_prob).
    #     0.5 disables abstention; 0.0 makes negative-EV trades unselectable.
    #   ndcg_at ∈ {1, 2, 3}: top-K position the LambdaRank loss weights.
    #     K=2 matches MAX_OPEN_POSITIONS in return_gate.simulate_window.
    'ev_gated_ranker': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'ev_scale':           (10.0, 100.0),
        'ev_floor':           (0.0, 0.5),
        'ndcg_at':            [1, 2, 3],
    },
    # Bagged EV-Gated Ranker: K=3..7 EV-gated rankers each fit on a bootstrap
    # row sample with seed diversity, scored as mean - conf_lambda*std across
    # bags. Same XGB tree-family + EV-gate HPs as ev_gated_ranker (HP intuitions
    # carry over from the unbagged sibling) plus three ensemble knobs:
    #   n_bags ∈ {3, 5, 7} — variance reduction is ~1/K with diminishing
    #     returns past 5; at 7 bags the per-iter wall-time is ~6 min, well
    #     under the 30 min budget.
    #   conf_lambda ∈ [0.0, 1.5] — strength of std penalty. 0 = plain
    #     bagging (pure variance reduction); ~0.5 = mild consensus filtering;
    #     >1.0 = aggressive abstention when bags disagree. Tighter range than
    #     bagged_xgb_regressor (which is in [0,3]) because predict_proba is
    #     bounded [0,1] here, so the std term is also smaller in absolute
    #     scale — λ=1.5 already pushes mean−std into clipped territory.
    #   bootstrap_frac ∈ [0.7, 1.0] — fraction of unique trading **dates**
    #     sampled per bag (with replacement; all rows of each sampled date
    #     are included, preserving date-group ranker integrity). 0.7 =
    #     aggressive regime decorrelation; 1.0 = classic bootstrap on
    #     dates (~63% unique dates per bag at full sample size). Date-level
    #     bootstrap is what drives the cross-bag std into a meaningful
    #     consensus signal — row-level bootstrap shatters the within-date
    #     pairwise structure the ranker depends on.
    'bagged_ev_gated_ranker': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'ev_scale':           (10.0, 100.0),
        'ev_floor':           (0.0, 0.5),
        'ndcg_at':            [1, 2, 3],
        'n_bags':             [3, 5, 7],
        'conf_lambda':        (0.0, 1.5),
        'bootstrap_frac':     (0.7, 1.0),
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
    # Magnitude-Weighted XGBoost Classifier: binary y∈{0,1} target with per-row
    # sample_weight = (|pnl|·magnitude_scale + base_weight) normalised to mean 1.
    # Same XGBoost-tree HP space as xgb_huber_regressor (HP intuitions carry over)
    # plus three weighting knobs:
    #   magnitude_scale ∈ [5, 30] — gradient amplification on high-|pnl| rows.
    #     5 ≈ marginal weighting (a +14% trade gets ~70% more weight than a +4%);
    #     30 ≈ aggressive weighting (a +14% trade gets ~3.5× a +4% trade).
    #     Beyond 30 the loss collapses onto target-hit rows only and other
    #     wins become noise.
    #   base_weight ∈ [0.3, 1.5] — floor for marginal-pnl rows. <0.3 starves
    #     trees of marginal-pnl samples (loss only sees outliers); >1.5
    #     drowns the magnitude signal back into uniform weighting.
    #   pos_class_weight ∈ [1.0, 6.0] — XGBoost scale_pos_weight to address
    #     the ~12% pos_rate from labels.py CLEAN_WIN constraint. Range tighter
    #     than xgboost's [1.0, 8.0] because the magnitude weighting already
    #     amplifies wins (which are typically larger |pnl| than losses post-
    #     stop-loss-truncation), so additional class-level upweighting risks
    #     redundant gradient.
    'xgb_magnitude_classifier': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'magnitude_scale':    (5.0, 30.0),
        'base_weight':        (0.3, 1.5),
        'pos_class_weight':   (1.0, 6.0),
    },
    # Strict-win: y = (pnl > 0) target, ~22% positive rate vs ~12% clean-win.
    # Less imbalance → pos_class_weight centered lower (1.0..3.5).
    'xgb_strict_win': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'magnitude_scale':    (5.0, 30.0),
        'base_weight':        (0.3, 1.5),
        'pos_class_weight':   (1.0, 3.5),
    },
    # Focal-loss XGBoost classifier: y = (pnl > 0) like xgb_strict_win, but the
    # loss is FL = -alpha_t · (1 - p_t)^gamma · log(p_t) instead of weighted
    # binary CE. Down-weights well-classified examples and forces gradient onto
    # hard regime-edge cases — addresses the W2/W5/W7 WR<40% saturation across
    # 16 strict_win HP sweeps. Same XGB tree-family HP space as xgb_strict_win
    # (intuitions carry over) plus two focal-loss knobs:
    #   focal_alpha ∈ [0.25, 0.75] — class-balance scalar. 0.5 = neutral; <0.5
    #     down-weights positive class (rarely wanted at 22% pos_rate); >0.5
    #     up-weights wins. The original Lin et al. paper uses 0.25 for highly
    #     imbalanced detection; for our 22% pos_rate the literature suggests
    #     a higher alpha (0.5-0.75) keeps the loss balanced after the focal
    #     down-weighting kicks in.
    #   focal_gamma ∈ [0.5, 4.0] — focusing strength. 0 = standard weighted
    #     CE; 2.0 is the canonical default (object detection). At gamma=4
    #     the loss becomes very peaky toward hard examples, which can starve
    #     the easy-class gradient on small windows. The sweep should find
    #     where the regime-generalization gain peaks before instability.
    'xgb_focal_loss': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.5),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'focal_alpha':        (0.25, 0.75),
        'focal_gamma':        (0.5, 4.0),
    },
    # Group-balanced focal-loss XGB classifier: stacks per-quarter
    # inverse-frequency sample weighting on top of focal loss. Forces the
    # classifier to weight every calendar quarter equally, attacking the
    # regime-shift failure mode that plain focal loss cannot reach (W2/W5/W7
    # WR<40% across 8 xgb_focal_loss iters #470-#477).
    #
    # Same XGB tree-family + focal-loss HP space as xgb_focal_loss, plus:
    #   group_balance_strength ∈ [0.0, 1.0] — interpolation between uniform
    #     weights (0.0 = falls back to plain focal loss) and full
    #     inverse-frequency balancing (1.0). Mid-range values let the sweep
    #     find a partial-balancing operating point if full rebalancing
    #     over-corrects on training-period dominant regimes. Center near 1.0
    #     to bias toward the structural hypothesis.
    'xgb_group_balanced_focal': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'focal_alpha':            (0.25, 0.75),
        'focal_gamma':            (0.5, 4.0),
        'group_balance_strength': (0.4, 1.0),
    },
    # Temporal-Mixup XGB classifier: synthetic train rows linearly interpolate
    # features+labels between two real samples drawn from temporally-distant
    # quarters. Trained with reg:logistic on real ∪ synthetic. Same XGB
    # tree-family HP space as xgb_focal_loss / xgb_group_balanced_focal (HP
    # intuitions carry over) plus three Mixup-specific knobs:
    #   mixup_alpha ∈ [0.1, 1.0] — Beta(α,α) shape parameter for the lambda
    #     mix coefficient. α<0.5 → U-shaped lambda (most synth rows are 90/10
    #     mixes, gentle augmentation); α=1.0 → uniform lambda (heavier
    #     blending). After the asymmetric max(λ, 1−λ) projection we apply,
    #     effective λ ∈ [0.5, 1.0]. The sweep should find where the
    #     regularisation gain peaks before label noise dominates.
    #   mixup_ratio ∈ [0.5, 2.0] — number of synthetic rows per real row.
    #     0.5 = mild augmentation (1.5× train data); 2.0 = aggressive
    #     (3× train data, ~10s extra fit time per window — well inside
    #     30 min budget). Above 2.0 the synthetic rows start dominating
    #     gradient and the model fits the interpolation manifold instead
    #     of the real distribution.
    #   mixup_min_quarters ∈ {0, 1, 2, 3} — minimum quarter-distance for the
    #     partner. 0 = vanilla Mixup (random partner anywhere); 1 = different
    #     quarter (the temporal-bridging hypothesis); 2-3 = more aggressive
    #     regime spanning, but on 6-month train windows often falls back to
    #     "any other quarter" because the constraint cannot be satisfied.
    'xgb_temporal_mixup': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'mixup_alpha':            (0.1, 1.0),
        'mixup_ratio':            (0.5, 2.0),
        'mixup_min_quarters':     [0, 1, 2, 3],
    },
    # Just-Train-Twice (JTT, Liu et al. 2021) XGB classifier: pass 1 ERM
    # identifier finds samples that ERM gets wrong, pass 2 retrains with those
    # samples upweighted. Implicit Group DRO without group labels — addresses
    # the W3/W6 summer-regime WR collapse that none of focal / group_balanced /
    # magnitude / temporal_mixup can reach (those reweight by static signals;
    # JTT reweights by model-discovered hard subset). Same XGB tree-family HP
    # space as xgb_focal_loss / xgb_temporal_mixup (HP intuitions carry over)
    # plus two JTT-specific knobs:
    #   lambda_up ∈ [10, 200] — upweight multiplier on misclassified pass-1
    #     rows. Liu et al. (2021) Table 1 reports best λ_up ∈ {20, 50, 100}
    #     across vision/NLI tasks; tabular literature (Idrissi et al. 2022)
    #     suggests slightly lower (~20-50) on smaller datasets. Range [10, 200]
    #     covers both — λ=10 is mild reweighting (ERM ≈ 80% of gradient still),
    #     λ=200 is aggressive (mistake gradient dominates 4-5×). The sweep
    #     should locate the slope where mistake-set fitting stops trading off
    #     against majority-set fit.
    #   pass1_estimators_frac ∈ [0.3, 0.7] — fraction of n_estimators used to
    #     train the identifier. <0.3 risks an under-trained identifier whose
    #     "mistakes" are noise rather than the hard subset; >0.7 risks an
    #     over-trained identifier with mistake_rate ≈ 0 (memorizes training
    #     set, JTT degrades to ERM). 0.5 is the JTT-paper default (early-stop
    #     halfway). The sweep should find the slope where mistake_rate lands
    #     in the 15-35% band that the paper validates as "informative hard
    #     subset, not pure noise".
    'xgb_jtt': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'lambda_up':              (10.0, 200.0),
        'pass1_estimators_frac':  (0.3, 0.7),
    },
    # Quarterly Group DRO: per-calendar-quarter logloss-based reweighting after
    # a pass-1 ERM identifier. The natural group-level analogue of JTT (which
    # weights ROWS by mistakes); the loss-driven analogue of group_balanced_focal
    # (which weights GROUPS by inverse-frequency). Same XGB tree-family HP space
    # as xgb_jtt / xgb_focal_loss (HP intuitions carry over) plus three knobs:
    #   dro_strength ∈ [0.0, 3.0] — exponent on relative quarter loss. 0 = ERM
    #     (uniform weights, falls back to plain XGBoost classifier); 1 = linear
    #     (worst-loss quarter ~2x weight if its loss is 2x mean); 2 = quadratic
    #     (worst quarter dominates); 3 = aggressive (mostly fits the lossy
    #     quarter). The sweep should find where regime-rebalance gain peaks
    #     before the lossy quarter starves the rest.
    #   pass1_estimators_frac ∈ [0.3, 0.7] — fraction of n_estimators used to
    #     train the identifier. Same rationale as xgb_jtt: <0.3 risks an
    #     under-trained identifier whose per-quarter logloss is noise; >0.7
    #     risks an over-trained identifier with logloss → 0 uniformly across
    #     quarters (DRO degrades to ERM).
    #   weight_smoothing ∈ [0.0, 0.5] — additive smoothing on the loss ratio,
    #     fraction of mean_R. 0 = no smoothing (pure relative-loss weighting,
    #     can collapse to zero on outlier-low-loss quarters); 0.5 = strong
    #     smoothing (dampens DRO toward uniform).
    'xgb_quarterly_dro': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'dro_strength':           (0.0, 3.0),
        'pass1_estimators_frac':  (0.3, 0.7),
        'weight_smoothing':       (0.0, 0.5),
    },
    # Top-K Classifier: y = (trade is in top-K of its date by pnl AND pnl > 0).
    # Direct alignment with the gate's top-K-per-date selection rule. ~2-3% pos
    # rate on K=2 → high pos_class_weight needed. Same XGB tree-family HP space
    # as xgb_focal_loss / xgb_strict_win (HP intuitions carry over) plus two
    # trainer-specific knobs:
    #   top_k ∈ {1, 2, 3} — K=2 matches MAX_OPEN_POSITIONS in the gate. K=1
    #     is more selective (the day's single best); K=3 admits a third pick
    #     per date which the gate then filters down. Sweep should find where
    #     the train-time relabelling matches the gate's actual selection.
    #   pos_class_weight ∈ [10, 50] — wider range than xgboost's [1.0, 8.0]
    #     because the K=2 positive rate is ~2-3% (vs 22% for strict-win, 12%
    #     for clean-win). At K=2 the analytic balanced weight is ~30-50; the
    #     sweep covers both moderate (10-15) and aggressive (40-50) regimes.
    'xgb_topk_classifier': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'top_k':                  [1, 2, 3],
        'pos_class_weight':       (10.0, 50.0),
    },
    # Adversarial-Validation-Reweighted XGBoost. Same XGB tree-family HP space
    # as xgb_quarterly_dro / xgb_focal_loss (HP intuitions carry over) plus
    # four AVR-specific knobs:
    #   adv_alpha ∈ [0.0, 3.0] — exponent applied to P(test-like)^alpha. 0=ERM
    #     (no AVR), 1=raw probability, >1=sharpen toward most-test-like rows,
    #     <1=soften. Sweep covers all three regimes; recency_huber's failure
    #     at decay=1.5 (#207) suggests the sweet spot is likely in [0.5, 1.5].
    #   adv_test_frac ∈ [0.15, 0.4] — fraction of train (by unique date) used
    #     as the pseudo-test for the inner AVR classifier. The gate's inner
    #     val cutoff is 0.20, so the natural anchor is ~0.25; smaller values
    #     give a sharper but noisier "future-like" signal, larger values
    #     dilute it.
    #   adv_n_estimators ∈ {50, 100, 200} — capacity of the inner classifier.
    #     Too high → overfits pseudo-labels and weights collapse to {0, 1}.
    #     Too low → can't separate distributions and AVR adds nothing.
    #   weight_clip ∈ [3.0, 20.0] — caps any single row's weight at K× the
    #     mean. Looser clips let extreme test-like rows dominate (gradient
    #     concentration); tighter clips spread the gradient. Importance-
    #     weighting under sparse domain overlap historically benefits from
    #     moderate clips (~5-10).
    'xgb_adv_val': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'adv_alpha':              (0.0, 3.0),
        'adv_test_frac':          (0.15, 0.40),
        'adv_n_estimators':       [50, 100, 200],
        'adv_max_depth':          [3, 4, 6],
        'weight_clip':            (3.0, 20.0),
    },
    # Meta-labeling (López de Prado, AFML Ch.3). Two-stage XGBoost: stage 1
    # predicts P(pnl > 0); stage 2 predicts P(pnl > 0 | features, stage1_pred)
    # using stage-1's OOF prediction as an additional input feature.
    # HP space follows the prior xgb_adv_val structure (XGB-tree family) but
    # with a SEPARATE space for stage 1 vs stage 2:
    #   - stage 2 typically wants tighter regularization (smaller depth, higher
    #     min_child_weight) — it's filtering on a smaller, late-half train pool
    #     and is more vulnerable to overfit. Defaults reflect this.
    #   - stage1_train_frac ∈ [0.4, 0.7] — fraction of unique train dates given
    #     to stage 1's first pass. <0.4 starves stage 1 of signal; >0.7 starves
    #     stage 2 of training data. Sweep covers the natural range around 0.5.
    #   - learning rates and depths drawn independently per stage so the sweep
    #     can find e.g. (deep stage 1 + shallow stage 2) or vice versa.
    'xgb_meta_label': {
        'stage1_max_depth':         [3, 4, 6, 8],
        'stage1_learning_rate':     (0.01, 0.10),
        'stage1_n_estimators':      [200, 400, 800],
        'stage1_min_child_weight':  [1, 5, 10, 20],
        'stage1_gamma':             (0.0, 0.5),
        'stage1_subsample':         (0.6, 1.0),
        'stage1_colsample_bytree':  (0.5, 0.9),
        'stage1_reg_alpha':         (0.0, 0.5),
        'stage1_reg_lambda':        (0.5, 2.0),
        'stage2_max_depth':         [2, 3, 4, 6],
        'stage2_learning_rate':     (0.01, 0.10),
        'stage2_n_estimators':      [200, 400, 800],
        'stage2_min_child_weight':  [5, 10, 20, 40],
        'stage2_gamma':             (0.0, 0.5),
        'stage2_subsample':         (0.6, 1.0),
        'stage2_colsample_bytree':  (0.5, 0.9),
        'stage2_reg_alpha':         (0.0, 0.5),
        'stage2_reg_lambda':        (0.5, 2.0),
        'stage1_train_frac':        (0.40, 0.70),
    },
    'xgb_meta_label_regime': {
        'stage1_max_depth':         [3, 4, 6, 8],
        'stage1_learning_rate':     (0.01, 0.10),
        'stage1_n_estimators':      [200, 400, 800],
        'stage1_min_child_weight':  [1, 5, 10, 20],
        'stage1_gamma':             (0.0, 0.5),
        'stage1_subsample':         (0.6, 1.0),
        'stage1_colsample_bytree':  (0.5, 0.9),
        'stage1_reg_alpha':         (0.0, 0.5),
        'stage1_reg_lambda':        (0.5, 2.0),
        'stage2_max_depth':         [2, 3, 4, 6],
        'stage2_learning_rate':     (0.01, 0.10),
        'stage2_n_estimators':      [200, 400, 800],
        'stage2_min_child_weight':  [5, 10, 20, 40],
        'stage2_gamma':             (0.0, 0.5),
        'stage2_subsample':         (0.6, 1.0),
        'stage2_colsample_bytree':  (0.5, 0.9),
        'stage2_reg_alpha':         (0.0, 0.5),
        'stage2_reg_lambda':        (0.5, 2.0),
        'stage1_train_frac':        (0.40, 0.70),
    },
    # MC-Dropout feature-mask classifier (xgb_mcdropout_classifier) — XGB tree
    # HP space mirrors xgb_strict_win (the parent class providing the y=pnl>0
    # alignment and per-row magnitude weighting). Three dropout-specific knobs:
    #   drop_rate ∈ [0.05, 0.40] — fraction of features NaN-masked per row per
    #     pass. Low end (0.05–0.10) yields conservative uncertainty estimates
    #     (most predictions look stable); high end (0.30–0.40) aggressively
    #     stress-tests feature dependence and may collapse mean toward 0.5
    #     for rows whose signal comes from any few-feature subset.
    #   n_dropout_passes ∈ {8, 12, 16, 24} — number of MC samples. 8 is the
    #     SE/std floor where std estimate becomes reliable (CI for std with
    #     n=8 is wide but useful for ranking); 24 is the compute ceiling
    #     (~1.5s per window × 7 windows ≈ 11s extra at predict, negligible).
    #   conf_lambda ∈ [0.0, 2.0] — std-penalty strength. 0 = plain MC mean
    #     (averages out predict noise but keeps fragile-feature predictions);
    #     0.5 = canonical Bayesian-style downweighting; 1.5+ = heavy penalty,
    #     only the most stable rows clear high thresholds.
    'xgb_mcdropout_classifier': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.01, 0.10),
        'n_estimators':       [200, 400, 800],
        'min_child_weight':   [1, 5, 10, 20],
        'gamma':              (0.0, 0.3),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (0.5, 2.0),
        'magnitude_scale':    (5.0, 30.0),
        'base_weight':        (0.3, 1.0),
        'pos_class_weight':   (1.5, 5.0),
        'drop_rate':          (0.05, 0.40),
        'n_dropout_passes':   [8, 12, 16, 24],
        'conf_lambda':        (0.0, 2.0),
    },
    # Diverse-objective rank-fusion ensemble (xgb_rank_fusion). Three internally
    # fit XGBoost bases (strict-win BCE, huber pnl regressor, quarterly DRO),
    # combined via within-batch quantile-rank geometric mean — no learned
    # weights. The fusion is fixed; only the SHARED base HPs are swept, plus
    # loss-specific knobs for each base. Keeping a SHARED tree-shape HP profile
    # across bases means train mode doesn't blow up into a 30-dimensional
    # search; the bases get their diversity from objective and per-base seed
    # offsets, not from divergent depth/lr settings. n_estimators capped at 500
    # because three boosters run sequentially per window — at 800 the gate
    # could breach the 30 min wall on dense splits.
    'xgb_rank_fusion': {
        'max_depth':              [3, 4, 6],
        'learning_rate':          (0.02, 0.08),
        'n_estimators':           [200, 300, 500],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.3),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'pos_class_weight':       (1.5, 5.0),
        'magnitude_scale':        (5.0, 25.0),
        'base_weight':            (0.3, 1.0),
        'huber_slope':            (0.02, 0.10),
        'dro_strength':           (0.5, 2.0),
        'pass1_estimators_frac':  (0.3, 0.7),
        'weight_smoothing':       (0.05, 0.30),
        # Hard abstention floor on per-base rank quantile (iter #1499).
        # 0.0 = legacy fusion (matches the historical 6/7 config at iter 661).
        # 0.3 = the new bear-regime breakthrough setting — unlocked W5
        # (0/20 across the prior 20 iters → +35.6% ann, 40.9% WR) at the cost
        # of trade-count in benign W7 windows. Sweep should explore whether
        # 0.15–0.25 retains the W5 unlock without starving W1/W7.
        'abstain_min_q':          (0.0, 0.35),
    },
    # Regime-blend classifier (xgb_regime_blend). Two parallel strict-win
    # XGB heads — generalist on FULL train + bear-specialist on the lower-
    # quantile-by-market_breadth_adv subset — softly blended at predict by
    # per-row regime feature. Single tree-family HP set (both heads share),
    # plus three blend-mechanic knobs:
    #   bear_quantile ∈ [0.25, 0.55] — fraction of training rows labelled
    #     "bear" for the specialist head. 0.25 = tight (specialist sees the
    #     deepest bear rows only, but starves on small windows); 0.55 = loose
    #     (specialist sees ~half of train, blurs the bear/bull discrimination
    #     but keeps sample size up). The sweep should find where bear-only
    #     learning beats noise.
    #   temperature ∈ [0.05, 0.50] — softness of the predict-time blend.
    #     Small temp = nearly hard routing (bear rows → 100% specialist,
    #     bull rows → 100% generalist); large temp = mostly 50/50 averaging,
    #     specialist contribution barely felt. 0.15 (default) gives a ~70/30
    #     blend at 1 IQR away from the threshold.
    #   regime_feature_idx ∈ {15, 16, 19, 12} — which feature column drives
    #     the blend. 15 = last__market_breadth_adv (default), 16 =
    #     last__market_breadth_above_sma20, 19 = last__market_new_highs,
    #     12 = last__sector_breadth. All four are bounded [0,1] regime
    #     features from feature_eng.py; the sweep should find which signal
    #     cleanest separates bear/bull when applied as the blend gate.
    'xgb_regime_blend': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'magnitude_scale':        (5.0, 25.0),
        'base_weight':            (0.3, 1.0),
        'pos_class_weight':       (1.5, 4.0),
        'bear_quantile':          (0.25, 0.55),
        'temperature':            (0.05, 0.50),
        'regime_feature_idx':     [15, 16, 19, 12],
    },
    # Recency-consensus classifier (xgb_recency_consensus). Two strict-win
    # XGB heads — one trained with uniform sample weights (full-history view),
    # one with exp-decay recency weighting (recent-history view) — geo-mean
    # fused at predict. Single shared XGB tree-shape HP set + magnitude
    # weighting (same as strict_win), plus the two recency-specific knobs:
    #   recency_halflife_days ∈ [20, 150] — half-life in calendar days for
    #     the exp decay on the recent head's sample weights. 20d gives a
    #     SHARP recent focus (last ~1 mo of a 6-mo train dominates); 150d
    #     gives a SOFT recent focus (gentle tilt toward the second half of
    #     train). Default 60d. The sweep should find the half-life that
    #     best previews each test window's regime — too short and the
    #     recent head has high variance from small ESS; too long and the
    #     two heads collapse to the same model (consensus disappears).
    #   min_recent_weight ∈ [0.05, 0.40] — floor on the recency weight for
    #     the OLDEST train rows. 0.05 nearly drops them (high recent bias);
    #     0.40 keeps them as a strong regularizer. Default 0.20.
    'xgb_recency_consensus': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'magnitude_scale':        (5.0, 25.0),
        'base_weight':            (0.3, 1.0),
        'pos_class_weight':       (1.5, 4.0),
        'recency_halflife_days':  (20.0, 150.0),
        'min_recent_weight':      (0.05, 0.40),
    },
    # Day-quality consensus (xgb_day_quality_consensus). Inherits all the
    # recency_consensus knobs and adds two head-C specific axes:
    #   day_quality_floor ∈ [0.05, 0.60] — floor for head C's effective
    #     output before the geo-mean. 0.05 = head C can suppress a score
    #     to ~5% of its un-gated value on the worst days; 0.60 = at most
    #     40% suppression. 0.30 default.
    #   day_quality_weight ∈ [0.25, 2.5] — exponent on head C in the
    #     weighted geo-mean (p_a * p_b * p_dq_eff^w)^(1/(2+w)). w=0.25
    #     keeps head C as a gentle tiebreaker; w=2.5 makes head C
    #     dominant (risk: regime miscalls dictate whole batch).
    'xgb_day_quality_consensus': {
        'max_depth':              [3, 4, 6, 8],
        'learning_rate':          (0.01, 0.10),
        'n_estimators':           [200, 400, 800],
        'min_child_weight':       [1, 5, 10, 20],
        'gamma':                  (0.0, 0.5),
        'subsample':              (0.6, 1.0),
        'colsample_bytree':       (0.5, 0.9),
        'reg_alpha':              (0.0, 0.5),
        'reg_lambda':             (0.5, 2.0),
        'magnitude_scale':        (5.0, 25.0),
        'base_weight':            (0.3, 1.0),
        'pos_class_weight':       (1.5, 4.0),
        'recency_halflife_days':  (20.0, 150.0),
        'min_recent_weight':      (0.05, 0.40),
        'day_quality_floor':      (0.05, 0.60),
        'day_quality_weight':     (0.25, 2.5),
    },
    # First NN trainer post-pyc loss. The HP space is intentionally small so
    # train mode can converge quickly. The two structural knobs that matter
    # most are dropout (regularization vs underfit) and pos_class_weight
    # (precision vs recall trade-off — XGB family converged around 1.5-3.5
    # for this dataset).
    'torch_attentive_mlp': {
        'hidden_dim':         [64, 128, 256],
        'bottleneck_dim':     [32, 64, 128],
        'dropout':            (0.10, 0.50),
        'learning_rate':      (5e-4, 3e-3),
        'weight_decay':       (1e-5, 1e-3),
        'batch_size':         [256, 512, 1024],
        'max_epochs':         [30, 50, 80],
        'patience':           [4, 6, 10],
        'pos_class_weight':   (1.0, 4.0),
    },
    # Iter #711 — sequence-aware GRU. Consumes raw (N, 20, F) so the model
    # can read temporal regime evolution rather than the 4-stat aggregate.
    # Kept small (hidden ≤ 128, max_epochs ≤ 16) to fit per-window training
    # under the 30-min wall budget across 7 walk-forward windows on CPU.
    'torch_seq_gru': {
        'hidden_dim':         [32, 48, 64, 96],
        'dropout':            (0.10, 0.45),
        'learning_rate':      (5e-4, 3e-3),
        'weight_decay':       (1e-5, 1e-3),
        'batch_size':         [256, 512, 1024],
        'max_epochs':         [8, 12, 16],
        'patience':           [3, 4, 6],
        'pos_class_weight':   (1.0, 3.5),
    },
    # Iter #712 — Transformer encoder over time + learnable [CLS] token.
    # `d_model` choices are all divisible by every `nhead` choice (48 % 4,
    # 64 % 4, 64 % 8, 96 % 4, 96 % 8, 128 % 4, 128 % 8 all clean) so the
    # constructor's clamp never triggers under sampled configs. Higher
    # dropout floor than GRU (0.20 vs 0.10) since attention has more
    # capacity and iter #711's failure mode was over-aggressive predictions
    # (WR 14-37%, never DD-bound) — the prior is "regularise harder".
    'torch_seq_transformer': {
        'd_model':            [48, 64, 96, 128],
        'nhead':              [4, 8],
        'num_layers':         [1, 2, 3],
        'dim_feedforward':    [96, 128, 192, 256],
        'dropout':            (0.20, 0.50),
        'learning_rate':      (3e-4, 2e-3),
        'weight_decay':       (1e-5, 1e-3),
        'batch_size':         [256, 512, 1024],
        'max_epochs':         [8, 12, 16],
        'patience':           [3, 4, 6],
        'pos_class_weight':   (1.0, 3.5),
    },
    # Iter #713 — deep-ensemble GRU. Same per-member space as torch_seq_gru
    # (hidden_dim / dropout / pos_class_weight / lr) plus the two new
    # ensemble knobs: n_models (3-7, the precision-vs-time trade-off; K=5
    # default fits ~14 min of training in the 30-min wall) and
    # disagreement_penalty (0.5-2.5, the λ in score = mean − λ·std; higher
    # λ = stricter abstention, fewer trades but higher precision).
    # max_epochs trimmed to 8-12 vs single-GRU 8-16 because K members
    # multiply runtime; rely on ensemble averaging to recover the variance
    # reduction that more epochs gave a single net.
    'torch_seq_gru_ensemble': {
        'n_models':              [3, 5, 7],
        'hidden_dim':            [32, 48, 64, 96],
        'dropout':               (0.10, 0.45),
        'learning_rate':         (5e-4, 3e-3),
        'weight_decay':          (1e-5, 1e-3),
        'batch_size':            [256, 512, 1024],
        'max_epochs':            [8, 10, 12],
        'patience':              [3, 4],
        'pos_class_weight':      (1.0, 3.5),
        'disagreement_penalty':  (0.5, 2.5),
    },
    # Iter #713 — deep-ensemble GRU with HARD abstention by epistemic
    # uncertainty quantile. `abstain_quantile` is the key knob: it is the q
    # such that τ = quantile(val_std, q); rows with test-time std > τ are
    # forced to score -1e9 (skipped by every gate threshold). Lower q =
    # stricter (more rows abstain, fewer trades, higher WR if signal is
    # real). Range [0.30, 0.80] explores the WR/n_trades trade-off; default
    # 0.50 abstains on the noisiest 50% of predictions. Same per-member
    # spaces as torch_seq_gru_ensemble; n_models capped at 7 for wall-time.
    'torch_seq_gru_abstain': {
        'n_models':              [3, 5],
        'hidden_dim':            [32, 48, 64, 96],
        'dropout':               (0.10, 0.45),
        'learning_rate':         (5e-4, 3e-3),
        'weight_decay':          (1e-5, 1e-3),
        'batch_size':            [256, 512, 1024],
        'max_epochs':            [6, 8, 10],
        'patience':              [3, 4],
        'pos_class_weight':      (1.0, 3.5),
        'abstain_quantile':      (0.30, 0.80),
        # Iter #1737: per-date mean-prob abstention quantile. 0.0 disables.
        # 0.40 cuts ~40% of low-confidence days entirely — bear-regime kill.
        'date_abstain_q':        (0.0, 0.50),
    },
    # Iter #714 — single GRU + per-date XGB day-quality gate. GRU spaces are
    # identical to torch_seq_gru so prior single-GRU HP wins transfer. New
    # knobs are the day-gate XGB (shallow tree, conservative HPs since ~120
    # unique training dates) and the blend coefficient that controls how
    # aggressively day-quality demotes scores on predicted-bad days:
    #   blend=0.0  → score = p_gru * p_day            (hardest demotion)
    #   blend=0.5  → score = p_gru * (0.5 + 0.5*p_day) (default, soft)
    #   blend=1.0  → score = p_gru                    (gate disabled)
    'torch_seq_gru_day_gate': {
        'hidden_dim':                 [32, 48, 64, 96],
        'dropout':                    (0.10, 0.45),
        'learning_rate':              (5e-4, 3e-3),
        'weight_decay':               (1e-5, 1e-3),
        'batch_size':                 [256, 512, 1024],
        'max_epochs':                 [8, 12, 16],
        'patience':                   [3, 4, 6],
        'pos_class_weight':           (1.0, 3.5),
        'day_gate_max_depth':         [2, 3, 4],
        'day_gate_n_estimators':      [40, 60, 80, 120],
        'day_gate_learning_rate':     (0.03, 0.10),
        'day_gate_min_child_weight':  (1.0, 10.0),
        'day_gate_blend':             (0.20, 0.95),
    },
    # Iter #734 — sklearn ExtraTreesClassifier. Genuinely different inductive
    # bias from every XGB/LGBM variant in the registry: at each split it picks
    # a RANDOM threshold (not the greedy-optimal one) on a random feature
    # subset, then averages many fully-grown trees. Hypothesis: the XGB family
    # has plateaued at 5-6/7 because greedy splits latch onto training
    # patterns that don't transfer through W5 (bull→bear flip) and W7 (recent
    # foreign-flow-divergent rally). Extra Trees' built-in randomisation
    # softens that overfit.
    #
    # HP space — kept small (8 knobs) so train mode can cover it in ~50 draws:
    #   n_estimators ∈ {300, 500, 800, 1200} — more trees = less variance;
    #     wall-time is ~1-3 s/window even at 1200 (parallel n_jobs=-1).
    #   max_depth ∈ {0=None, 8, 12, 20} — None = fully grown (classic ExtraTrees);
    #     limited depth regularises further.
    #   min_samples_leaf ∈ {5, 10, 20, 50} — leaf-size regulariser; ~20 default
    #     keeps leaves stable on ~80k training rows.
    #   min_samples_split ∈ {2, 10, 20}
    #   max_features ∈ {'sqrt', 'log2', 0.3, 0.5} — fraction or rule for
    #     candidate features at each split (sklearn accepts float in [0,1]).
    #   bootstrap ∈ {False, True} — False = classic ExtraTrees (Geurts 2006),
    #     True = combines random-split + bag-row variance reduction.
    #   pos_class_weight ∈ [1.0, 4.0] — handles the ~25% positive label rate.
    'sklearn_extra_trees': {
        'n_estimators':       [300, 500, 800, 1200],
        'max_depth':          [0, 8, 12, 20],
        'min_samples_leaf':   [5, 10, 20, 50],
        'min_samples_split':  [2, 10, 20],
        'max_features':       ['sqrt', 'log2'],
        'bootstrap':          [0, 1],
        'pos_class_weight':   (1.0, 4.0),
    },
    # Logistic regression with elastic net + polynomial interaction features.
    # Genuinely new inductive bias vs the rest of the registry (no other linear
    # model). Convex MLE objective + smooth logistic surface lower variance for
    # small-sample regimes (W1) and extrapolate more sanely on out-of-distribution
    # rows (W5 hostile-bear). Train mode sweep knobs:
    #   C ∈ [0.01, 10]   — inverse reg strength; smaller = stronger penalty,
    #     more useful when degree=2 polynomial features explode dimensionality.
    #   l1_ratio ∈ [0.1, 0.9] — elasticnet mix; higher = more L1 sparsity.
    #   pos_class_weight ∈ [1.0, 5.0] — handles the ~11% positive label rate.
    #   degree ∈ {1, 2} — 1 = pure linear (fast, 96 features); 2 = pairwise
    #     interactions (4656 features, ~30s/window fit, captures "regime × signal").
    #   max_iter ∈ {500, 1000, 2000} — SAGA convergence budget.
    'logistic_elastic_net': {
        'C':                 (0.01, 10.0),
        'l1_ratio':          (0.1, 0.9),
        'pos_class_weight':  (1.0, 5.0),
        'degree':            [1, 2],
        'max_iter':          [500, 1000, 2000],
    },
    # KNN classifier — non-parametric, memory-based, locally adaptive.
    # First trainer in the registry that does not fit a global parametric
    # model. Targets W1's small-train slice (8k rows) where parametric
    # variance dominates, and W3/W4's OOD bear-regime rows where global
    # decision surfaces (linear or axis-aligned trees) misbehave.
    #   n_neighbors ∈ {25..400} — bias/variance dial. Smaller = more local,
    #     higher variance but adapts to regime shifts; larger = smoother,
    #     more reliable probabilities. 100 is the bias-variance sweet spot
    #     given ~10% positive label rate (≥10 positives among 100 neighbors).
    #   weights ∈ {uniform, distance} — distance-weighted sharpens the
    #     probability estimate; uniform is the K-vote classic.
    #   metric ∈ {manhattan, euclidean} — L1 robust to heavy-tailed
    #     features (atr_pct, volume_ratio outliers); L2 is the textbook
    #     default.
    #   pca_components ∈ {0, 16, 32, 64} — 0 = full 96-d (no reduction);
    #     >0 = PCA reduction (mitigates curse of dimensionality, collapses
    #     redundant last/mean/std/dev axes).
    #   leaf_size ∈ {20, 30, 50} — ball_tree internal — query speed knob;
    #     accuracy invariant.
    'knn_classifier': {
        'n_neighbors':        [25, 50, 100, 200, 400],
        'weights':            ['uniform', 'distance'],
        'metric':             ['manhattan', 'euclidean'],
        'pca_components':     [0, 16, 32, 64],
        'leaf_size':          [20, 30, 50],
    },
    # QDA — generative probabilistic classifier (iter #772). reg_param is the
    # primary lever (covariance shrinkage toward sphericity); pca_components
    # de-redundifies the 96-d aggregate; tol gates the rank-deficiency floor.
    'qda_classifier': {
        'reg_param':          [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
        'tol':                [1e-5, 1e-4, 1e-3],
        'pca_components':     [0, 16, 32, 48, 64],
    },
    # GaussianNB — feature-independence generative (iter #788). var_smoothing
    # is the primary regularization (added to variance for numerical
    # stability; tiny=brittle, large=quasi-uniform); pca_components
    # decorrelates inputs so the independence assumption holds by
    # construction; prior_class1 explicitly tilts the prior (0 = MLE from
    # the training base rate ≈ 0.22).
    'gaussian_nb': {
        'var_smoothing':      [1e-12, 1e-10, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
        'pca_components':     [0, 16, 32, 48, 64],
        'prior_class1':       [0.0, 0.10, 0.15, 0.20, 0.30, 0.40],
    },
    # Iter #804: XGB classifier with CV-fit isotonic post-hoc calibration.
    # Hypothesis is that GaussianNB and QDA saturate scores in the bear
    # windows (W3/W4/W5) so threshold sweep does not bite; the calibrator
    # makes thr=0.55 mean "training WR ≥ 55%" by construction. HP space
    # mirrors xgboost (the underlying classifier) plus calibration knobs
    # — method ∈ {isotonic, sigmoid} and cv ∈ {3, 5}. Tighter reg_lambda
    # range than xgboost since CV-folds are smaller and overfit-prone
    # without early stopping.
    'xgb_iso_calibrated': {
        'max_depth':          [3, 4, 6, 8],
        'learning_rate':      (0.02, 0.08),
        'n_estimators':       [200, 300, 500],
        'min_child_weight':   [5, 10, 20, 40],
        'gamma':              (0.0, 0.4),
        'subsample':          (0.6, 1.0),
        'colsample_bytree':   (0.5, 0.9),
        'reg_alpha':          (0.0, 0.5),
        'reg_lambda':         (1.0, 4.0),
        'calib_cv':           [3, 5],
        'calib_method':       ['isotonic', 'sigmoid'],
    },
    # Kernel logistic regression via Nyström RBF approximation (iter #819).
    # Hypothesis: first kernel-space classifier in the registry — smooth
    # nonlinear local-conjunction modelling that trees fragment and linear
    # methods miss. Key knobs:
    #   n_components ∈ {150, 300, 500, 800} — Nyström landmark count. More
    #     landmarks → better kernel approximation but slower fit; 300 is
    #     the default sweet spot for 96-d aggregates.
    #   gamma ∈ {0.0 ('scale'), 0.01, 0.05, 0.1, 0.5} — RBF bandwidth.
    #     0.0 = sklearn 'scale' (= 1/(n_features * X.var())); smaller gamma
    #     = wider kernel (more global similarity); larger gamma = narrower
    #     (more local).
    #   C ∈ {0.1, 0.5, 1.0, 3.0, 10.0} — inverse L2 regularization on LR
    #     head. Small C = strong regularization; large C = fits training
    #     decision boundary tightly.
    #   pca_components ∈ {0, 16, 32, 64} — pre-kernel decorrelation.
    #     0 = use full 96-d (default; preserves regime feature scales);
    #     >0 = orthogonal projection collapses redundant temporal aggregates.
    #   class_weight ∈ {'balanced', 'none'} — re-weight loss by class
    #     frequency. balanced compensates for the ~22% positive class
    #     without resampling.
    'kernel_logreg': {
        'n_components':       [150, 300, 500, 800],
        'gamma':              [0.0, 0.01, 0.05, 0.1, 0.5],
        'C':                  [0.1, 0.5, 1.0, 3.0, 10.0],
        'pca_components':     [0, 16, 32, 64],
        'class_weight':       ['balanced', 'none'],
        # iter #1300 — CV calibration. 'isotonic' = non-parametric monotone
        # remap; 'sigmoid' = Platt scaling. iter #1300 default sweep showed
        # isotonic boosts bear-regime W6 (ann +43% → +358%) and lifts W1/W5/
        # W7 WR by 3-13pp, but regresses W3/W4 (calibration smooths
        # discriminative signal in bull regimes). Keep 'none' in the grid so
        # the historical 4/7 sweep peak (gamma=0.5, pca=32) remains reachable.
        'calibrate':          ['none', 'isotonic', 'sigmoid'],
        # iter #1515 — multi-scale Nyström. When both > 0, a second RBF map
        # at a different gamma is concatenated with the primary via
        # FeatureUnion. 0 = single-scale (legacy). The two scales together
        # let the LR head separate fine-grained discrimination (lower gamma)
        # from broader regime structure (higher gamma).
        'gamma_secondary':           [0.0, 0.05, 0.5, 2.0, 5.0],
        'n_components_secondary':    [0, 200, 400, 600],
    },
    # Full-Bayesian Gaussian Process Classifier (iter #835). Knobs:
    #   n_inducing ∈ {300, 600, 1000} — subsample size for the exact kernel
    #     matrix (O(N^3) in n_inducing). 600 ≈ 15s per fold; 1000 ≈ 50s.
    #   length_scale ∈ {0.5, 1.0, 2.0, 5.0} — RBF initial length-scale; the
    #     optimizer refines it via marginal likelihood within
    #     [length_scale_bounds_lo, length_scale_bounds_hi].
    #   constant_value ∈ {0.5, 1.0, 2.0} — ConstantKernel amplitude prior.
    #   n_restarts_optimizer ∈ {0, 1, 3} — number of random restarts for the
    #     marginal-likelihood optimizer; higher = better hyperparameters
    #     but slower fit. 1 is a good default.
    #   max_iter_predict ∈ {50, 100, 200} — Laplace approximation Newton
    #     iterations. 100 default is normally enough; raise if posterior
    #     looks under-converged.
    # gaussian_process_classifier (iter #1099 structural rewrite):
    #   n_components ∈ {10, 15, 20, 30} — PCA target dims. Lower = more
    #     aggressive denoising; higher = retains more variance. 20 covers
    #     ~70-85% of variance on the 96-d aggregate panel (empirically).
    #   kernel_type ∈ {'matern', 'rbf'} — Matern(nu=2.5) is the new default
    #     (finite-differentiability prior); 'rbf' kept for ablation only.
    #   matern_nu ∈ {1.5, 2.5} — Matern smoothness. 1.5 = once-differentiable
    #     (rougher), 2.5 = twice-differentiable (smoother, closer to RBF).
    #   noise_level ∈ {0.1, 0.5, 1.0, 2.0} — WhiteKernel initial noise.
    #     Higher decouples the length-scale from per-point fitting.
    #   length_scale_bounds_hi ∈ {3.0, 10.0, 30.0} — Matern length scale cap.
    #     With PCA(20) the optimal scale is empirically smaller than 96-d,
    #     so we cap tighter to prevent the iter-1015 saturation.
    'gaussian_process_classifier': {
        'n_inducing':              [600, 1000, 1500],
        'n_components':            [10, 15, 20, 30],
        'kernel_type':             ['matern', 'rbf'],
        'matern_nu':               [1.5, 2.5],
        'length_scale':            [0.5, 1.0, 2.0, 5.0],
        'length_scale_bounds_lo':  [1e-2, 1e-1, 3e-1],
        'length_scale_bounds_hi':  [3.0, 1e1, 3e1],
        'noise_level':             [0.1, 0.5, 1.0, 2.0],
        'noise_level_bounds_lo':   [1e-4, 1e-3, 1e-2],
        'noise_level_bounds_hi':   [1.0, 1e1, 1e2],
        'constant_value':          [0.5, 1.0, 2.0],
        'n_restarts_optimizer':    [0, 1, 3],
        'max_iter_predict':        [50, 100, 200],
    },
    # MLP classifier (iter #851). Knobs:
    #   hidden_layer_1 ∈ {32, 64, 128, 256} — first hidden width. 64 default
    #     gives ~6k params on 96-d input; 256 explodes to ~25k (risk of
    #     overfit on ~30k-row windows).
    #   hidden_layer_2 ∈ {0, 16, 32, 64} — second hidden width. 0 collapses
    #     to a single hidden layer.
    #   alpha ∈ {1e-4, 1e-3, 5e-3, 1e-2} — L2 regularization strength. Larger
    #     = smoother decision surface; the diagnosis-cited W4/W5 anti-
    #     selection should respond to higher alpha.
    #   learning_rate_init ∈ {3e-4, 1e-3, 3e-3} — Adam LR. 1e-3 sklearn
    #     default; lower trades training speed for stability.
    #   max_iter ∈ {150, 300, 500} — epoch cap. Early stopping usually cuts
    #     well below the cap; raise only if val loss is still trending.
    #   batch_size ∈ {128, 256, 512} — minibatch size for Adam. Smaller =
    #     noisier gradient (implicit regularizer); larger = faster.
    #   activation ∈ {'relu', 'tanh'} — non-linearity. relu default; tanh
    #     bounds activations and can stabilize on noisy financial features.
    #   n_iter_no_change ∈ {10, 15, 25} — early-stopping patience. Higher
    #     gives the optimizer more time to escape plateaus.
    'mlp_classifier': {
        'hidden_layer_1':       [32, 64, 128, 256],
        'hidden_layer_2':       [0, 16, 32, 64],
        'alpha':                [1e-4, 1e-3, 5e-3, 1e-2],
        'learning_rate_init':   [3e-4, 1e-3, 3e-3],
        'max_iter':             [150, 300, 500],
        'batch_size':           [128, 256, 512],
        'activation':           ['relu', 'tanh'],
        'n_iter_no_change':     [10, 15, 25],
    },
    # When Claude mode adds new trainers (LSTM, LoRA, RL, ...), it appends here.
    'tabpfn_v25': {
    'n_estimators':          [2, 4, 8],
    'softmax_temperature':   (0.5, 1.5),
    'balance_probabilities': [False, True],
    'average_before_softmax':[False, True],
    'max_train_rows':        [10000, 20000, 30000],
    'random_state':          [42, 7, 1337],
},
    # models/search_spaces.py — entry to add to SEARCH_SPACES dict
'torch_time_moe': {
    'hidden_dim':         [64, 128, 256],
    'dropout':            (0.0, 0.4),
    'learning_rate':      (1e-4, 1e-2),
    'weight_decay':       (1e-6, 1e-3),
    'batch_size':         [128, 256, 512],
    'encode_batch_size':  [32, 64, 128],
    'epochs':             [5, 10, 20],
    'patience':           [2, 3, 5],
    'use_raw_features':   [True, False],
    'embed_pool':         ['mean', 'last'],
    'random_state':       [42, 7, 1337],
},
    # models/search_spaces.py — entry to add to SEARCH_SPACES dict
'tabm_classifier': {
    'k':            [16, 32, 64],
    'n_blocks':     [2, 3, 4],
    'd_block':      [128, 256, 512, 768],
    'dropout':      (0.0, 0.3),
    'lr':           (1e-4, 5e-3),
    'weight_decay': (1e-5, 1e-1),
    'batch_size':   [256, 512, 1024],
    'max_epochs':   [60, 120, 200],
    'patience':     [6, 12, 20],
    'grad_clip':    [0.5, 1.0, 2.0],
    'seed':         [0, 1, 42, 2026],
},
# TabM with Lower-Confidence-Bound scoring (iter #921). Same training space as
# tabm_classifier plus `disagreement_penalty` — the λ in score = mean − λ·std
# over the k=32 BatchEnsemble members. λ=0 reduces to plain TabM; higher λ
# trades trade-count for win-rate. Range mirrors torch_seq_gru_ensemble.
'tabm_lcb': {
    'k':                     [16, 32, 64],
    'n_blocks':              [2, 3, 4],
    'd_block':               [128, 256, 512, 768],
    'dropout':               (0.0, 0.3),
    'lr':                    (1e-4, 5e-3),
    'weight_decay':          (1e-5, 1e-1),
    'batch_size':            [256, 512, 1024],
    'max_epochs':            [60, 120, 200],
    'patience':              [6, 12, 20],
    'grad_clip':             [0.5, 1.0, 2.0],
    'disagreement_penalty':  (0.1, 2.5),
    'seed':                  [0, 1, 42, 2026],
},
# HistGradientBoostingClassifier with monotonic constraints (iter #936).
# `use_monotonic` toggles the constraint vector — sweeping ON vs OFF lets
# train mode quantify the constraint's contribution. pos_class_weight
# spans the same range as sklearn_extra_trees / xgb_iso_calibrated.
'histgb_monotonic': {
    'max_iter':            [200, 400, 800],
    'max_leaf_nodes':      [15, 31, 63, 127],
    'learning_rate':       (0.01, 0.10),
    'min_samples_leaf':    [20, 50, 100, 200],
    'l2_regularization':   (0.0, 5.0),
    'max_bins':            [127, 255],
    'n_iter_no_change':    [10, 20, 30],
    'pos_class_weight':    (1.0, 5.0),
    'use_monotonic':       [True, False],
    'random_state':        [42, 7, 1337],
},
    'torch_itransformer': {
    'dim':           [64, 128, 192],
    'depth':         [2, 3, 4],
    'heads':         [2, 4, 8],
    'dim_head':      [16, 32, 64],
    'dropout':       (0.05, 0.4),
    'learning_rate': (1e-4, 5e-3),
    'weight_decay':  (1e-6, 1e-3),
    'batch_size':    [512, 1024],
    'epochs':        [8, 12, 18],
    'patience':      [3, 4, 6],
    'pos_weight':    (1.0, 1.8),
},
'histgb_monotonic_bagged': {
    'n_bags':            [3, 5, 7],
    'aggregation':       ['median', 'mean', 'lcb', 'trimmed_lcb'],
    'conf_lambda':       (0.0, 1.5),
    'bag_frac':          (0.85, 1.0),
    'max_iter':          [200, 400],
    'max_leaf_nodes':    [15, 31, 63],
    'learning_rate':     (0.015, 0.06),
    'min_samples_leaf':  [50, 100],
    'l2_regularization': (0.0, 3.0),
    'pos_class_weight':  (1.5, 4.5),
    'random_state':      [42, 7, 1337],
},
# TabNet (Arik & Pfister, AAAI 2021) — sparse per-row feature selection via
# sparsemax-attention mask, sequential decision steps. Different inductive
# bias from anything else registered: every row gets evaluated by a SMALL
# subset of features per decision step, with mask-reuse penalty diversifying
# which features get selected across steps. HP sweep ranges follow the
# paper's recommendations for small datasets (~10-100k rows): n_d ≈ n_a,
# n_steps in 3-5, gamma slightly > 1, lambda_sparse in [1e-4, 1e-2].
'torch_tabnet': {
    'n_d':            [8, 16, 24, 32],
    'n_a':            [8, 16, 24, 32],
    'n_steps':        [3, 4, 5],
    'gamma':          (1.0, 2.0),
    'dropout':        (0.0, 0.3),
    'lambda_sparse':  (1e-5, 1e-2),
    'lr':             (5e-3, 5e-2),
    'weight_decay':   (1e-6, 1e-3),
    'batch_size':     [128, 256, 512],
    'epochs':         [40, 60, 100],
    'patience':       [8, 12, 20],
    'momentum':       (0.01, 0.1),
    'pos_weight':     [None, 1.0, 1.5, 2.0, 3.0],
    'seed':           [42, 7, 1337, 2026],
},
    'torch_mamba': {
    'd_model':       [32, 48, 64, 96],
    'n_layers':      [2, 3, 4],
    'd_state':       [8, 12, 16, 24],
    'd_conv':        [3, 4],
    'expand':        [1, 2],
    'dropout':       (0.05, 0.4),
    'learning_rate': (5e-4, 5e-3),
    'weight_decay':  (1e-6, 1e-3),
    'batch_size':    [128, 256, 512],
    'epochs':        [15, 25, 40],
    'patience':      [5, 8, 12],
    'pos_weight':    [None, 1.2, 1.5, 2.0],
},
    # models/search_spaces.py — entry to add to SEARCH_SPACES dict
'torch_patchtst': {
    'd_model':              [64, 96, 128, 192],
    'num_hidden_layers':    [2, 3, 4],
    'num_attention_heads':  [2, 4, 8],
    'patch_length':         [2, 4, 5],
    'patch_stride':         [2, 4, 5],
    'ffn_dim':              [128, 256, 512],
    'channel_attention':    [False, True],
    'use_cls_token':        [True, False],
    'pooling_type':         ['mean', 'max'],
    'share_embedding':      [True, False],
    'dropout':              (0.0, 0.4),
    'attention_dropout':    (0.0, 0.3),
    'learning_rate':        (1e-4, 5e-3),
    'weight_decay':         (1e-6, 1e-3),
    'batch_size':           [128, 256, 512],
    'epochs':               [15, 25, 40],
    'patience':             [5, 8, 12],
    'pos_weight':           [None, 1.2, 1.5, 2.0],
    'ssl_pretrain_epochs':  [0, 3, 5, 10],
    'random_mask_ratio':    [0.3, 0.4, 0.5],
},
    # models/search_spaces.py — entry to add to SEARCH_SPACES dict
'tabicl_v2': {
    'n_estimators':         [1, 2],
    'softmax_temperature':  (0.5, 1.5),
    'average_logits':       [True, False],
    'norm_methods':         ['none', 'standard', 'robust'],
    'outlier_threshold':    (2.0, 6.0),
    'batch_size':           [1, 2],
    'max_train_rows':       [8000, 12000, 15000],
    'test_chunk_size':      [1000, 2000, 4000],
    'offload_mode':         ['cpu', 'auto'],
    # use_modal=True routes fit/predict through scripts/modal_runner.py
    # (A100-40GB). Approved 2026-05-25 (ml-approval #1) under $30/mo cap.
    'use_modal':            [False, True],
},
    'torch_chronos2': {
    'seq_channel':     ['close', 'log_ret', 'zscore_close'],
    'embedding_pool':  ['mean', 'last', 'max'],
    'concat_tabular':  [True, False],
    'head_C':          (0.01, 10.0),
    'head_max_iter':   [1000, 2000],
    'batch_size':      [128, 256, 512],
},
# IsolationForest-gated HistGB. New family: anomaly-gated supervised
# classification. IF knobs control OOD detection sensitivity;
# gate_threshold / gating_alpha control how aggressively OOD rows are
# damped before the threshold sweep. HistGB knobs inherit the iter #936
# winning ranges to keep the base classifier identical to the proven
# config — we want to isolate the gating mechanism's lift.
'anomaly_gated_histgb': {
    'if_n_estimators':    [100, 200, 400],
    'if_max_samples':     (0.3, 0.9),
    'if_contamination':   (0.02, 0.20),
    'gate_threshold':     (0.05, 0.40),
    'gating_alpha':       (0.5, 3.0),
    # Iter #1675: day-level abstention quantile. 0.0 disables. Range chosen to
    # bracket bottom 5-40% of training day-anomaly quantile — the hostile-bear
    # tail (W5 breadth=31.7% sits near the bottom 15-25% of training days).
    'day_abstain_q':      (0.0, 0.40),
    'max_iter':           [200, 400, 800],
    'max_leaf_nodes':     [15, 31, 63],
    'learning_rate':      (0.01, 0.10),
    'min_samples_leaf':   [20, 50, 100],
    'l2_regularization':  (0.0, 3.0),
    'pos_class_weight':   (1.5, 4.0),
    'use_monotonic':      [True, False],
    'anomaly_as_feature': [True, False],
    'random_state':       [42, 7, 1337],
},
# DAE + LR head (iter #1530). The autoencoder bottleneck width and noise
# std are the regime-generalization knobs; head_C and pos_class_weight tune
# the LR sharpness/recall trade-off. include_recon_err toggles whether the
# OOD signal column is fed to the head (paper ablation hook).
'dae_logreg': {
    'bottleneck_dim':       [8, 12, 16, 20, 24],
    'hidden_dim':           [24, 32, 48, 64],
    'noise_std':            (0.05, 0.30),
    'dropout':              (0.10, 0.40),
    'ae_learning_rate':     (5e-4, 3e-3),
    'ae_weight_decay':      (1e-6, 1e-4),
    'ae_batch_size':        [256, 512, 1024],
    'ae_max_epochs':        [15, 25, 40],
    'ae_patience':          [3, 5, 8],
    'head_C':               (0.05, 5.0),
    'head_max_iter':        [1000, 2000],
    'include_recon_err':    [0, 1],
    'include_raw_x':        [0, 1],
    'pos_class_weight':     (1.5, 3.5),
    'abstain_strength':     (0.0, 6.0),
    'abstain_recon_q':      (0.40, 0.85),
    'random_state':         [42, 7, 1337],
},
    'torch_frets': {
    'embed_size':    [32, 64, 128],
    'hidden_size':   [64, 128, 256],
    'dropout':       (0.0, 0.4),
    'learning_rate': (1e-4, 5e-3),
    'weight_decay':  (1e-6, 1e-3),
    'pos_weight':    (1.0, 2.5),
    'epochs':        [10, 20, 40],
    'batch_size':    [128, 256, 512],
},
'kernel_anomaly_blend': {
    'gate_threshold':         (0.05, 0.40),
    'gating_alpha':           (0.5, 3.0),
    'anom_max_iter':          [200, 400, 800],
    'anom_max_leaf_nodes':    [15, 31, 63],
    'anom_learning_rate':     (0.01, 0.10),
    'anom_min_samples_leaf':  [20, 50, 100],
    'anom_l2_regularization': (0.0, 3.0),
    'anom_pos_class_weight':  (1.5, 4.0),
    'anom_use_monotonic':     [True, False],
    'kernel_n_components':    [200, 300, 500],
    'kernel_gamma':           (0.1, 2.0),
    'kernel_C':               (0.3, 10.0),
    'kernel_pca_components':  [0, 16, 32],
    'kernel_class_weight':    ['none', 'balanced'],
    'fusion_mode':            ['max', 'mean', 'rank_max', 'rank_min'],
    'random_state':           [42, 7, 1337],
},
'rocket_classifier': {
    'n_kernels':                [1000, 2000, 4000],
    'kernel_length':            [7, 9, 11],
    'max_channels_per_kernel':  [4, 6, 9, 12],
    'ridge_alpha':              (0.1, 10.0),
    'class_weight':             ['balanced', 'none'],
    'random_state':             [42, 7, 1337, 2024],
},
    'torch_xlstm': {
    'embedding_dim':  [32, 64, 128],
    'num_blocks':     [2, 3, 4],
    'num_heads':      [2, 4, 8],
    'dropout':        (0.0, 0.3),
    'learning_rate':  (1e-4, 5e-3),
    'weight_decay':   (1e-5, 1e-3),
    'pos_weight':     [1.0, 1.5, 2.0, 2.5],
    'epochs':         [15, 25, 35],
    'batch_size':     [256, 512],
},
'kernel_histgb_stack': {
    'histgb_max_iter':            [200, 400, 800],
    'histgb_max_leaf_nodes':      [31, 63, 127],
    'histgb_learning_rate':       (0.01, 0.10),
    'histgb_min_samples_leaf':    [20, 50, 100],
    'histgb_l2_regularization':   (0.0, 3.0),
    'histgb_pos_class_weight':    (1.5, 5.0),
    'histgb_use_monotonic':       [True, False],
    'kab_gate_threshold':         (0.05, 0.40),
    'kab_gating_alpha':           (0.5, 3.0),
    'kab_anom_max_iter':          [200, 400, 800],
    'kab_anom_learning_rate':     (0.02, 0.10),
    'kab_anom_pos_class_weight':  (1.5, 4.0),
    'kab_kernel_n_components':    [200, 300, 500],
    'kab_kernel_gamma':           (0.3, 2.0),
    'kab_kernel_C':               (0.5, 10.0),
    'kab_fusion_mode':            ['max', 'mean', 'rank_max', 'rank_min'],
    'stack_weight':               (0.2, 0.8),
    'fusion_mode':                ['regime_aware', 'rank_mean', 'prob_mean', 'rank_max', 'rank_min'],
    'regime_breadth_col_idx':     [16, 41],
    'regime_threshold':           (0.30, 0.55),
    'regime_bear_stack_weight':   (0.50, 0.95),
    'regime_bull_stack_weight':   (0.10, 0.55),
    'random_state':               [42, 7, 1337],
},
'rocket_bagged_calibrated': {
    'n_kernels':                [2000, 4000, 6000],
    'kernel_length':            [7, 9, 11],
    'max_channels_per_kernel':  [4, 6, 9, 12],
    'n_bags':                   [3, 5, 7],
    'bag_frac':                 (0.55, 0.85),
    'ridge_alpha':              (0.1, 10.0),
    'pos_class_weight':         (1.0, 3.0),
    'calibrate':                ['isotonic', 'none'],
    'per_date_zscore':          [True, False],
    'score_norm_mode':          ['rank', 'zscore', 'none'],
    'random_state':             [42, 7, 1337, 2024],
},
'torch_timesfm': {
    'horizon':         [1, 3, 5],
    'channel_index':   [-1, 0, 7, 8],
    'head_hidden':     [64, 128, 256],
    'dropout':         (0.0, 0.35),
    'learning_rate':   (1e-4, 5e-3),
    'weight_decay':    (1e-5, 1e-3),
    # pos_weight pinned <=1.5 (memory: iter #1722 collapsed at 2.0).
    'pos_weight':      [1.0, 1.25, 1.5],
    'epochs':          [20, 30, 50],
    'batch_size':      [256, 512, 1024],
    'inference_batch': [512, 1024, 2048],
    # seq_stats_channels — list-valued HP for per-stock dispersion injection
    # (mean/std/min/max/last-first delta per listed channel). Each candidate
    # is a fixed channel-list preset; train mode picks one.
    # Channel ref: 0=atr_pct 1=volume_ratio 7=ret_1d_xrank 8=ret_5d_xrank
    # 17=up_days_5d 18=atr_ratio_20_60 23=set_ret_5d_zscore_60d.
    'seq_stats_channels': [
        None,
        [0, 1, 7, 8, 17],
        [0, 1, 7, 8, 17, 18],
        [7, 8, 17, 18],
        [0, 1, 18, 23],
    ],
    'random_state':    [42, 7, 1337],
},
    'modernnca': {
    'd_embed':       [64, 128, 192, 256],
    'n_blocks':      [1, 2, 3, 4],
    'hidden_mult':   [1, 2, 3],
    'dropout':       (0.0, 0.4),
    'plr_freq':      [16, 24, 32, 48],
    'plr_sigma':     (0.3, 2.0),
    'plr_d_emb':     [8, 16, 24, 32],
    'lr':            (1e-4, 5e-3),
    'weight_decay':  (1e-6, 1e-3),
    'sns_rate':      (0.05, 0.6),
    'batch_size':    [512, 1024, 2048],
    'epochs':        [30, 50, 75, 100],
    'patience':      [5, 8, 12],
},
'torch_sundial': {
    'channel_index':         [-1, 7, 8, 23],
    'lookback':              [20],
    'forecast_length':       [3, 5, 10],
    'num_samples':           [8, 16, 32],
    'hidden_dim':            [64, 128, 256],
    'n_head_layers':         [1, 2, 3],
    'dropout':               (0.05, 0.30),
    'learning_rate':         (1e-4, 5e-3),
    'weight_decay':          (1e-6, 1e-3),
    'pos_weight':            (0.8, 1.5),
    'use_tabular_features':  [True, False],
    'inference_batch_size':  [128, 256, 512],
    'batch_size':            [256, 512, 1024],
    'n_epochs':              [30, 40, 60],
    'seed':                  [1234, 42, 7],
},
'label_spreading': {
    'n_neighbors':          [10, 20, 30, 50, 80],
    'alpha':                (0.05, 0.4),
    'max_iter':             [20, 30, 50],
    'subsample_size':       [3000, 5000, 7500],
    'prior_weight':         (0.2, 0.8),
    'gb_max_iter':          [150, 250, 400],
    'gb_max_leaf_nodes':    [15, 31, 63],
    'gb_learning_rate':     (0.02, 0.10),
    'gb_min_samples_leaf':  [30, 50, 80],
    'gb_l2_regularization': (0.5, 3.0),
    'pos_class_weight':     (1.0, 3.0),
    'random_state':         [42, 7, 1337],
},
    'torch_fastkan': {
    'hidden_dims':              [[64, 32], [128, 64], [128, 64, 32], [256, 128]],
    'num_grids':                [4, 8, 12, 16],
    'grid_min':                 (-3.0, -1.5),
    'grid_max':                 (1.5, 3.0),
    'learning_rate':            (3e-4, 3e-3),
    'weight_decay':             (1e-5, 1e-3),
    'n_epochs':                 [40, 60, 100],
    'batch_size':               [256, 512, 1024],
    'pos_weight':               (1.0, 2.5),
    'patience':                 [5, 8, 12],
    'dropout':                  (0.0, 0.3),
    'spline_weight_init_scale': (0.05, 0.2),
},
    'torch_ttm': {
    'ttm_prediction_length': [24, 48, 96],
    'head_hidden':         [64, 128, 256],
    'dropout':             (0.05, 0.35),
    'learning_rate':       (3e-4, 3e-3),
    'weight_decay':        (1e-5, 1e-3),
    'pos_weight':          (1.0, 1.5),
    'epochs':              [15, 25, 40],
    'batch_size':          [256, 512, 1024],
    'patience':            [3, 5, 7],
    'grad_clip':           (0.5, 2.0),
    'include_last_row':    [True, False],
},
    'torch_toto2': {
    'model_id':            ['Datadog/Toto-2.0-4m', 'Datadog/Toto-2.0-22m'],
    'context_length':      [256, 512],
    'horizon':             [32, 64, 96],
    'decode_block_size':   [64, 128, 256],
    'head_hidden':         [64, 128, 256],
    'dropout':             (0.05, 0.35),
    'learning_rate':       (3e-4, 3e-3),
    'weight_decay':        (1e-5, 1e-3),
    'pos_weight':          (1.0, 1.5),
    'epochs':              [15, 25, 40],
    'batch_size':          [128, 256, 512],
    'inference_batch':     [32, 64, 128],
    'patience':            [3, 5, 7],
    'grad_clip':           (0.5, 2.0),
    'include_last_row':    [True, False],
    'has_missing_values':  [False],
},
    'torch_mantis': {
    'model_id':            ['paris-noah/Mantis-8M'],
    'model_class':         ['MantisV1'],
    'target_length':       [256, 512],
    'multivariate_mode':   ['per_channel'],
    'head_hidden':         [64, 128, 256],
    'dropout':             (0.05, 0.35),
    'learning_rate':       (3e-4, 3e-3),
    'weight_decay':        (1e-5, 1e-3),
    'pos_weight':          (1.0, 1.5),
    'epochs':              [15, 25, 40],
    'batch_size':          [128, 256, 512],
    'inference_batch':     [64, 128, 256],
    'patience':            [3, 5, 7],
    'grad_clip':           (0.5, 2.0),
    'include_last_row':    [True, False],
},
}


def _archived_trainers() -> set[str]:
    """Names archived by Phase 3C (2026-05-27). Returned as a set so we can
    filter the search-space surface without deleting their entries from
    SEARCH_SPACES (the data stays in source for easy restoration).
    Imported lazily to avoid a circular dep with models.trainers_archive."""
    try:
        from models.trainers_archive import TRAINERS_ARCHIVED
        return set(TRAINERS_ARCHIVED.keys())
    except Exception:
        return set()


def sample(trainer: str, rng: np.random.RandomState | None = None) -> dict:
    """Draw one configuration for a given trainer."""
    if trainer in _archived_trainers():
        raise ValueError(
            f'Trainer {trainer!r} is archived (Phase 3C). '
            f'See models/trainers_archive.py for resurrection criteria.')
    if trainer not in SEARCH_SPACES:
        raise ValueError(f'No search space defined for {trainer!r}. '
                         f'Add one in models/search_spaces.py.')
    if rng is None:
        rng = np.random
    space = SEARCH_SPACES[trainer]
    cfg = {}
    for key, val in space.items():
        if isinstance(val, list):
            # Index-select instead of rng.choice(val): when the options are
            # themselves sequences (e.g. hidden_dims=[[64,32],[128,64,32]]),
            # rng.choice tries to build an ndarray and dies with
            # "inhomogeneous shape" on ragged sublists. Indexing the Python
            # list sidesteps numpy entirely and returns the native object.
            choice = val[int(rng.randint(len(val)))]
            if isinstance(choice, (list, tuple)):
                cfg[key] = list(choice)
            else:
                cfg[key] = type(val[0])(choice)
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
    """Active trainer list — excludes Phase 3C archived names."""
    archived = _archived_trainers()
    return sorted(k for k in SEARCH_SPACES.keys() if k not in archived)
