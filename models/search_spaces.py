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
