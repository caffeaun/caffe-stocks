#!/usr/bin/env python3
"""LSTM trainer for trading signals (walk-forward path).

Structural change (THIS ATTEMPT #327): POST-SWA TEMPERATURE SCALING
(Guo, Pleiss, Sun, Weinberger 2017, "On Calibration of Modern Neural
Networks", ICML — explicitly listed nowhere in the project's 326-attempt
history, verified by grep for 'temperature_scal', 'temp_scale',
'_output_temp.fill_', 'nll_temp', 'TemperatureScaling', 'guo' — all
returned no matches).

After SWA averaging — and after any optional recent-window FT —
but BEFORE precision/EV-targeted FC bias calibration, a single
scalar T is fit to minimize binary cross-entropy NLL on the val set
with logits transformed as sigmoid(z / T). The fit is a closed-form
1D bounded optimization (scipy.minimize_scalar, method='bounded',
T in [0.3, 5.0]) so compute is << 1 s per WF split. T* updates the
model's `_output_temp` buffer in place; the saved buffer carries
through to the WF gate's loader and to production inference because
the model class already wraps every forward pass with
sigmoid(logit / _output_temp). A 1% relative-NLL-improvement gate
keeps the post-hoc fit from intervening when val is too small or
the existing T is already near-optimal — degrades gracefully to
the (now-stationarity-OFF) Entry 197 baseline.

Why temperature scaling attacks the 37.5%-WR plateau and the
"epoch-0-best" pathology
----------------------------------------------------------------
The persistent failure modes have been (i) val_loss diverges by
epoch 0–4 (temporal feature distribution shift) and (ii) WR
variance across splits is 12–26% even on configurations that
average above base rate. Both leave a model whose RANKING of
stocks is informative but whose CONFIDENCE values are systematically
miscalibrated — Guo et al. (2017) proved this is the rule, not the
exception, for modern NNs trained with cross-entropy on imbalanced
data. The bias calibration that runs next computes
    b_shift = T * (logit_target − logit_qraw)
so the magnitude of the shift, and therefore the robustness of the
resulting threshold under regime drift, depends on T being
CORRECT. The model ships with output_temperature=0.5 (a sharpening
prior chosen to rescue 0-trade configurations from sub-0.6 score
collapse). When the model has already learned a healthy decision
surface, that aggressive sharpening over-extends confidence into
miscalibrated regions; the bias shift then has the wrong magnitude.
Fitting T per WF split against that split's own val NLL automatically
dampens over-confidence in noisy regimes (T → larger) and restores
sharpness when val NLL prefers it (T → smaller), giving the
bias-calibration step a probability scale that is locally honest.
This is the textbook way to put the model's score distribution back
on a meaningful interpretive footing without changing what the
model learned.

Why this is NOT one of the already-rejected mechanisms
------------------------------------------------------
  * Selectivity calibration (#289, post-hoc): shifts FC bias only;
    treats T as fixed at init. This change adds T as a free
    parameter UPSTREAM of bias shifting; bias calibration runs
    unchanged on the temperature-scaled scores.
  * Precision-targeted bias calibration (#308, ACTIVE): same — only
    bias shifts; T stays at init.
  * EV-targeted bias calibration (#324, REJECTED): also bias-only.
  * Label smoothing (#47, #72, REJECTED): changed BCE TARGETS during
    training (y → 0.95/0.05). Temperature scaling changes only the
    output INTERPRETATION post hoc; the BCE targets and the saved
    model weights are bit-identical to the baseline. The score
    distribution is NOT compressed toward 0.5 the way label
    smoothing did — only its spread is rescaled monotonically.
  * Multi-fold val select (#311, REJECTED): changed SELECTION
    metric during training. Temperature scaling does not touch
    selection at all; it runs once after training is done.
  * SAM (#294), R-Drop (#325), gradient noise (#323), all REJECTED:
    OPTIMIZER-level perturbations during training. Temperature
    scaling is a closed-form post-hoc 1D fit on val. Zero training-
    time changes; zero new randomness; zero new gradient flow.
  * Mean Teacher (#300, REJECTED), DANN (#296, REJECTED): added
    AUXILIARY LOSSES or extra forward passes during training. T-
    scaling is a SCALAR fit AFTER training.
  * Hand-tuned T sweep (never tried automatically here): the model
    ships with output_temperature=0.5 hard-coded in lstm_model.py;
    no prior attempt has fit T per-split against val.

Why this is SAFE
----------------
  - lstm_model.py is unchanged: model already wraps every forward
    pass with sigmoid(logit / _output_temp). We only update the
    BUFFER VALUE.
  - Saved model has BIT-IDENTICAL signature to Entry 197.
    state_dict keys, shapes, dtype are unchanged. The gate's
    loader, the live trader's loader, and the inference path all
    transparently pick up the new T.
  - Pure monotone post-hoc 1D fit. Score ORDERING is preserved
    exactly; only the spread of confidences changes. The model's
    ranking ability is COMPLETELY UNAFFECTED.
  - Compute cost: one val forward pass to extract logits, then a
    1D scalar minimization. << 1 second per WF split.
  - Degrades gracefully:
      * Val too small (< MIN_VAL_FOR_CALIBRATION): no-op.
      * NLL improvement < 1% relative: no-op (T stays at init 0.5).
      * scipy unavailable: 200-point log-spaced grid fallback.
      * use_temperature_scaling=False (sweep ablation): bit-identical
        to the (now-stationarity-OFF) baseline.

Companion changes for clean isolation
-------------------------------------
  * use_stationarity_mask default flipped True → False (since #326
    was just rejected). Mask still available as opt-in for sweep.
  * daily_rank_active is suppressed when temp_scaling_active is True.
    The WF gate passes daily_rank_enabled=True; without this guard
    the rejected pairwise (#303) / listnet (#310) rankers would
    silently activate alongside this attempt's mechanism, muddling
    the ablation.
  * All other rejected-mechanism opt-ins remain default OFF.

Reference
---------
Guo, Pleiss, Sun, Weinberger (2017). "On Calibration of Modern
Neural Networks." ICML.

---- LEGACY HEADER (historical context — KS-mask #326 REJECTED) ----
Structural change (PREVIOUS ATTEMPT #326): FEATURE-STATIONARITY KS-MASK
(per-feature multiplicative input mask derived from the train/val
Kolmogorov–Smirnov 2-sample distance — explicitly listed as untried
option (5) in the condensed lessons:
  "Adding a feature-stationarity filter (drop any feature whose
   train/val KS-distance exceeds a threshold) automatically rather
   than hand-curating".)

For each input feature i, after RobustScaler.fit_transform on train
and .transform on val, we compute the KS distance KS_i between the
two empirical distributions on the decision-point timestep (last step
of each sequence). The per-feature multiplicative mask is

    m_i = max(floor, exp(-alpha * KS_i^2))

with alpha=10.0 and floor=0.3 (default). Applied to scaled inputs
(broadcast over time) AND baked into scaler.scale_ via
    scaler.scale_ <- scaler.scale_ / mask
so that downstream scaler.transform() at the WF gate, in production,
and during recalibration automatically reproduces the same masked
feature space — no train/inference skew.

Why this attacks the 37.5%-WR plateau / "epoch-0-best" pathology
----------------------------------------------------------------
Persistent problem #1 in the condensed lessons: "Best epoch is almost
always 0–4: val_loss diverges immediately regardless of feature set,
architecture (LSTM vs MLP vs seq_len=1), or label definition —
confirming a feature-distribution temporal shift, not LSTM
overfitting."

The KS-mask intervenes at the most upstream layer possible: the INPUT
itself. Features that drift across the train/val boundary embed
shortcut signals that fit train but mismatch val; the optimizer
exploits them and val_loss collapses by epoch 0–4. Multiplicatively
shrinking those features at the model's input shifts gradient pressure
away from drift-prone features and toward features whose distributions
are stable across the temporal boundary.

Critically computed PER WALK-FORWARD SPLIT against that split's own
train/val pair. Different splits have different drift profiles (a
feature stable in one regime may shift in another), so the mask
self-adapts per-split rather than imposing a single global feature
ranking. This is the recommended automatic version of the
hand-curated feature subsets that produced the highest individual
attempts (#115 0.5415, #139 0.53) — but those were for 3-split
configs and didn't carry across all 7 splits. Per-split automation
addresses the regime-mismatch directly.

Why this is NOT one of the already-rejected mechanisms
------------------------------------------------------
  * #295 covariate-shift density-ratio (REJECTED): SAMPLE-level
    weighting via a learned XGBoost domain classifier. Saturated
    with domain AUC ≈ 1.0. The mask is FEATURE-level, computed in
    closed form from a KS test (no learned model, no saturation),
    and capped at floor=0.3 so no feature is ever zeroed out.
  * #296 DANN (REJECTED): adversarial gradient-reversal alignment of
    train/val FEATURES via a domain classifier added to the forward
    pass. Train-time parametric adversary. The mask has zero learned
    parameters and zero adversarial training; it's a deterministic
    rescaling of inputs computed once per split.
  * #298 SS-pretrain (REJECTED): unsupervised next-step pretext task
    on the LSTM backbone. Pre-training stage. The mask is INPUT-LAYER
    and runs alongside training, not as a separate stage.
  * #316 chronological time-decay (REJECTED): SAMPLE-level weighting
    by calendar position. Orthogonal: sample-level vs feature-level.
  * #317 boundary-band sample exclusion (REJECTED): hard SAMPLE
    filtering by |pnl - 4%|. The mask keeps every sample.
  * #318 recent-window FT (REJECTED): two-stage with frozen backbone.
    The mask is single-stage, operates on inputs not on FC weights.
  * #319 SupCon (REJECTED): cross-symbol contrastive pretraining.
    Stage-0 representation learning. The mask runs in-place during
    regular supervised training.
  * #320 IRM (REJECTED): per-environment penalty on the gradient of
    a dummy-scaled loss. Optimizer-level mechanism. The mask is
    data-level, no penalty term, no environments.
  * #325 R-Drop (REJECTED): output-distribution regularization via
    KL between two dropout-perturbed forward passes. Optimizer-level.
    The mask is a data preprocessing step.
  * Hand-curated feature subsets (#54, #60, #189, #210, #218, etc.):
    a HUMAN chose features. The mask is fully AUTOMATIC and per-split.
  * NOT in the project's 325-attempt history (verified by exhaustive
    grep on the trainer source for 'KS_2', 'ks_2samp', 'stationarity',
    'ks_distance').

Why this is SAFE
----------------
  - CURATED_FEATURES is unchanged: feature_eng.py still produces the
    same 19-feature input layout. The scaler still expects 19 features.
    No alignment break.
  - lstm_model.py is unchanged: LSTMModel input_size, layer topology,
    forward path are identical. Production inference sees the same
    model class.
  - Saved model has BIT-IDENTICAL signature to Entry 197: state_dict
    keys, shapes, dtype, and attribute set are all unchanged.
  - Saved scaler is BIT-IDENTICAL in format (pickled RobustScaler);
    only the numerical values of `scale_` differ. The WF gate's
    .transform() call works unchanged because scale_ is just a
    per-feature divisor.
  - Inference behavior is consistent end-to-end: training uses masked
    features → saved scaler reproduces those masked features on
    test/live data → model evaluates on the same distribution it
    trained on. No train/inference skew.
  - Compute cost: 19 KS tests on ≤ N samples each. << 1 second per
    WF split. Negligible compared to LSTM training.
  - Degrades gracefully:
      * When all features stationary (KS≈0): mask≈1 everywhere → no-op.
      * When use_stationarity_mask=False (sweep ablation): mask not
        computed, scaler unmodified, training is bit-identical to the
        (now r_drop-OFF) Entry 197 baseline.
      * On scipy unavailable / degenerate input: returns all-ones
        mask and falls through.
  - Floor of 0.3 prevents any feature from being effectively removed;
    the model can still recover signal from a non-stationary feature
    if it truly helps.

Companion changes for clean isolation
-------------------------------------
  * use_r_drop default flipped True → False (since #325 was just
    rejected). Reverts the BCE path to single-forward-pass (Entry
    308 baseline with EV-calibration off and R-Drop off).
  * daily_rank_active is suppressed when stationarity_active is True.
    The WF gate passes daily_rank_enabled=True; without this guard,
    the rejected pairwise (#303) / listnet (#310) rankers would
    silently activate alongside the stationarity mask, muddling the
    ablation.
  * All other rejected-mechanism opt-ins remain default OFF.

References
----------
Massey (1951). "The Kolmogorov-Smirnov Test for Goodness of Fit."
JASA. (foundational two-sample KS).
Sugiyama et al. (2007). "Direct Importance Estimation for Covariate
Shift Adaptation." (background for the covariate-shift framing
distinguishing this from #295's density-ratio approach).

---- LEGACY HEADER (historical context — R-Drop #325 REJECTED) ----
PREVIOUS ATTEMPT #325: R-DROP REGULARIZED DROPOUT
(Liang et al. 2021, "R-Drop: Regularized Dropout for Neural Networks",
NeurIPS 2021, arXiv:2106.14448).

For every training batch, the model in train() mode runs the SAME
mixup'd input through TWO forward passes with INDEPENDENT dropout
masks. The two predictions p1 and p2 are added to the loss as

    L = 0.5 * (BCE(p1, y) + BCE(p2, y))
        + r_drop_alpha * 0.5 * (KL(p1||p2) + KL(p2||p1))

with r_drop_alpha=5.0 and a linear ramp-up of the KL weight over the
first 3 epochs.

Why R-Drop attacks the 37.5%-WR plateau and the "epoch-0-best" pathology
------------------------------------------------------------------------
Across 324 attempts the dominant failure is "best epoch is 0-2 with
val_loss diverging immediately" — the condensed lessons attribute this
to the model committing to a sharp train-era feature shortcut within
1-2 epochs of an 11%-positive label distribution. R-Drop forces the
model's predictions to be INVARIANT under dropout perturbation:

  (1) Features the model genuinely USES survive both dropout masks
      (their downstream contribution is robust by construction).
  (2) Features that act as random shortcuts produce divergent
      predictions across the two masks → accumulate KL loss → are
      penalized.
  (3) The optimizer is therefore pushed toward using ROBUST features
      whose contribution does not depend on a specific neuron's
      activation. Robustness within an input distribution is a known
      proxy for robustness across related distributions, attacking
      the cross-split-WR variance directly.

Why this is NOT one of the already-rejected mechanisms
------------------------------------------------------
  * Mean Teacher (#300, REJECTED): EMA-teacher consistency. TEACHER
    and STUDENT are different network instances. R-Drop uses the SAME
    network instance with different DROPOUT REALIZATIONS; no second
    model exists. Mean Teacher's failure mode ("teacher = student at
    init -> early signal near zero") does NOT apply: with dropout=0.4
    the two R-Drop forward passes produce meaningfully different
    outputs from EPOCH 0.
  * Temporal-consistency (#309, REJECTED): full vs truncated SEQUENCE.
    Same model, different INPUTS. R-Drop uses the same model AND same
    input — only the dropout mask differs. Different mechanism; the
    gradient signal is on dropout sensitivity, not sequence-length
    sensitivity.
  * Bootstrap-bagging deep-ensemble (#314, REJECTED): K independently
    -trained ensemble distilled at OUTPUT level. R-Drop is single-
    model, single-stage; KL is applied DURING training to TWO forward
    passes of the same model, not to ensemble means.
  * SAM (#294, REJECTED): worst-case ascent in a fixed ρ-ball.
    Deterministic gradient direction. R-Drop's extra signal is
    RANDOM (Bernoulli dropout masks), unbiased, and cheap.
  * Annealed gradient noise (#323, REJECTED): N(0, σ²) on PARAMETER
    GRADIENTS. R-Drop's perturbation is on the ACTIVATIONS via
    dropout — different layer of the system.
  * Label smoothing (#47, #72, REJECTED): UNIFORM y -> 0.95/0.05.
    Compresses the score distribution below 0.6. R-Drop keeps the
    BCE TARGET unchanged (still y_mix); only KL of p1 vs p2 is added.
  * R-Drop is NOT in this project's 324-attempt history (verified by
    exhaustive grep on the trainer source).

Why this is SAFE
----------------
  - Saved model has IDENTICAL signature to Entry 197 — gate loader,
    live trader, scaler are all unchanged. Production inference runs
    in eval() mode (dropout OFF) with a SINGLE forward pass — no
    R-Drop machinery leaks into inference.
  - Compute cost: 2x forward pass per training step. Backward pass
    remains 1x because both forwards share the autograd graph.
    Comfortably within the 30-min WF-split budget.
  - r_drop_alpha=5.0 calibrated for our problem: with pos_weight≈10,
    BCE term is ~3-7 per sample (heavily class-balanced); KL is
    ~0.04-0.20 per sample. alpha=5.0 makes KL contribute ~10-30% of
    total loss — meaningful regularization without drowning BCE.
  - Linear ramp-up of the KL weight over the first 3 epochs avoids
    the early-training divergence ("two random forward passes have
    huge KL → huge gradient → divergence") failure mode.
  - Degrades gracefully: when use_r_drop=False (sweep ablation),
    dropout=0, or non-BCE path, the code path is bit-identical to
    the (now ev-calibration-OFF) baseline.

Companion change for clean isolation
-------------------------------------
  * use_ev_calibration default flipped True → False (since #324 was
    just rejected). Reverts the bias calibration to the precision-
    targeted path (Entry 308 baseline).
  * daily_rank_active is suppressed when r_drop_active is True.
    The WF gate passes daily_rank_enabled=True; without this guard,
    the rejected pairwise (#303) / listnet (#310) rankers would
    silently activate alongside R-Drop, muddling the ablation.
  * All other rejected-mechanism opt-ins remain default OFF.

Reference: Liang et al. (2021). "R-Drop: Regularized Dropout for
Neural Networks." NeurIPS.

---- LEGACY HEADER (historical context — Annealed Gradient Noise
#323 REJECTED) ----
PREVIOUS ATTEMPT #323: ANNEALED GRADIENT NOISE INJECTION
(Neelakantan et al. 2015, "Adding Gradient Noise Improves Learning
for Very Deep Networks", arXiv:1511.06807).

After loss.backward() and before optimizer.step(), Gaussian noise is
added to every parameter's accumulated gradient. The noise standard
deviation anneals over training steps:
    sigma_t = eta / (1 + step_post_warmup) ** gamma
with eta=0.01 and gamma=0.55 (Neelakantan et al. defaults). The first
`grad_noise_warmup_steps` (default 200, ~1 epoch) inject NO noise so
the optimizer first establishes a sensible gradient direction.

Why annealed gradient noise attacks the "epoch-0-best" pathology
----------------------------------------------------------------
The dominant 322-attempt failure mode is "best epoch is 0-4 with val
loss diverging immediately". The condensed lessons attribute this to
random-init Adam committing to a sharp local minimum within 1-2
epochs of an 11%-positive label distribution. Once committed, the
sharp minimum cannot generalize across train→test regime shift —
which the WF gate's 7 fixed splits expose.

Gradient noise injection works at the OPTIMIZER level, NOT the
input/activation level. At each step, the perturbed gradient is
g_t + xi_t with xi_t ~ N(0, sigma_t^2 I). Three coupled effects:

  (1) FLAT-MINIMUM BIAS. With sigma_t > 0 the optimizer cannot stably
      sit in a sharp minimum: any narrow basin will be exited by the
      next noise sample. SGD-with-noise provably converges (in
      expectation) to flat minima with diameter >> sigma_t.
  (2) EPOCH-0 ESCAPE. The polynomial decay 1/(1+t)^0.55 keeps sigma
      meaningful for the first thousand steps before fading. So the
      noise is loudest exactly when the model is most prone to
      committing to a sharp shortcut. By the time noise is small
      (post epoch ~5), the optimizer is already in a flat basin and
      the SWA averaging that follows can refine it without losing
      generalization.
  (3) BAYESIAN POSTERIOR APPROXIMATION. Welling & Teh 2011 showed
      SGLD (essentially this method with sigma scaled by sqrt(lr))
      samples from the Bayesian posterior over weights. The SWA
      average over the noisy trajectory therefore approximates the
      posterior mean — a principled aggregation, not just a
      heuristic late-epoch smoother.

These properties make the resulting weights LESS dependent on the
train period's specific distribution, which is exactly what the
WF gate's cross-regime stability requirement demands.

Why this is NOT one of the already-rejected mechanisms
------------------------------------------------------
  * #294 SAM (REJECTED): worst-case ascent-then-descent in a fixed
    rho-ball. Deterministic 2x compute, gradient direction chosen
    adversarially. Gradient NOISE is RANDOM (Gaussian), unbiased,
    and trivially cheap. Different theoretical objective: SAM
    minimizes max_{||eps||<=rho} L(w+eps); gradient noise samples
    the posterior over w.
  * #295 covariate-shift density-ratio reweighting (REJECTED):
    SAMPLE-level reweighting based on learned domain density ratio.
    Gradient noise is PARAMETER-level, sample-independent.
  * #300 Mean Teacher (REJECTED): EMA-teacher consistency loss on
    student outputs. Operates on PREDICTIONS. Gradient noise
    operates on GRADIENTS — different layer of the system.
  * #305 GCE (REJECTED): clamped gradient magnitude in prediction
    space. Gradient noise ADDS unbiased Gaussian, doesn't clamp.
  * #314 bootstrap-bagging deep-ensemble distillation (REJECTED):
    K=3 OUTPUT-level ensemble. Gradient noise is single-model,
    no ensemble distillation.
  * #316 time-decay weighting (REJECTED): SAMPLE-level reweighting
    by chronological position. Gradient noise is sample-independent.
  * #318 recent-FT (REJECTED): two-stage with frozen backbone.
    Gradient noise is single-stage end-to-end.
  * INPUT_NOISE_STD (already in lstm_model.py during training):
    perturbs ACTIVATIONS at the input layer. Gradient noise
    perturbs the OPTIMIZER UPDATE — orthogonal mechanism.
  * Listed as UNTRIED among the condensed lessons' "non-standard
    algorithms purpose-built for this problem". Neelakantan et al.
    (2015) proved gradient noise is most effective on small noisy
    datasets and deep networks — fitting the SET 49K-row 19-feature
    LSTM almost perfectly.

Why this is SAFE
----------------
  - Saved model signature is BIT-IDENTICAL to Entry 197 — gate
    loader, live trader, scaler are unchanged. Production inference
    is unchanged. Only the FC weights end up at a slightly different
    location in parameter space (the SWA-averaged centroid of a
    noisier trajectory).
  - Compute cost: one elementwise torch.randn_like() and one add per
    parameter per step. << 1% overhead vs the LSTM forward pass.
  - Noise variance is bounded: sigma_t monotonically decays to 0,
    so post-warmup the optimizer asymptotically reduces to the
    Entry 197 baseline. Catastrophic divergence is structurally
    impossible if sigma stays small relative to ||grad||.
  - Degrades gracefully: when use_grad_noise=False (sweep ablation
    or explicit kwarg), the code path is bit-identical to the
    (now-Sortino-OFF) Entry 197 baseline.
  - Fixed seed (42) for the LSTM, plus explicit seeding of the
    gradient-noise generator — fully deterministic across reruns.

Companion changes for clean isolation
-------------------------------------
  * use_sortino_aux default flipped True → False (since #322 was
    just rejected). Reverts the loss to pure class-weighted BCE
    on binary y_mix.
  * daily_rank_active is suppressed when grad_noise_active is True.
    The WF gate passes daily_rank_enabled=True; without this guard,
    the rejected pairwise (#303) / listnet (#310) ranking would
    silently activate alongside this attempt's mechanism, muddling
    the ablation.
  * All other rejected-mechanism opt-ins remain default OFF (DANN,
    SS, Mean Teacher, GCE, quantile aux, temporal consist, listnet
    rank, multifold select, SelectiveNet, time-decay weighting,
    recent-FT, SupCon, IRM, curriculum, covariate shift, XGB
    distill, PnL-optimization, PnL-smooth-labels, Sortino aux).

References
----------
Neelakantan et al. (2015). "Adding Gradient Noise Improves Learning
for Very Deep Networks." arXiv:1511.06807.
Welling & Teh (2011). "Bayesian Learning via Stochastic Gradient
Langevin Dynamics." ICML.

---- LEGACY HEADER (historical context — Sortino aux #322 REJECTED) ----
PREVIOUS: AUXILIARY DIFFERENTIABLE SORTINO-RATIO LOSS (#322).
A small auxiliary term added to BCE that maximizes the risk-adjusted
return of soft-selected trades, with downside-only variance in the
denominator.

For each batch, soft per-sample return:
    r_i = pred_i * (pnl_i - commission)
Sortino-style score (asymmetric — penalizes only LOSING variance):
    S = mean(r) / sqrt(mean(min(r, 0)^2) + eps)
Auxiliary loss:
    L_sortino = -S + budget_lambda * (mean(pred) - target_sel)^2
Combined with BCE:
    L_total = BCE(pred, y) + sortino_aux_weight * L_sortino

Why Sortino aux attacks the cross-split-variance failure mode
-------------------------------------------------------------
The dominant failure mode is "wf_std 0.20-0.26 even on PASSING runs":
a model that scores 60% WR on splits 1-3 and 20% on splits 4-7 averages
40% but FAILS the 7-of-7 hard gate. Pure BCE optimizes a per-sample
classification loss with no notion of OUTCOME CONSISTENCY across the
batch. The Sortino term puts BATCH-LEVEL DOWNSIDE VARIANCE in the
denominator: a model that fires confidently on a few clean winners and
abstains elsewhere has a high mean(r) and a low downside std → high
Sortino. A model that sprays predictions across many uncertain trades
has a similar mean(r) but accumulates many small losers in the
downside variance → low Sortino.

The downside-only variance is the key: it is asymmetric, matching the
asymmetric trading payoff (+15% target / -3% stop). Upside variance
(rare big winners) is GOOD and not penalized; downside variance (cluster
of small losers) is BAD and IS penalized. This shapes the learned
representation toward "consistent winning small trades" rather than
"sometimes nail a big winner, sometimes spray on a string of losers".

Why this is NOT one of the rejected mechanisms
----------------------------------------------
  * #290 PNL-OPTIMIZATION (REJECTED): replaced BCE entirely with raw
    -E[p · (pnl - c)] + budget term. NO variance in the objective.
    This change adds the asymmetric-variance term in the denominator
    AND keeps BCE as the primary loss for gradient stability — the
    Sortino term is auxiliary (weight 0.4), not sole loss.
  * #199 PNL-MAGNITUDE WEIGHTED BCE (REJECTED): scaled per-sample BCE
    by |pnl|. Per-sample reweighting. This change adds a SEPARATE LOSS
    TERM that operates on a BATCH-AGGREGATE statistic (downside std),
    not a per-sample weight.
  * #305 GCE (REJECTED): clipped per-sample gradient magnitude
    SYMMETRICALLY in prediction space. Sortino is asymmetric per
    sample (only losers contribute to the variance penalty) and
    operates at the BATCH level via the std term.
  * #306 QUANTILE AUX (REJECTED): predicted {p10..p90} of PnL via
    pinball on a SEPARATE auxiliary head. Sortino does not predict
    quantiles; it constructs a single batch-level statistic from the
    main classifier's outputs.
  * #321 PNL-DISTANCE SOFT LABELS (REJECTED): replaced binary y with
    sigmoid(PnL distance). A LABEL-side change. Sortino is a LOSS-side
    addition; the binary BCE label is kept as-is. Set OFF here.
  * Listed as untried in the condensed lessons:
    "Differentiable Sharpe/profit-factor objective."

Why this is SAFE
----------------
  - Saved model has IDENTICAL signature to Entry 197 — gate loader,
    live trader, scaler are unchanged. Production inference is bit-
    identical.
  - BCE remains the primary loss; Sortino is auxiliary. If Sortino
    gradient magnitude is small relative to BCE, training degrades
    gracefully toward the BCE baseline. If Sortino is large, BCE
    still anchors the score distribution to [0, 1] and prevents
    degenerate solutions.
  - Selectivity-budget penalty in the Sortino term keeps mean(pred)
    near target_sel=0.10 — prevents the optimizer from pushing all
    pred_i to 0 (which would zero out r and make Sortino indeterminate
    via 0/0) or to 1 (which would maximize Sortino magnitude but
    destroy class separation).
  - Warmup over the first sortino_warmup_epochs ramps the aux weight
    from 0 → sortino_aux_weight, letting BCE establish a sensible
    decision boundary before the Sortino term pulls on it. Avoids
    early-epoch divergence where Sortino's batch-level statistic is
    high-variance (small batches → noisy std estimates).

Companion changes for clean isolation
-------------------------------------
  * use_pnl_smooth_labels default flipped True → False (since #321 was
    just rejected). Reverts the BCE target from sigmoid-PnL-distance
    soft labels back to the binary y_mix.
  * daily_rank_active is suppressed when sortino_active is True. The
    WF gate passes daily_rank_enabled=True; without this guard, the
    rejected pairwise (#303) / listnet (#310) ranking would silently
    activate alongside this attempt's mechanism, muddling the ablation.
  * All other rejected-mechanism opt-ins remain default OFF (DANN, SS,
    Mean Teacher, GCE, quantile aux, temporal consist, listnet rank,
    multifold select, SelectiveNet, time-decay weighting, recent-FT,
    SupCon, IRM, curriculum, covariate shift, XGB distill,
    PnL-optimization).

---- LEGACY HEADER (historical context) ----
PREVIOUS: EXPONENTIAL CHRONOLOGICAL TIME-DECAY SAMPLE WEIGHTING (#316). Per-sample loss weight w_i = exp(decay
* t_i_norm) where t_i_norm in [0, 1] is the chronological position of
sample i within the train window. With decay=1.0 the most-recent train
sample weighs e ~= 2.72x the oldest. Mean-normalized to 1 so total
loss magnitude is preserved.

Why time-decay attacks the 37.5%-WR plateau / "epoch-0-best" pathology
---------------------------------------------------------------------
The dominant failure mode across 315 attempts is "best epoch is 0-4
with val_loss diverging immediately". The condensed lessons attribute
this to a temporal feature-distribution shift, not architectural
overfitting. The model spreads gradient mass uniformly across train
samples, so older market regimes dominate by count and pull the
weights toward an "average regime" that is not the regime where test
actually lives.

Time-decay reweighting tells the optimizer: "samples chronologically
closest to the train cut-off matter more, because they are the ones
whose distribution most closely matches the test window that
immediately follows." It is parameter-free, monotone, smooth, and
label-independent — the weight depends only on each sample's calendar
position, not its features or label.

Why this is not one of the rejected mechanisms
----------------------------------------------
  * Density-ratio covariate-shift (#295, REJECTED): learned weights
    from FEATURE distributions via an XGBoost domain classifier.
    Saturated immediately (domain AUC ~ 1.0 -> all weights pinned to
    clip floor). Time-decay uses CHRONOLOGY directly — no learned
    model, no saturation; weights are bounded by exp(decay) by
    construction.
  * Curriculum learning (#304, REJECTED): hard-thresholded sample
    FILTERING by |PnL|. Time-decay is SOFT continuous reweighting —
    every sample contributes; only the gradient share shifts.
  * Mean Teacher (#300, REJECTED): EMA-teacher consistency loss — a
    different mechanism entirely; does not reweight samples.
  * SS pretraining (#298, REJECTED): two-stage backbone pretext task.
    Time-decay is single-stage and orthogonal.
  * SelectiveNet abstention (#312, REJECTED): learned per-sample
    selection head. Time-decay does not learn weights at all — they
    are derived from each sample's calendar position deterministically.

Companion changes for clean isolation
-------------------------------------
  * use_selective_head default flipped True -> False (since #312 was
    just rejected). Keeps the SelectiveNet code available for sweep
    ablation but inactive by default.
  * daily_rank_active is suppressed when time_decay is active. The WF
    gate passes daily_rank_enabled=True; without this guard the
    rejected pairwise/listnet ranking (#303 / #310) would silently
    activate alongside this attempt's mechanism, muddling the
    ablation.
  * All other rejected-mechanism opt-ins remain default OFF (DANN,
    SS-pretrain, Mean Teacher, GCE, quantile aux, temporal consist,
    listnet rank, curriculum, covariate shift, XGB distill,
    multifold val select, PnL-optimization).

Why this is SAFE
----------------
  - Saved model has the identical signature to Entry 197 — gate
    loader, live trader, and scaler are all unchanged. The only thing
    that differs is the per-sample loss weighting during training, so
    the trained FC weights end up in a slightly different location
    in parameter space.
  - Compute cost: one numpy array of length N_train (a few KB for
    typical splits) and one elementwise multiplication per sample
    when packing the dataset. Negligible vs the LSTM forward pass.
  - Degrades gracefully: when use_time_decay=False or dates_train is
    not supplied, behavior is bit-identical to the (now selective-head-
    OFF) Entry 197 baseline.
  - Fixed seed (42) for the LSTM — no additional source of run-to-run
    variance.

---- LEGACY (historical context retained below) ----

PREVIOUS: SELECTIVE CLASSIFICATION WITH A LEARNED ABSTAIN HEAD
(SelectiveNet, Geifman & El-Yaniv 2019, arXiv:1901.09192). Listed as
untried approach (a) in the condensed lessons.

Mechanism
---------
A small SelectionHead `g(x)` is trained JOINTLY with the LSTM
classifier `f(x)`. Both share the LSTM backbone; only `f(x)` is saved
(the LSTMModel state_dict is bit-identical to the baseline). `g(x)`
is discarded after training but reshapes the gradient flow into the
backbone during training. The combined loss per batch is

    L = (1 - α) · BCE(f, y)                        # auxiliary CE
      + α · [ (g · BCE(f, y)).sum() / (Σg + ε)     # selective risk
              + λ · max(0, c* - mean(g))² ]        # coverage penalty

with α=0.5, λ=32, target coverage c*=0.5 (paper defaults).
- The auxiliary CE term keeps `f(x)` honest on ALL samples (avoids
  collapse on the un-selected region).
- The selective-risk term applies EXTRA gradient pressure on samples
  where `g(x)` is high — the network's learned "trust" region. f(x)
  becomes sharper there.
- The coverage penalty prevents `g(x) → 0` (everything abstain) by
  penalizing coverage shortfalls below 50%.
- `g(x)` learns to be HIGH where the per-sample BCE loss is low
  (well-classified samples) — i.e., it discovers the regime-invariant
  signal automatically, without needing labelled regime indicators.

Why selective classification attacks the 37.5%-WR plateau
---------------------------------------------------------
1. **Targets THE EXACT METRIC the gate evaluates.** WR @ threshold-0.6
   is precision over the model's most-confident slice. The selective-
   risk objective optimizes "loss on the selected slice", which is
   the differentiable cousin of "1 - precision on the selected slice".
   Existing BCE optimizes loss UNIFORMLY across the score
   distribution; SelectiveNet routes more gradient toward the slice
   the live system actually trades.

2. **Regime-invariant trust region by construction.** Splits 1–7 cover
   bull / chop / drawdown regimes with base rates 6%–20%. A classifier
   that is uniformly competent across regimes is hard to learn (304
   attempts proved this); a classifier that knows WHEN to abstain is
   strictly easier. SelectiveNet fits the latter — even if f(x) cannot
   crack hard regimes, g(x) learns to refuse them, leaving f(x) free
   to specialize on the easy parts of the input space. Post-training
   bias calibration can then map "f score on g-trusted samples" → 0.6,
   which is precisely what we want.

3. **NO new architecture in production.** g(x) lives only in the
   trainer file. The saved LSTMModel state_dict has the same keys,
   shapes, and dtype as Entry 197. Production inference, the WF gate's
   model loader, and the live trader are untouched. Only the FC
   weights of the saved model differ — they were trained against a
   slightly different objective.

Why this is NOT one of the already-rejected mechanisms
-------------------------------------------------------
  * FP-penalty / survival-weighting (#47, #56, #198, #215, #218,
    REJECTED): tried to SHAPE TRAINING via *fixed* per-class loss
    multipliers. Repeatedly collapsed scores or yielded 0 trades.
    SelectiveNet's selection weight is LEARNED per-sample, so it does
    not impose a single global precision target.
  * Mean Teacher (#300, REJECTED): same architecture, EMA copy,
    consistency loss between teacher/student outputs. Targeted
    regularization. SelectiveNet's auxiliary head learns a NEW
    function (selection), not a copy of the existing one.
  * DANN (#296, REJECTED): adversarial alignment of train/val features
    via gradient reversal. SelectiveNet uses a NON-adversarial
    auxiliary head; no GRL, no domain classifier.
  * Quantile aux (#306, REJECTED): aux head predicts continuous PnL
    quantiles via pinball loss. Magnitude regression, not selection.
  * Precision-targeted bias calibration (Entry 308, ACTIVE baseline):
    POST-HOC threshold search using val labels. SelectiveNet shapes
    f(x) DURING training; the calibrator still runs afterwards on the
    SelectiveNet-trained model. They compose cleanly.
  * SelectiveNet itself: **never tried** in this project's 311-attempt
    history.

Companion changes for clean isolation
-------------------------------------
  * use_multifold_select default flipped True → False. Multi-fold
    val_loss selection (#311) was just rejected on the 7-split gate;
    keeping it on would muddle the SelectiveNet ablation.
  * daily_rank_active is suppressed when selective_active is True
    (the WF gate passes daily_rank_enabled=True; without this guard,
    pairwise ranking from #303 — also rejected — would silently
    activate when multifold is now off). The WF gate's call is
    deliberately overridden so the structural change is isolated.
  * All other rejected-mechanism opt-ins remain default OFF (DANN,
    SS-pretrain, Mean Teacher, GCE, quantile aux, temporal consist,
    listnet rank, curriculum, covariate shift, XGB distill,
    PnL-optimization).

Why this is SAFE
----------------
  - Saved model has identical signature to Entry 197 — gate loader,
    live trader, scaler are all unchanged.
  - Compute cost: one extra forward pass through SelectionHead
    (a 2-layer MLP on hidden_size=48 features) per training batch.
    ≪10% overhead vs the LSTM forward.
  - Degrades gracefully: if selective_active is False (sweep ablation
    or the WF gate explicitly passes use_selective_head=False), the
    code path is bit-identical to the (now multifold-OFF) Entry 197
    baseline.
  - Fixed seed (42) for both the LSTM and the SelectionHead — no
    additional source of run-to-run variance.

References
----------
Geifman & El-Yaniv (2019). "SelectiveNet: A Deep Neural Network with
an Integrated Reject Option." ICML.

---- LEGACY (historical context retained below) ----
PREVIOUS: MULTI-FOLD WORST-CASE VAL_LOSS SELECTION (#311, REJECTED).

Within each WF split's training run, the val set (the last 15% of each
split's train period) is partitioned into K=4 chronological mini-folds.
Each epoch we compute K separate val-loss values and use the WORST of
the K (the maximum) — not the mean — as the selection metric for:
  (1) early-stopping patience counter
  (2) SWA snapshot inclusion gate
  (3) ReduceLROnPlateau scheduler signal
The full-val mean loss is still computed and logged for diagnostics,
but it is NEVER used to make selection decisions when multifold is on.

Why worst-of-K val_loss attacks the cross-split-variance pathology
-----------------------------------------------------------------
The condensed lessons identify "massive cross-split variance" (wf_std
0.20-0.26 even on PASSING runs) as the primary failure mode: a model
that scores 60% WR on splits 1-3 and 20% on splits 4-7 averages 40%
but FAILS the 7-of-7 hard gate. Every prior structural attempt
(#294-310) has tried to attack this from the LOSS or ARCHITECTURE
side. None has touched the **model selection criterion**, which is
the actual bottleneck:
  - Currently: best snapshot = lowest mean val_loss
  - Mean val_loss is dominated by samples from the LATEST 25% of val
    (closest in time to the train cut-off)
  - That window's distribution is closest to train, so a model that
    overfits to train-era features looks great on mean val_loss
  - On test (which is in a DIFFERENT regime — 4-6 months later), the
    same overfit pattern collapses → low WR on hard splits

Multi-fold val_loss selection works by surfacing the temporal
heterogeneity that mean val_loss hides. If the model has truly learned
regime-invariant features, ALL K folds should have low loss. If it
has memorized a shortcut tied to the latest val time-slice, the
EARLIEST val fold (furthest from the train cut-off, hence furthest
from the train distribution) will have higher loss → worst-of-K stays
high → that snapshot is not selected → SWA averaging skips it →
training continues searching.

Why this is structurally different from prior selection-side attempts
---------------------------------------------------------------------
  * Precision@0.6 early-stop (#63, #64, #67, REJECTED): tried to pick
    a training epoch where the 0.6 threshold produced good precision.
    Unstable because (a) 0.6 is an arbitrary threshold mid-training
    and (b) it gated on a rare-event metric on a small slice. This
    change uses a STABLE loss-family metric across the WHOLE val,
    just partitioned chronologically.
  * Selectivity calibration (Entry 289, ACTIVE baseline): post-hoc
    bias shift after training. This new mechanism intervenes DURING
    training selection. Composes cleanly — the post-hoc calibrator
    still runs on the SWA-averaged weights chosen by multi-fold.
  * Precision-targeted bias calibration (Entry 308, post-hoc): same
    composition argument — runs AFTER the multi-fold selection
    determines which weights to keep.
  * Mean Teacher consistency (#300, REJECTED): targeted EARLY-EPOCH
    OVERFITTING via an EMA teacher. Multi-fold attacks the SAME
    pathology but via SELECTION (cheap, parameter-free) rather than
    via additional gradient signal (expensive, can fight with BCE).
  * SWA averaging itself (already active): improves boundaries by
    smoothing snapshots. Multi-fold makes WHICH snapshots SWA
    averages over more robust — they're now selected on worst-of-K,
    not mean. Composes; doesn't replace.

Why worst-of-K is the right aggregation
---------------------------------------
  - mean(K folds) = same as full-val mean loss → no new information.
  - max(K folds) = WORST-CASE robustness → selects models that
    generalize across the whole val period, not just the part most
    similar to train.
  - quantile(K, 0.75) = "soft worst-of-K" — useful but adds a tuning
    knob (which quantile?). max is parameter-free and matches the
    WF gate's own MIN_VALID_SPLITS=7 logic ("ALL splits must pass") —
    the model selection metric is now the same SHAPE as the gate.

Why this is SAFE
----------------
  - Zero architectural changes: no new parameters, no new optimizer
    state, no modification to LSTMModel or the inference pipeline.
  - Zero loss-function changes: training gradient is BCE (or whichever
    loss the gate enables) exactly as before.
  - Disabling daily_rank_active when multifold_active gives a pure
    "Entry 197 baseline + multi-fold selection" ablation. Pairwise
    (#303) and listnet (#310) ranking variants both failed; isolating
    THIS attempt's change avoids stacking with known-failed
    mechanisms.
  - K=4 is the maximum value where each fold gets >= 250 samples on
    typical val sizes (~5K samples). Smaller folds = noisier loss
    estimates = unreliable selection signal.
  - Compute cost: same as before. We collect val predictions once and
    compute K losses by slicing — pennies of overhead per epoch.
  - Degrades gracefully: if val is too small (<256 total) or the K
    partition produces folds smaller than `multifold_min_size`, we
    fall back to single-fold (full-val mean) selection.

Companion changes (clean baseline isolation):
  - use_listnet_rank default flipped True → False (since #310 was
    just rejected). When multifold is active, daily_rank_active is
    forced False anyway, so this only affects sweep-mode ablation.
  - All other rejected-mechanism opt-ins remain default OFF as before.

---- LEGACY (historical context retained below) ----
Previous attempt: LISTWISE TOP-K SOFTMAX RANKING
(ListNet-style) on per-day cross-sections using CONTINUOUS PnL.

Replaces the pairwise margin-hinge daily ranking (from #303, REJECTED
as default, but still passed unconditionally by the WF gate via
`daily_rank_enabled=True`). The replacement is structurally different:

  * Pairwise (previous):  for each day, form every (winner_y=1, loser_y=0)
    pair on BINARY labels. Apply margin-hinge: max(0, margin - (s_w - s_l)).
    Gradient vanishes once each pair is cleanly separated (hinge margin).
    Winners on +4.1% and winners on +20% are treated identically; the
    +4%/−4% fence-line noise leaks straight into the pair formation.

  * Listwise top-K (THIS):  for each day, take softmax over the model's
    scores and match a TARGET distribution that puts 1/K mass on the
    top-K stocks by REALIZED PnL (continuous). Cross-entropy loss.
    - Uses PnL magnitudes, not binary labels → no 4%-fence noise at all.
    - Gradient on a winner pushes it above ALL losers in one backward
      pass (not pair by pair), so rare big winners get strong signal.
    - Target mass is on the HEAD of the ranking (top 20%), which is
      exactly where we need precision — the live threshold at 0.6
      selects only ~top decile of predictions anyway.
    - Softmax is temperature-scalable; at T=0.5 the target puts most
      pressure on the single top prediction per day, encouraging the
      model to commit confidently to its daily best pick.

Why listwise top-K attacks the 37.5%-WR plateau
------------------------------------------------
  1. REGIME-INVARIANT BY CONSTRUCTION. The target distribution is a
     RELATIVE ordering within each day. Base rate shifts across splits
     (6%→20%) change absolute profitability but not "which stocks
     outperformed the market on this particular day". Ranking signal
     carries across regimes in a way that threshold-based classification
     cannot.
  2. CONTINUOUS PnL → NO FENCE-LINE NOISE. Binary y=(pnl>4%) mislabels
     the 3-5% ambiguity band as 0/1 noise. Listwise uses pnl_d directly
     for top-K selection: a +4.1% trade and a +3.9% trade are near the
     same rank and produce similar gradient, not opposite ones.
  3. FULL-DISTRIBUTION GRADIENT FLOW. For each day with k winners, the
     softmax cross-entropy produces a single gradient signal where
     raising any winner's score AND lowering all losers' scores
     simultaneously reduces the loss. Pairwise margin-hinge decomposes
     this into independent pair terms, many of which are already zero
     once separated by the margin. Listwise gives denser gradient
     throughout training, including late epochs where pairwise would
     have saturated.
  4. HEAD-FOCUSED TARGET. The top-K=20% target concentrates supervisory
     signal on the part of the score distribution that the live system
     actually uses (pred ≥ 0.6 corresponds to roughly the top decile
     per split after calibration). We're training the ranking structure
     EXACTLY at the decision boundary, not at the tail we never trade.

Why this is NOT one of the already-rejected recipes
---------------------------------------------------
  * #303 pairwise margin-hinge ranking: uses binary labels, forms pairs,
    hinge margin. Saturates once separated. REJECTED on 7-split gate.
    This replacement swaps the loss FAMILY entirely — it's not a
    hyperparameter tweak, it's a different objective function class.
  * #306/#308 quantile regression: predicts ABSOLUTE magnitudes of PnL
    quantiles. Brittle when PnL distribution shifts across splits (a
    +5% winner in a bull split vs. a +5% winner in a flat split have
    different base-rate implications). Listwise predicts RELATIVE
    rank within each day — no absolute magnitude assumed. REJECTED.
  * ListNet (Cao et al. 2007) is the foundational listwise ranking
    algorithm; top-K truncation (Cao et al. 2008) focuses learning
    on the head. The combination is a textbook LTR primitive and has
    never appeared in this project's history.

Mechanism
---------
  1. Per batch (BCE path only), after the main classification loss is
     computed, run a CLEAN forward pass on X_batch (unmixed) and collect
     sigmoid scores pred_clean.
  2. For each unique date in the batch with >= min_per_day samples:
       a. Convert sigmoid pred_clean to logits (unbounded scale).
       b. p_i = softmax(logit / T); T=0.5 sharpens distribution.
       c. q_i = 1/K for top-K stocks by realized PnL that day, 0 else.
       d. L_day = -sum(q_i · log p_i).
  3. Total listwise loss = mean(L_day) across qualifying days.
  4. Added to main loss with weight = daily_rank_lambda (default 0.5 —
     same as the pairwise predecessor, no re-tuning).
  5. Composes with mixup (mixup-batch produces main loss; clean batch
     produces listwise loss). Mixup is still the primary augmentation.
  6. Dispatches through the existing daily_rank_active gating — if the
     caller sets daily_rank_enabled=True (as the WF gate does), listwise
     runs. Setting use_listnet_rank=False reverts to pairwise #303.

Why this is SAFE
----------------
  - Zero architectural changes: no new parameters, no new optimizer
    state, no modification to LSTMModel or the inference pipeline.
  - Degrades gracefully: if pnl_train is not supplied, falls back to
    pairwise (binary-label) ranking as before.
  - Compute: one extra forward pass on the clean batch, identical to
    the pairwise path that's already running. No additional overhead.
  - Temporal-consistency regularization (#309, REJECTED) is now
    DEFAULT OFF so this attempt's mechanism is isolated. Flag still
    available for sweep ablation.

---- LEGACY (historical context retained below) ----
PREVIOUS: TEMPORAL-CONSISTENCY SUB-SEQUENCE REGULARIZATION.

For each training batch, run the model TWICE: once on the full-length
sequence X[:, 0:T, :], once on a START-TRUNCATED sequence X[:, trunc:T, :]
(default trunc=3). Both forward passes produce a per-sample sigmoid score;
the L2 distance between them is added to the loss as a regularizer.
Gradient flows through BOTH passes, coupling the LSTM's final hidden
state at t=T-1 across two sequence lengths. The intuition: a model that
genuinely relies on RECENT features (days T-10 .. T-1) should be only
mildly sensitive to dropping the oldest 3 observations; a model that
has memorized regime-specific shortcuts in the first few timesteps will
react strongly to truncation. Penalizing the L2 distance pushes training
toward the former.

Why temporal-consistency attacks the 37.5%-WR plateau and the
"epoch-0-best" pathology
----------------------------------------------------------------
  - The "epoch-0-best" effect recurs across LSTM, MLP (seq_len=1),
    pctrank features, survival labels, and tiny 4K-param models. The
    condensed lessons attribute it to a DATASET-LEVEL temporal
    distribution shift. Random-init + ~11% positive rate causes the
    LSTM to find a high-confidence shortcut within 1-2 epochs and
    the val loss never recovers.
  - The classic regularizers (dropout, weight_decay, input noise,
    mixup, SAM, DANN, SS-pretrain, GCE) all try to make the
    optimization LANDSCAPE smoother or the LOSS less eager to commit.
    None of them constrain WHAT the model can learn.
  - Sub-sequence consistency DOES constrain the learned function
    class: it says "your prediction at t=T-1 should be robust to
    removing the first 3 observations". This is a STRUCTURAL property
    that is violated by the typical epoch-0 shortcut — which tends
    to use distinctive patterns from the start of the sequence to
    memorize specific (symbol, date-range) pairs. Models that predict
    well from recent context alone are pushed toward features like
    atr_pct, bb_position, volume_ratio which are recency-local and
    REGIME-INVARIANT.
  - The constraint is LABEL-FREE: pure self-supervision between two
    forward passes of the same model. So it adds gradient signal
    that does not inherit the 4%-fence-line label noise. At ~11%
    positive rate, much of the BCE gradient is driven by mislabeled
    fence-line samples (pnl in [3%, 5%]); the consistency gradient
    dilutes this noise with a clean regime-invariant signal.

Why this is NOT one of the already-rejected consistency mechanisms
------------------------------------------------------------------
  * Mean Teacher (#300, REJECTED): EMA teacher on the SAME input;
    student predictions are pushed toward their own moving-average.
    Failure mode: the teacher = student at init, so the early signal
    is near-zero; by the time it diverges the student has already
    overfit. Sub-sequence consistency is DIFFERENT WEIGHTS vs
    DIFFERENT INPUT: same weights, trivially-different input. The
    consistency gradient is MEANINGFULLY NONZERO from epoch 0 because
    the two sequence lengths genuinely produce different forward
    passes (especially under seq_normalize, where truncation changes
    the normalization statistics).
  * SS pretraining (#298, REJECTED): masked-timestep reconstruction
    BEFORE supervised training. Separate phase, no gradient flow
    between pretext and label losses. Sub-sequence consistency is
    JOINT: the gradient from consistency and BCE flows through the
    same forward pass, so the LSTM is shaped by both signals
    simultaneously.
  * DANN (#296, REJECTED): gradient reversal on a domain classifier
    aligning train vs val feature distributions. Requires a val pool
    on device, expensive domain classifier, and two forward passes.
    Sub-sequence consistency removes the domain classifier entirely
    — the alignment target is the SAME MODEL on a different input,
    not a separate adversary.
  * FGSM (#307, REJECTED): input perturbation via sign(grad).
    Introduces adversarial OUT-OF-DISTRIBUTION inputs; model may
    overfit to the perturbation. Sub-sequence consistency only uses
    IN-DISTRIBUTION inputs (the truncation is still a valid market
    sequence, just shorter).

Mechanism
---------
  1. Per batch in the BCE path:
        pred_full = model(X_mix)               # shape (B,)
        pred_trunc = model(X_mix[:, trunc:, :])  # shape (B,)
        consist_loss = ((pred_full - pred_trunc) ** 2).mean()
     Both forward passes are in train() mode so they share dropout/
     input-noise stochastics — the consistency target is robust to
     these random realizations.
  2. Ramp-up: the weight linearly grows from 0 over the first
     consist_rampup_epochs. At epoch 0 the consistency loss is
     effectively disabled, letting the classifier find a sensible
     decision surface before the regularizer pulls on it. From epoch
     consist_rampup_epochs onward, the weight is at its maximum
     (consist_weight).
  3. consist_weight=0.3 is chosen so at (pred_full, pred_trunc) =
     (0.7, 0.3) the consistency term contributes ~0.048 to the loss
     — roughly 15-20% of typical class-weighted BCE (~0.3-0.5). Large
     enough to shape gradients, small enough that BCE pressure
     dominates.
  4. Truncation length = 3 out of seq_len=20 drops 15% of the
     sequence. Short enough that RECENT dynamics (days 3-19) cover
     everything the model needs; long enough that sequence
     normalization stats change meaningfully between the two passes.
  5. Active only on the BCE path (use_xgb_distill=False,
     use_pnl_loss=False). Those divergent loss families already
     replace the main classification objective with their own; adding
     another term would dilute further.
  6. Composes with mixup: the X_mix already-interpolated batch is
     what gets truncated, so both passes see mixup'd sequences. This
     is consistent because mixup is a pure input-space augmentation
     that preserves the "short-sequence is a valid market input"
     property.

Why this is SAFE
----------------
  - Same model weights are used for both passes → no new parameters,
    no state_dict changes, no architectural modification to LSTMModel.
  - The truncated forward pass uses the existing LSTMModel.forward
    path unmodified (seq_normalize, input_dropout, INPUT_NOISE,
    attention pooling all work on any T).
  - Degrades gracefully to prior Entry 197 recipe when
    use_temporal_consist=False.
  - Compute cost: ~1.4x per step (second forward is T-3 LSTM steps
    instead of T, plus minor overhead). Comfortably within the 30-min
    WF budget.

Companion change: use_quantile_aux default flipped from True → False.
Quantile aux (added default-on in #308, originally from #306 which was
rejected on the 7-split gate) introduced an extra pinball loss on a
Linear(hidden,5) auxiliary head. #308 with this aux enabled was
rejected (wf_avg_wr=0). Removing it here isolates the temporal-
consistency contribution and prevents stacking two distributional-
shape regularizers on the same backbone.

---- LEGACY (pre-#308) ----
Structural change (HISTORIC): PRECISION-TARGETED FC BIAS CALIBRATION.
Replaces the post-training selectivity calibration (Entry 289 — which
shifted the FC bias so the top 10% of val predictions landed at
LIVE_THRESHOLD=0.6) with a PRECISION-TARGETED search: given val labels,
find the threshold achieving best empirical precision within a reasonable
flag-count band [2%, 25%] of val, then map that threshold to 0.6.
Each walk-forward split runs its own calibration against its own val
period — so the 0.6 decision boundary AUTO-ADAPTS to the signal strength
and regime of each split. Leaves training untouched (raw LSTM ordering
preserved); only the bias relocates where predictions cross 0.6.

Why precision-targeted calibration attacks the 37.5%-WR plateau
---------------------------------------------------------------
  - The WF gate's PRIMARY signal is WR (must beat market base rate
    across all 7 splits). Selectivity is a PROXY that diverges from
    WR whenever the raw score distribution is not uniformly informative
    across its tail: "top 10% by score" can be 40%-precision on one
    split and 15%-precision on the next, even at the same model
    architecture, because the raw sigmoid has no inherent claim on
    where "good trades" cluster in score-space.
  - Precision-targeted calibration uses y_val to DIRECTLY optimize the
    metric the gate cares about. The search finds the threshold whose
    val-period precision is highest within a safe flag-count range,
    then aligns 0.6 with that threshold. For each WF split the
    threshold is computed on that split's val, so behavior adapts
    per-split rather than imposing a single global selectivity rule.
  - Splits with weak signal (low top-decile precision) push the
    threshold HIGHER → fewer, more selective trades at higher WR.
    Splits with strong signal can push the threshold LOWER while
    still clearing target precision. This is the correct response to
    regime uncertainty — trade less when noisy, more when confident.
    Old selectivity calibration forced 10% regardless of signal.
  - The [2%, 25%] flag-count band enforces two constraints: (a)
    precision estimates are reliable (≥100 flagged val samples when
    val has ≥5K samples), (b) test-set flagging rate lands in a
    sensible range — extrapolating 2% val selectivity to a ~1K-sample
    test gives ≥20 trades, well above MIN_TRADES_PER_SPLIT=10.
  - Falls back to selectivity calibration when: no y_val supplied,
    or no threshold in the flag band has precision > base rate
    (signals zero edge), or val predictions are degenerate. So the
    new path never PERFORMS WORSE than the old path — it strictly
    augments it with label information.

Why this is NOT one of the already-rejected recipes
---------------------------------------------------
  * FP-penalty / survival-weighting (#47, #56, #198, #215, #218 —
    REJECTED). Tried to SHAPE TRAINING so high-confidence predictions
    were more selective. Repeatedly collapsed scores or yielded 0
    trades. This change leaves training untouched; the raw model's
    learned score ORDERING is preserved and only the decision
    boundary's location on that ordering shifts.
  * Precision@0.6 early stopping (#63, #64, #67 — REJECTED). Tried
    to pick a TRAINING epoch where the 0.6 threshold happened to
    produce good precision. Unstable because (a) 0.6 is an arbitrary
    threshold mid-training, (b) stopping on a rare-event metric
    computed on a small held-out slice is noisy. This change uses
    the WHOLE val set and explicitly searches over thresholds, after
    SWA has smoothed training noise.
  * Conformal prediction (explicitly listed as untried). Precision-
    targeted calibration is a practical relative of split-conformal
    selection: we compute an empirical nonconformity score (predicted
    vs realized) on val, pick the threshold that meets a precision
    target, then push it to 0.6 so downstream code doesn't change.
    Full conformal requires exchangeability (our setting is not), but
    the empirical threshold selection remains valid as a heuristic.
  * Old selectivity calibration (#289, ACTIVE baseline). Same math
    (monotone logit-shift of FC bias), but only uses the score
    ordering — no y_val. Old path is STILL available as the fallback.

Why distributional regression attacks the 37.5%-WR plateau
----------------------------------------------------------
  - The binary label y = (pnl > MIN_PROFIT_PCT=4%) has a structural
    ambiguity band: a +4.1% trade labels as winner, +3.9% as loser,
    despite near-identical trajectories. BCE (and GCE) treat these as
    HARD 0/1 labels — the magnitude information is discarded entirely.
    304 classification-based attempts peaked at 37.5% WR, and all
    variants of "smooth the loss around the threshold" (GCE/#305,
    label smoothing/#72, noise-robust losses) either didn't help or
    collapsed scores.
  - Quantile regression on raw PnL does NOT have a threshold. The
    pinball loss at quantile τ is magnitude-aware: predicting p50=+0.05
    when actual pnl=+0.041 contributes a small error; predicting
    p50=+0.05 when actual pnl=-0.03 contributes a large error. The
    gradient INTO the shared LSTM features therefore encodes "how much
    did I miss the magnitude by", not "was I on the right side of a
    fence". This is the fundamental information that all prior
    classification-loss attempts erased.
  - Multi-quantile prediction (τ ∈ {0.1, 0.25, 0.5, 0.75, 0.9}) forces
    the shared representation to encode FOUR distributional facts per
    sample: (1) central tendency / p50, (2) downside tail / p10 — the
    "stop-loss likely" case, (3) upside tail / p90 — the "target hit"
    case, (4) asymmetry / (p90-p50) vs (p50-p10). Asymmetry is
    precisely the signal we need: stocks with FAVORABLE SKEW — wide
    upside, tight downside — are the +15% candidates. Binary labels
    reveal asymmetry only by accident; explicit quantiles reveal it
    directly.

Why this is orthogonal to prior PnL-related attempts
----------------------------------------------------
  * #199 (PnL-weighted BCE — REJECTED): scaled per-sample loss by |pnl|.
    Still a CLASSIFICATION task, just with non-uniform sample weights.
    Gradient direction (positive↑ / negative↓) was unchanged.
    Quantile regression changes the TASK to REGRESSION on continuous
    magnitude, giving the features magnitude-gradient information
    that weighted BCE cannot.
  * #290 (PnL-optimization — REJECTED): replaced BCE with -E[p·(pnl -
    commission)], maximizing expected realized profit. Treated pred as
    a scalar "entry propensity"; had no distributional modeling and no
    notion of tail risk. Quantile regression gives the features access
    to {p10,p25,p50,p75,p90} — FIVE separate scalars about each
    sample's outcome distribution.
  * #304 (PnL-magnitude curriculum — REJECTED): filtered BATCHES by
    |pnl| thresholds. Still a CLASSIFICATION task on the survivors.
    Quantile regression trains on ALL samples (including small-|pnl|
    fence-liners) but contributes LESS gradient on them — because the
    pinball loss on a p50=+0.01 prediction for pnl=+0.02 is tiny.
    Soft magnitude weighting emerges naturally from the loss geometry,
    rather than being heuristically imposed via a curriculum schedule.
  * #305 (GCE — REJECTED, 0 trades): kept the binary task; softened
    the loss gradient. Score collapse (q≈0.5 everywhere → nothing
    clears 0.6) is the textbook failure mode when the loss geometry
    doesn't push for confident predictions. Quantile regression
    doesn't touch the classifier's BCE loss geometry — the classifier
    still has strong pressure to separate classes — so score collapse
    is not a risk.

Mechanism
---------
  1. Build a small auxiliary head: q_head = nn.Linear(hidden_size,
     len(quantile_levels)) — 5*hidden_size + 5 extra parameters on
     hidden=48 → 245 params (negligible compared to the ~10K LSTM
     params). Head lives on the trainer side, NOT in LSTMModel.
  2. On the BCE path (the only default path), the shared forward pass
     is refactored:
        feat = _lstm_features(model, X_mix)   # pre-FC pooled (B, H)
        pred = _head_forward(model, feat)     # main classifier output
        q_pred = q_head(feat)                 # (B, n_quantiles)
     This gives the aux head access to the EXACT SAME features the
     classifier uses, with the SAME dropout/noise realizations, so
     both heads are trained on identical representations.
  3. pnl_mix (mixup-consistent PnL target) is computed alongside y_mix
     in the existing mixup block. For fidelity with the rest of the
     mixup interpolation, we mix PnL linearly too — a convex
     combination of two adjacent-day PnLs is still a valid PnL-sample
     drawn from the same distribution, just softened.
  4. Loss composition:
        total_loss = class_weighted_bce(pred, y_mix)
                   + quantile_aux_weight * pinball(q_pred, pnl_clip, levels)
                   + (any existing aux losses — daily_rank, etc.)
  5. q_head.parameters() are added to the same Adam optimizer as the
     LSTM parameters. Gradient flows LSTM → pooled features → q_head
     and back, so the aux loss shapes the BACKBONE, not just the head.
  6. PnL clipping at [-0.05, +0.20] bounds the regression target to
     realistic trading outcomes (hard SL @ -3% + some slack; hard
     target @ +15% + some slack). Without clipping, one outlier +80%
     gap day would pull p90 up for months of trades.

Safety properties
-----------------
  - q_head is NOT in LSTMModel.state_dict() → saved model is
    bit-identical to the baseline's save format. Production inference
    does not see or need it.
  - Auxiliary loss scales the main loss; default quantile_aux_weight
    = 0.4 chosen so typical pinball loss (~0.01-0.03 on centered PnL)
    and typical class-weighted BCE (~0.3-0.7) contribute roughly
    comparable GRADIENT magnitudes after the weight multiplier. Too
    low → no representational effect; too high → classifier pressure
    drowned by regression pressure.
  - Degrades gracefully to pure Entry 197 BCE when pnl_train is
    unavailable, quantile_aux_weight=0, or use_quantile_aux=False.
  - Quantile prediction is monotonicity-preserving in expectation —
    the q_head outputs (B, 5) unordered values, but the pinball loss
    at increasing quantiles will pressure them into the right order.
    We do NOT add a monotonicity penalty: the classification task
    provides the downstream signal; the aux head exists to SHAPE
    features, not to be read at inference time.
  - Default use_gce is now FALSE (reverting #305).

Legacy note: GENERALIZED CROSS ENTROPY (GCE) loss remains available
as an opt-in (use_gce=True). Its original rationale and implementation
are preserved below; #305's default-on status is removed because it
collapsed scores on the WF gate.

GCE formula (binary, supports soft mixup targets)
-------------------------------------------------
For prediction p ∈ (0,1) and target y ∈ [0,1]:
    L_q(p, y) = y · (1 - p^q)/q   +   (1-y) · (1 - (1-p)^q)/q

  - q → 0 recovers standard cross-entropy (NOT robust to label noise)
  - q → 1 recovers mean absolute error (MAE, fully robust but slow
    to converge because the gradient magnitude is independent of how
    wrong the prediction is)
  - q = 0.7 is the Zhang & Sabuncu default, which sits on the Pareto
    frontier of convergence speed and noise robustness.

Why this attacks the 37.5%-WR plateau and the "epoch-0-best" pathology
----------------------------------------------------------------------
  - All 304 prior attempts exhibit val_loss ceilings at epoch 0-2.
    The lessons attribute this to "repeated candidate/gate misalign-
    ment" and to "train → test distribution shift that random-init
    SGD captures by fitting shortcut features before gradient noise
    averages out." Both of those hypotheses share one mechanism at
    the loss level: BCE has UNBOUNDED gradient magnitude for
    confident-wrong predictions. When the early-epoch model guesses
    on a noisy-label sample and guesses wrong, BCE drives an
    enormous gradient into the LSTM to FIT that one sample — exactly
    the "memorize shortcut features" failure mode.
  - The binary label y = (pnl > MIN_PROFIT_PCT=4%) has a thick
    ambiguity band by construction: a +4.1% trade labels as winner,
    +3.9% as loser, despite near-identical trajectories. These
    fence-line samples are effectively mislabeled relative to the
    signal the model CAN learn. Their noise rate is not 0% — it is
    probably 20-40% in the 3-5% pnl band — and BCE treats every one
    of them as ground truth.
  - GCE's gradient for (p → 0, y = 1) is -p^(q-1) = -p^(-0.3),
    which is bounded near the optimum instead of diverging. Large
    errors on individual samples no longer produce disproportionate
    gradient; the optimizer must see many consistent examples to
    shift the decision boundary. This is exactly the "ignore
    mislabeled fence-line samples, learn the robust signal from big
    wins and clean losers" behavior that past attempts tried to
    engineer via curriculum (#304), sample reweighting (#199, #261),
    and regime-conditional heads (#301) — all rejected.
  - Explicitly flagged as UNTRIED in the condensed lessons:
    "label-noise-robust losses (GCE, SCE) given the epoch-0
    memorization pattern."

Why this is NOT one of the already-rejected loss tweaks
-------------------------------------------------------
  * Label smoothing (#47, #72 — REJECTED) MOVES the target toward
    0.5, which eliminates the high-confidence objective entirely and
    collapses the score distribution below LIVE_THRESHOLD=0.6. GCE
    keeps the hard 0/1 target; only the LOSS GEOMETRY changes.
  * PnL-magnitude loss weighting (#199 — REJECTED) scaled per-sample
    loss by |pnl|, treating a -3% stop identically to a +3% scrape-
    through. GCE treats ALL per-sample losses identically on the
    target side; the robustness comes from gradient clipping in
    prediction space, not from sample weighting.
  * Focal loss (not explicitly rejected but in the same family):
    down-weights well-classified examples — the opposite of what's
    needed here. In our problem, confidently-correct predictions on
    big wins/clean losers are the SIGNAL, and fence-line noisy-
    labeled examples are what we need to de-emphasize. GCE does the
    right thing structurally.
  * Curriculum learning (#304, REJECTED) filters WHICH samples
    contribute per epoch; GCE changes the LOSS applied to every
    sample. Curriculum also had a heuristic schedule (strong→full)
    whose optimal timing is unknown; GCE's q=0.7 is a well-
    characterized default from the seminal paper on this exact
    problem (noisy-label deep classification).

Mechanism
---------
  1. Implement gce_loss(pred, y, q) as a drop-in per-sample loss
     function returning (N,)-shape tensor, matching BCELoss(reduction
     ='none').
  2. In the supervised-training path, when use_gce=True the per-sample
     loss is GCE; when False it is BCE (backwards-compatible).
  3. Mixup targets y_mix ∈ [0, 1] are handled natively by the GCE
     formula (y interpolates linearly between positive and negative
     branches, same as BCE).
  4. pos_weight class balancing is KEPT unchanged — multiplied
     per-sample onto the GCE output. We still want to rebalance the
     9% base rate; noise robustness is orthogonal to class imbalance.
  5. All other mechanisms (mixup, SWA, Adam, covariate-shift
     reweighting, daily-rank auxiliary, FC-bias calibration) remain
     as-is.

Safety properties
-----------------
  - At q → 0 (not recommended, but possible) GCE converges to CE up
    to a constant, so nothing degrades below the Entry 197 baseline.
  - GCE output is always in [0, 1/q] since (1 - p^q) ∈ [0,1] and the
    division by q bounds the magnitude; q=0.7 gives max loss ≈ 1.43
    per sample, comparable to BCE's ~-log(eps) ≈ 14 for confident
    wrong predictions — i.e., strictly smaller, never blowing up.
  - Curriculum is now DEFAULT OFF (use_curriculum=False). The WF
    gate does not explicitly pass use_curriculum, so the default
    propagates. Keeps #304's code available but inactive.

Legacy mechanisms retained as opt-in knobs (all default OFF):
  use_mean_teacher    (#300, rejected) — EMA teacher consistency
  use_ss_pretrain     (#298, rejected) — autoregressive next-step pretext
  use_dann            (#296, rejected) — Ganin adversarial alignment
  use_covariate_shift (#295, rejected) — density-ratio BCE reweighting
  use_xgb_distill     (#291, rejected) — soft-target distillation
  use_pnl_loss        (#290, rejected) — differentiable expected-profit
  use_curriculum      (#304, rejected) — PnL-magnitude curriculum
  daily_rank_enabled  (#303, rejected) — pairwise margin-hinge ranking
                                         (WF gate still passes True)

Preserves the Entry 197 recipe:
  - Mixup α=0.3
  - Class-balanced per-sample loss (pos_weight) — now GCE instead of BCE
  - SWA weight averaging over val-loss-improving snapshots
  - Post-training FC-bias calibration (#289) so the top 10% of val
    propensities lands at LIVE_THRESHOLD=0.6
"""
import os
import sys
import json
import hashlib
import argparse
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import sqlite3
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from lstm_model import LSTMModel
from feature_eng import prepare_data, BASE_FEATURES
from labels import label_trade, label_trade_with_pnl, check_early_stop, STOP_PCT, TARGET_PCT, TRAILING_TRIGGER, TRAILING_FLOOR, MAX_HOLD, MIN_PROFIT_PCT

BASE_PATH = '/home/kanoonth-ai/projects/caffe-stocks'
DB_PATH = os.path.join(BASE_PATH, 'data', 'candles.db')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2020, arXiv:2010.01412).

    At each step:
      1. Ascend to the worst-case point within a ρ-ball:  w ← w + ρ·g/||g||
      2. Compute gradient at the perturbed point
      3. Step back to original w and descend using the perturbed-point grad

    Converges to FLAT minima rather than sharp ones. Flat minima generalize
    substantially better under covariate shift — the exact pathology behind
    the 293-attempt 37.5% WR ceiling and the universal "epoch-0-best" effect
    (sharp minima fit the train-era distribution so tightly that any shift
    drives val/test loss up immediately). SAM targets this structurally by
    constraining convergence to low-loss *neighborhoods*, not points.

    Trades ~2x compute per step for fundamentally more robust generalization.
    rho=0.05 is the Foret et al. default.
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        assert rho >= 0.0, f'Invalid rho: {rho}'
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'e_w' not in self.state[p]:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        if closure is None:
            self.base_optimizer.step()
            return None
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return None

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        stacked = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    stacked.append(p.grad.norm(p=2).to(shared_device))
        if not stacked:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(stacked), p=2)


# Must match walk_forward_gate.py's LSTM_THRESHOLD and live inference threshold.
LIVE_THRESHOLD = 0.6
# Target val selectivity for bias calibration — the fraction of val samples
# whose predictions should clear LIVE_THRESHOLD after calibration. 0.10 sits
# between the ~9% positive base rate (would be the oracle) and the monitor's
# 30% spray cap, leaving headroom for test-set regime drift. Translates to
# ~10% * test_size trades per split: 200-500 (comfortably above the
# MIN_TRADES_PER_SPLIT=10 gate, well below the spray monitor).
TARGET_VAL_SELECTIVITY = 0.10
# Lower bound on val size for bias calibration — below this, quantile
# estimates are noisy enough to over/under shoot target selectivity.
MIN_VAL_FOR_CALIBRATION = 200


def load_sequences(seq_len=20, lookahead=10):
    conn = sqlite3.connect(DB_PATH)
    data = pd.read_sql_query('SELECT * FROM candles ORDER BY timestamp', conn)
    conn.close()
    data = data.sort_values('timestamp')

    if len(data) == 0:
        raise RuntimeError('candles table is empty — run compute_indicators.py first')

    missing = [f for f in BASE_FEATURES if f not in data.columns]
    if missing:
        raise RuntimeError(f'Missing features: {missing}')

    data, features = prepare_data(data)

    # Cache key (v5: clean baseline — no xsec_demean, matches #288 revert)
    cache_key_parts = [
        str(os.path.getmtime(DB_PATH)),
        ','.join(features),
        f'{seq_len},{lookahead}',
        f'{STOP_PCT},{TARGET_PCT},{TRAILING_TRIGGER},{TRAILING_FLOOR},{MAX_HOLD}',
        f'min_profit={MIN_PROFIT_PCT}',
        'v5_clean_baseline',
    ]
    cache_hash = hashlib.md5('|'.join(cache_key_parts).encode()).hexdigest()[:12]
    cache_path = os.path.join(BASE_PATH, 'data', f'sequences_cache_{cache_hash}.npz')

    if os.path.exists(cache_path):
        print(f'Loading cached sequences: {cache_path}')
        cached = np.load(cache_path, allow_pickle=True)
        pnl = cached['pnl'] if 'pnl' in cached else None
        return cached['X'], cached['y'], cached['dates'], cached['early_sl'], features, pnl

    X_all, y_all, pnl_all, dates_all, early_sl_all = [], [], [], [], []

    for symbol in data['symbol'].unique():
        sdf = data[data['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        if len(sdf) < seq_len + lookahead:
            continue

        for i in range(len(sdf) - seq_len - lookahead):
            seq = sdf[features].iloc[i:i+seq_len].values
            if np.isnan(seq).any():
                continue
            X_all.append(seq)
            dates_all.append(sdf.iloc[i+seq_len-1]['timestamp'])

            entry_price = sdf.iloc[i+seq_len-1]['close']
            window = sdf.iloc[i+seq_len:i+seq_len+lookahead]
            label, pnl = label_trade_with_pnl(
                window['high'].values, window['low'].values,
                window['close'].values, entry_price)
            y_all.append(1 if pnl > MIN_PROFIT_PCT else 0)
            pnl_all.append(pnl)

            early_sl = check_early_stop(window['low'].values, entry_price)
            early_sl_all.append(early_sl)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    pnl = np.array(pnl_all, dtype=np.float32)
    dates = np.array(dates_all)
    early_sl = np.array(early_sl_all, dtype=np.float32)

    np.savez(cache_path, X=X, y=y, pnl=pnl, dates=dates, early_sl=early_sl)
    import glob as globmod
    for old in sorted(globmod.glob(os.path.join(BASE_PATH, 'data', 'sequences_cache_*.npz')))[:-2]:
        os.remove(old)
    print(f'Sequences cached: {cache_path}')

    return X, y, dates, early_sl, features, pnl


def time_split(X, y, dates, early_sl=None, pnl=None, train_pct=0.70, val_pct=0.85):
    sort_idx = np.argsort(dates)
    X, y, dates = X[sort_idx], y[sort_idx], dates[sort_idx]
    if early_sl is not None:
        early_sl = early_sl[sort_idx]
    if pnl is not None:
        pnl = pnl[sort_idx]

    unique_dates = np.unique(dates)
    train_cutoff = unique_dates[int(train_pct * len(unique_dates))]
    val_cutoff = unique_dates[int(val_pct * len(unique_dates))]

    train_mask = dates < train_cutoff
    val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
    test_mask = dates >= val_cutoff

    splits = {
        'X_train': X[train_mask], 'X_val': X[val_mask], 'X_test': X[test_mask],
        'y_train': y[train_mask], 'y_val': y[val_mask], 'y_test': y[test_mask],
        'train_dates': dates[train_mask], 'val_dates': dates[val_mask], 'test_dates': dates[test_mask],
    }

    if early_sl is not None:
        splits['early_sl_train'] = early_sl[train_mask]
        splits['early_sl_val'] = early_sl[val_mask]
        splits['early_sl_test'] = early_sl[test_mask]

    if pnl is not None:
        splits['pnl_train'] = pnl[train_mask]
        splits['pnl_val'] = pnl[val_mask]
        splits['pnl_test'] = pnl[test_mask]

    assert max(splits['train_dates']) < min(splits['val_dates']), "Data leakage: train/val overlap"
    assert max(splits['val_dates']) < min(splits['test_dates']), "Data leakage: val/test overlap"
    return splits


def _aggregate_sequence_np(X):
    """Aggregate (N, seq_len, F) sequences into (N, 4F): last | mean | std | (last-mean).
    Matches the model's MLP-mode aggregation, extended with std for dispersion info.
    Deterministic — no randomness — so the teacher sees the same representation
    on every retrain within a walk-forward split.
    """
    last = X[:, -1, :].astype(np.float32)
    mean = X.mean(axis=1).astype(np.float32)
    std = X.std(axis=1).astype(np.float32)
    dev = last - mean
    return np.concatenate([last, mean, std, dev], axis=1)


def _train_xgb_teacher(X_train_scaled, y_train, X_val_scaled, y_val, verbose=False):
    """Train an XGBoost teacher on aggregated sequence features.
    Returns (train_soft, val_soft) — probability of positive class.

    Regularization stack is deliberately conservative: shallow trees,
    aggressive feature/row subsampling, high min_child_weight. The goal
    is generalization stability across WF splits, not peak training AUC.
    """
    try:
        import xgboost as xgb
    except ImportError:
        if verbose:
            print('  XGBoost not installed — distillation disabled, falling back')
        return None, None, None

    X_tr_agg = _aggregate_sequence_np(X_train_scaled)
    X_val_agg = _aggregate_sequence_np(X_val_scaled)

    pos_rate = float(np.mean(y_train))
    spw = float(min((1.0 - pos_rate) / max(pos_rate, 1e-6), 15.0))

    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        scale_pos_weight=spw,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=10,
        gamma=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=25,
        random_state=42,
        n_jobs=2,
        tree_method='hist',
        verbosity=0,
    )
    try:
        clf.fit(
            X_tr_agg, y_train,
            eval_set=[(X_val_agg, y_val)],
            verbose=False,
        )
    except Exception as exc:
        if verbose:
            print(f'  XGB teacher fit failed: {exc} — distillation disabled')
        return None, None, None

    train_soft = clf.predict_proba(X_tr_agg)[:, 1].astype(np.float32)
    val_soft = clf.predict_proba(X_val_agg)[:, 1].astype(np.float32)

    if verbose:
        try:
            from sklearn.metrics import roc_auc_score
            tr_auc = float(roc_auc_score(y_train, train_soft)) if len(set(y_train)) > 1 else float('nan')
            va_auc = float(roc_auc_score(y_val, val_soft)) if len(set(y_val)) > 1 else float('nan')
            print(f'  XGB teacher: best_iter={getattr(clf, "best_iteration", "?")} '
                  f'train_AUC={tr_auc:.3f} val_AUC={va_auc:.3f}')
            print(f'    val soft mean={float(val_soft.mean()):.3f} '
                  f'Q90={float(np.quantile(val_soft, 0.90)):.3f} '
                  f'≥0.5 frac={(val_soft >= 0.5).mean():.3f}')
        except Exception:
            pass

    return train_soft, val_soft, clf


def _label_cleaning_oof_xgb(X_train_scaled, y_train, n_folds=3,
                              noise_frac=0.10, verbose=False):
    """Confident-learning label cleaning via OOF XGBoost (THIS ATTEMPT #332).

    Northcutt, Jiang, Chuang (2021), "Confident Learning: Estimating
    Uncertainty in Dataset Labels", JAIR. Listed as untried option (e)
    in the condensed lessons:
       "Confident-learning / label-cleaning to prune ambiguous
        near-threshold trades before training."

    Algorithm
    ---------
      1. Aggregate (N, T, F) sequences to (N, 4F) via the existing
         _aggregate_sequence_np helper (last | mean | std | last-mean).
      2. K-fold the train set CHRONOLOGICALLY. For each fold k, fit an
         XGBoost classifier on the OTHER K-1 folds and predict on fold k.
         Concatenate the K held-out predictions to get an OUT-OF-FOLD
         probability per train sample (no sample's own label leaks
         into its own OOF prediction).
      3. Per-sample noise score = P(WRONG class) under OOF model:
              y=1 -> noise = 1 - p_oof
              y=0 -> noise =     p_oof
         High noise = model strongly disagrees with the binary label.
      4. Remove the top `noise_frac` fraction of samples by noise score.
      5. Return a boolean keep_mask. The trainer applies it to
         X_train_scaled, y_train, dates_train, pnl_train, and
         early_sl_train BEFORE any downstream per-sample bookkeeping.

    Why this attacks the 37.5%-WR plateau / "epoch-0-best" pathology
    ---------------------------------------------------------------
    The condensed lessons identify the +4% fence-line as a major source
    of effective label noise: a +4.1% trade and a +3.9% trade follow
    near-identical trajectories but receive opposite binary labels. With
    BCE's UNBOUNDED gradient on confident-wrong predictions and the
    optimizer's tendency to commit to a sharp shortcut within 1-2 epochs
    (driving the well-documented "epoch-0-best" effect), a single noisy
    fence-line sample drives an arbitrarily large gradient pull. The
    optimizer fits these noisy samples first, val_loss diverges
    immediately because those memorized shortcuts do not generalize,
    and we plateau at 37.5% WR.

    Confident learning attacks this AT THE DATA LEVEL using INDEPENDENT
    MODEL EVIDENCE. XGBoost extracts different feature combinations than
    the LSTM (tree splits on static aggregations vs recurrent dynamics),
    so its OOF disagreement with a binary label is a genuinely
    independent signal of label unreliability — not a tautological
    "the LSTM doesn't learn this sample" check.

    Why this is NOT one of the already-rejected mechanisms
    ------------------------------------------------------
      * #291 XGB teacher distillation (REJECTED): used XGB's in-sample
        probabilities as SOFT TARGETS during BCE training. Every train
        sample participated; only the loss target softened. Confident
        learning REMOVES the noisiest samples; surviving samples train
        on their original BINARY labels.
      * #292 tree-ensemble distillation (REJECTED): same family —
        soft-target distillation. Same critique.
      * #199 PnL-magnitude weighted BCE (REJECTED): per-sample
        REWEIGHTING by |pnl|. Magnitude-driven; doesn't identify
        mislabeled samples. Confident learning identifies them via
        OOF disagreement, then removes them entirely.
      * #304 PnL-magnitude curriculum (REJECTED): hard-thresholded
        sample FILTERING by |PnL| in early epochs. PnL-magnitude based.
        Confident learning is MODEL-EVIDENCE based: a +4.1% sample that
        XGB confidently labels positive is KEPT (model agrees with the
        label), while a +4.1% sample XGB labels negative is REMOVED
        (model disagrees -> likely mislabeled).
      * #317 boundary-band sample exclusion (REJECTED): hard SAMPLE
        filtering by |pnl - 4%| < band. Same critique as #304.
      * #295 covariate-shift density-ratio (REJECTED): SAMPLE-level
        weighting via a learned domain classifier (predicts train-vs-val
        origin). Different mechanism — distribution alignment, not
        label-noise identification.
      * #321 PnL-distance soft labels (REJECTED): replaced binary y
        with sigmoid((pnl - 0.04) / scale). Continuous LABELS based on
        PnL MAGNITUDE. Confident learning keeps binary labels and
        REMOVES samples based on MODEL EVIDENCE.
      * Listed as UNTRIED in the condensed lessons (option e).

    Why this is SAFE
    ----------------
      - Saved model has IDENTICAL signature to Entry 197. Gate loader,
        live trader, scaler are all unchanged. Production inference is
        bit-identical regardless of whether label cleaning was applied.
      - Conservative defaults: noise_frac=0.10 removes only the top 10%
        most-disagreed-with samples. With ~10.8% positive rate this
        primarily targets samples in the noisy +3-5% PnL band where
        the binary label is least reliable.
      - Asymmetric guard: if cleaning would remove >50% of positives
        (would degenerate the rare-class signal), the cleaning is
        skipped and the un-cleaned dataset is used.
      - K=3 chronological folds — each XGB sees 2/3 of train. Modest
        compute (~5-15 sec per fold on aggregated features).
      - All downstream computations (covariate-shift weights, time-decay
        weights, env IDs, dates, pnl) are derived AFTER label cleaning,
        so they reference the cleaned dataset by construction.
      - Degrades gracefully:
          * use_label_cleaning=False (sweep ablation): no-op.
          * XGBoost unavailable: skipped (returns all-True keep mask).
          * <300 train samples: skipped.
          * Degenerate labels: skipped.
          * Too many fold failures: skipped.

    Companion change for clean isolation
    ------------------------------------
      * use_brier_loss default flipped True -> False (since #331 was
        just rejected).
      * daily_rank_active is suppressed when label_cleaning_active is
        True. The WF gate passes daily_rank_enabled=True; without this
        guard the rejected pairwise (#303) / listnet (#310) rankers
        would silently activate alongside this attempt's mechanism,
        muddling the ablation.
      * All other rejected-mechanism opt-ins remain default OFF.

    Reference
    ---------
    Northcutt, Jiang, Chuang (2021). "Confident Learning: Estimating
    Uncertainty in Dataset Labels." JAIR.
    """
    n = len(y_train)
    try:
        import xgboost as xgb
    except ImportError:
        if verbose:
            print('  Label cleaning: XGBoost unavailable — skipped (no-op)')
        return np.ones(n, dtype=bool), {'applied': False, 'reason': 'no xgboost'}

    if n < 300:
        return np.ones(n, dtype=bool), {'applied': False,
                                         'reason': f'too few samples ({n})'}

    pos_rate = float(np.mean(y_train))
    if pos_rate <= 0.0 or pos_rate >= 1.0:
        return np.ones(n, dtype=bool), {
            'applied': False,
            'reason': f'degenerate labels (pos_rate={pos_rate:.3f})'}
    spw = float(min((1.0 - pos_rate) / max(pos_rate, 1e-6), 15.0))

    X_agg = _aggregate_sequence_np(X_train_scaled)

    fold_size = n // int(n_folds)
    oof_preds = np.full(n, -1.0, dtype=np.float32)
    fold_aucs = []

    for k in range(int(n_folds)):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < int(n_folds) - 1 else n
        val_idx = np.arange(val_start, val_end)
        train_idx = np.concatenate([
            np.arange(0, val_start),
            np.arange(val_end, n),
        ])
        if len(train_idx) < 50 or len(val_idx) < 10:
            continue
        if len(set(np.asarray(y_train)[train_idx])) < 2:
            if verbose:
                print(f'  Label cleaning: fold {k} train has single class — skipping')
            continue
        try:
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.6,
                scale_pos_weight=spw,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=10,
                objective='binary:logistic',
                random_state=42 + k,
                n_jobs=2,
                tree_method='hist',
                verbosity=0,
            )
            clf.fit(X_agg[train_idx], np.asarray(y_train)[train_idx])
            oof_preds[val_idx] = clf.predict_proba(
                X_agg[val_idx])[:, 1].astype(np.float32)
            try:
                from sklearn.metrics import roc_auc_score
                if len(set(np.asarray(y_train)[val_idx])) > 1:
                    fold_aucs.append(float(roc_auc_score(
                        np.asarray(y_train)[val_idx], oof_preds[val_idx])))
            except Exception:
                pass
        except Exception as exc:
            if verbose:
                print(f'  Label cleaning: fold {k} XGB fit failed ({exc}) — skipping')
            continue

    valid_mask = oof_preds >= 0.0
    if int(valid_mask.sum()) < 0.5 * n:
        return np.ones(n, dtype=bool), {
            'applied': False,
            'reason': (f'too many fold failures '
                       f'({int(valid_mask.sum())}/{n} valid OOF)'),
        }

    oof_clipped = np.clip(oof_preds, 0.0, 1.0).astype(np.float32)
    y_arr = np.asarray(y_train).astype(np.float32)
    noise_score_all = np.where(
        y_arr > 0.5,
        1.0 - oof_clipped,
        oof_clipped,
    ).astype(np.float32)
    noise_score = np.where(valid_mask, noise_score_all, 0.0).astype(np.float32)

    # CLASS-CONDITIONAL pruning (Northcutt-style). With ~10.8% positive
    # rate, a global noise_score threshold is dominated by positives
    # (most have P(pos) < 0.5 under XGB regardless of label correctness).
    # Computing the (1 - noise_frac) quantile WITHIN EACH CLASS removes
    # the top noise_frac of MISLABELED-LOOKING samples in each class
    # while preserving class balance.
    y_bool = y_arr > 0.5
    pos_valid = valid_mask & y_bool
    neg_valid = valid_mask & ~y_bool
    n_pos_valid = int(pos_valid.sum())
    n_neg_valid = int(neg_valid.sum())
    if n_pos_valid < 20 or n_neg_valid < 20:
        return np.ones(n, dtype=bool), {
            'applied': False,
            'reason': (f'too few per-class valid OOF '
                       f'(pos={n_pos_valid}, neg={n_neg_valid})'),
        }
    thr_pos = float(np.quantile(noise_score[pos_valid], 1.0 - float(noise_frac)))
    thr_neg = float(np.quantile(noise_score[neg_valid], 1.0 - float(noise_frac)))
    # Keep samples NOT in the top noise_frac of their own class.
    # Samples without OOF prediction (valid_mask=False) default to kept.
    keep_pos = pos_valid & (noise_score < thr_pos)
    keep_neg = neg_valid & (noise_score < thr_neg)
    keep_mask = keep_pos | keep_neg | (~valid_mask)
    threshold = float(max(thr_pos, thr_neg))  # for diagnostics

    n_kept = int(keep_mask.sum())
    n_removed = n - n_kept
    pos_total = int(y_arr.sum())
    pos_kept = int((y_arr[keep_mask] > 0.5).sum())
    pos_removed = pos_total - pos_kept

    if n_removed == 0:
        return np.ones(n, dtype=bool), {
            'applied': False,
            'reason': 'no samples flagged at this noise_frac',
        }
    if pos_total > 0 and pos_kept < 0.5 * pos_total:
        return np.ones(n, dtype=bool), {
            'applied': False,
            'reason': (f'cleaning would remove >50% positives '
                       f'({pos_removed}/{pos_total}) — abort'),
        }

    avg_oof_auc = float(np.mean(fold_aucs)) if fold_aucs else float('nan')
    info = {
        'applied': True,
        'n_total': int(n),
        'n_kept': int(n_kept),
        'n_removed': int(n_removed),
        'noise_frac_target': float(noise_frac),
        'noise_threshold': float(threshold),
        'noise_threshold_pos_class': float(thr_pos),
        'noise_threshold_neg_class': float(thr_neg),
        'pos_kept': int(pos_kept),
        'pos_removed': int(pos_removed),
        'pos_total': int(pos_total),
        'pos_rate_before': float(pos_rate),
        'pos_rate_after': float(y_arr[keep_mask].mean()) if n_kept > 0 else 0.0,
        'avg_oof_auc': avg_oof_auc,
        'n_folds': int(n_folds),
        'n_valid_oof': int(valid_mask.sum()),
        'pruning': 'class_conditional',
    }

    if verbose:
        print(f'  Confident-learning label cleaning (THIS ATTEMPT #332):')
        print(f'    K={int(n_folds)} chronological folds, OOF AUC '
              f'(avg) = {avg_oof_auc:.3f} (>0.55 = meaningful XGB signal)')
        print(f'    Class-conditional pruning at noise_frac={noise_frac:.2f} '
              f'(top {100*float(noise_frac):.0f}% per-class P(wrong class)):')
        print(f'      Positive class: removed {pos_removed}/{pos_total} '
              f'(thr={thr_pos:.3f})')
        print(f'      Negative class: removed {n_removed - pos_removed}/'
              f'{n - pos_total} (thr={thr_neg:.3f})')
        print(f'    Total removed: {n_removed}/{n} ({100*n_removed/n:.1f}%)')
        print(f'    Pos rate: {pos_rate:.3f} -> '
              f'{info["pos_rate_after"]:.3f}')

    return keep_mask, info


def _compute_covariate_shift_weights(X_train_scaled, X_val_scaled, verbose=False):
    """Density-ratio estimation for covariate-shift correction.

    Trains a shallow XGBoost domain classifier on aggregated sequence
    features (concatenated last/mean/std/last-mean) to distinguish train
    from val samples. Per-sample weights for training are computed as
    w(x) = p(val|x) / p(train|x) via the odds ratio, then clipped to
    [0.1, 10.0] and normalized to mean 1.

    Theory: under covariate shift (p_train(x) ≠ p_test(x) but p(y|x) is
    shared), importance-weighted ERM with w(x) = p_test(x)/p_train(x)
    produces the minimum-variance unbiased risk estimator for test
    performance. We use val as a proxy for test (forward-in-time window
    inside the train period). The odds ratio from a balanced domain
    classifier estimates this ratio up to a constant that cancels after
    mean-normalization.

    Clipping bounds: 0.1/10.0 keeps gradient variance finite without
    zeroing out any sample. Mean-normalization preserves total training
    loss magnitude so other hyperparameters (lr, etc.) don't need
    rescaling.

    Returns (weights, info_dict). On XGBoost failure returns
    uniform ones + empty info — training degrades to the un-corrected
    Entry 197 recipe rather than erroring out.
    """
    try:
        import xgboost as xgb
    except ImportError:
        if verbose:
            print('  Covariate-shift: XGBoost unavailable — uniform weights')
        return np.ones(len(X_train_scaled), dtype=np.float32), {'applied': False,
                                                                 'reason': 'no xgboost'}

    X_tr_agg = _aggregate_sequence_np(X_train_scaled)
    X_val_agg = _aggregate_sequence_np(X_val_scaled)

    X_combined = np.concatenate([X_tr_agg, X_val_agg], axis=0)
    y_domain = np.concatenate([
        np.zeros(len(X_tr_agg), dtype=np.float32),
        np.ones(len(X_val_agg), dtype=np.float32),
    ])

    # Cap compute by subsampling if combined set is very large. 60K rows
    # of a shallow-depth XGBoost takes ~10 s on CPU — well within budget.
    n_total = len(X_combined)
    if n_total > 60000:
        rng = np.random.RandomState(0)
        sub_idx = rng.choice(n_total, size=60000, replace=False)
        X_fit = X_combined[sub_idx]
        y_fit = y_domain[sub_idx]
    else:
        X_fit = X_combined
        y_fit = y_domain

    try:
        clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=20,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=2,
            tree_method='hist',
            verbosity=0,
        )
        clf.fit(X_fit, y_fit)
    except Exception as exc:
        if verbose:
            print(f'  Covariate-shift: domain classifier fit failed ({exc}) — uniform')
        return np.ones(len(X_train_scaled), dtype=np.float32), {'applied': False,
                                                                 'reason': f'fit failed: {exc}'}

    p_val = clf.predict_proba(X_tr_agg)[:, 1].astype(np.float32)
    p_val = np.clip(p_val, 0.05, 0.95)

    # w(x) = p(val|x) / p(train|x) up to a constant from Bayes' rule
    # — the constant drops out after mean-normalization below.
    weights = p_val / (1.0 - p_val)
    weights = np.clip(weights, 0.1, 10.0)
    weights = weights * (len(weights) / max(weights.sum(), 1e-6))
    weights = weights.astype(np.float32)

    info = {
        'applied': True,
        'min': float(weights.min()),
        'median': float(np.median(weights)),
        'max': float(weights.max()),
        'std': float(weights.std()),
    }
    try:
        from sklearn.metrics import roc_auc_score
        info['domain_auc'] = float(roc_auc_score(
            y_fit, clf.predict_proba(X_fit)[:, 1]))
    except Exception:
        info['domain_auc'] = float('nan')

    if verbose:
        # Concentration of weight mass in top-decile train samples: how much
        # the correction actually redistributes. A uniform distribution gives
        # 0.10; >= 0.25 means real shift is being corrected.
        sorted_w = np.sort(weights)[::-1]
        top_decile_share = float(
            sorted_w[:max(1, len(weights) // 10)].sum() / max(weights.sum(), 1e-6))
        info['top_decile_share'] = top_decile_share
        print(f'  Covariate-shift weights: min={info["min"]:.3f} '
              f'median={info["median"]:.3f} max={info["max"]:.3f} '
              f'std={info["std"]:.3f}')
        print(f'    Domain classifier AUC={info["domain_auc"]:.3f} '
              f'(>0.55 indicates meaningful distribution shift)')
        print(f'    Top-decile share of weight mass={top_decile_share:.3f} '
              f'(0.10=uniform, >0.25=meaningful correction)')

    return weights, info


def _window_warp_batch(X_np, p=0.5, window_frac=0.2, scale_low=0.5,
                        scale_high=2.0, rng=None):
    """Per-sequence window warping augmentation (THIS ATTEMPT #334).

    For each sequence in the batch, with probability p:
      1. Pick a random contiguous window of size window_frac * T.
      2. Resample that window at a different rate (scale ~ U[scale_low, scale_high]):
         scale > 1 stretches the window (slows the dynamics down);
         scale < 1 compresses it (speeds the dynamics up).
      3. Splice [pre, warped_window, post] and resample the whole sequence
         back to length T via linear interpolation along the time axis.

    Theoretical basis: Um et al. 2017, "Data Augmentation of Wearable Sensor
    Data for Parkinson's Disease Monitoring using Convolutional Neural
    Networks." Window warping was the MOST EFFECTIVE single-source aug
    for time-series classification in their study (better than jitter,
    scaling, or magnitude-warping alone).

    Why window warping attacks the 37.5%-WR plateau / cross-split-WR variance
    -------------------------------------------------------------------------
    Stocks move at variable paces across regimes. A momentum signature that
    plays out over 4 days in a fast/volatile regime can stretch to 7 days
    in a slow regime. The LSTM trained on raw sequences sees only ONE
    specific timing per pattern. Window warping exposes the model to
    plausible alternate-pace versions of the SAME pattern, forcing it to
    learn pace-invariant features. Pace-invariance is exactly what
    transfers across the 7 WF splits (each with different volatility and
    momentum dynamics) — and is structurally orthogonal to:
      * jitter (per-timestep IID noise — already done by INPUT_NOISE_STD)
      * mixup (between-sample interpolation)
      * label-side mechanisms (smoothing, soft labels, etc.)

    Why this is NOT one of the already-rejected mechanisms
    ------------------------------------------------------
      * Mixup (Entry 197 baseline, ACTIVE): convex combination of TWO
        sequences. Window warping operates on ONE sequence, distorting
        its temporal structure. They compose: window-warp first, then
        mixup the warped sequences.
      * INPUT_NOISE_STD inside LSTMModel (always active): IID Gaussian
        per-timestep. Window warping is structured, smooth, and
        temporally coherent.
      * Manifold mixup (#328, REJECTED): mix at hidden layer of two
        sequences. Same multi-sample mechanism, different layer.
      * Curriculum / boundary-band exclusion (#304, #317, REJECTED):
        sample FILTERING by PnL. Window warping keeps every sample.
      * SupCon (#319, REJECTED): contrastive pretraining stage.
        Window warping runs DURING supervised training as on-the-fly
        per-batch augmentation.
      * Time-series-specific augmentations are listed as UNTRIED option
        (6) in the condensed lessons:
          "Time-series augmentations (jitter, scaling, time-warping,
           window-warping) instead of plain Gaussian noise."

    Why this is SAFE
    ----------------
      - Saved model has IDENTICAL signature to Entry 197. Gate loader,
        live trader, scaler are all unchanged. Production inference is
        bit-identical — no warping at inference time.
      - CURATED_FEATURES untouched: the warping operates on already-
        scaled features in (B, T, F) shape. Feature semantics, ordering,
        and count are all preserved.
      - lstm_model.py untouched.
      - Compute cost: per-sample numpy interp on a (T, F) array. With
        T=20, F=19, 256-batch, ~5ms per batch on CPU. << 1% overhead vs
        the LSTM forward pass.
      - Degrades gracefully: when use_window_warp=False (sweep ablation)
        the path is bit-identical to the (now-label-cleaning-OFF)
        Entry 197 baseline. Per-sequence prob p=0.5 means HALF the
        sequences are unchanged, so the unwarped distribution is
        always represented.
      - Bounded distortion: with default scale_range=[0.5, 2.0] and
        window_frac=0.2 (4 of 20 timesteps), the warped sequence has
        between 90% and 120% of its original length before final
        re-interpolation back to T=20. The max temporal displacement
        of any feature value is ~3 timesteps — well within the LSTM's
        receptive field, so it learns "the same pattern at slightly
        different pace" rather than "a totally different pattern".

    Args:
      X_np: (B, T, F) numpy array of (already-scaled) input sequences.
      p: probability of warping each sequence (others pass through).
      window_frac: window size as fraction of T.
      scale_low/scale_high: uniform speedup factor range.
      rng: numpy RandomState for determinism.

    Returns: (B, T, F) numpy array, same shape and dtype as input.
    """
    if rng is None:
        rng = np.random.RandomState(20260429)
    B, T, F = X_np.shape
    if T < 4:
        return X_np  # too short to warp meaningfully
    out = X_np.copy()
    ws = max(2, int(round(T * float(window_frac))))
    if ws >= T - 1:
        return X_np
    old_t_full = np.linspace(0.0, 1.0, T)
    for i in range(B):
        if rng.random() > float(p):
            continue
        start = int(rng.randint(0, T - ws))
        end = start + ws
        scale = float(rng.uniform(float(scale_low), float(scale_high)))
        new_ws = max(2, int(round(ws * scale)))
        old_w = np.linspace(0.0, 1.0, ws)
        new_w = np.linspace(0.0, 1.0, new_ws)
        warped = np.empty((new_ws, F), dtype=X_np.dtype)
        for f in range(F):
            warped[:, f] = np.interp(new_w, old_w, X_np[i, start:end, f])
        full = np.concatenate([X_np[i, :start], warped, X_np[i, end:]], axis=0)
        if full.shape[0] != T:
            old_full = np.linspace(0.0, 1.0, full.shape[0])
            resampled = np.empty((T, F), dtype=X_np.dtype)
            for f in range(F):
                resampled[:, f] = np.interp(old_t_full, old_full, full[:, f])
            out[i] = resampled
        else:
            out[i] = full
    return out


def _time_mask_batch(X_np, p=0.5, mask_size_frac=0.15, n_masks=2, rng=None):
    """SpecAugment-style time masking augmentation (THIS ATTEMPT #335).

    Park, Chan, Zhang, Chiu, Zoph, Cubuk, Le (2019), "SpecAugment: A Simple
    Data Augmentation Method for Automatic Speech Recognition", Interspeech.
    Proven major generalization win for audio (Park et al. cut WER 30-50%
    on LibriSpeech), translates naturally to time-series classification.

    For each sequence in the batch, with probability p, apply `n_masks`
    independently sampled contiguous time-block masks. Each mask zeros out
    a random contiguous slice of size ~U[1, mask_size_frac * T] across ALL
    features. Zero is the per-feature MEDIAN after RobustScaler centering,
    so masking simulates "no information at this timestep — assume typical
    market" rather than "extreme outlier here".

    Why time masking attacks the 37.5%-WR plateau / "epoch-0-best" pathology
    -----------------------------------------------------------------------
    The dominant 334-attempt failure mode is val_loss diverging by epoch
    0-4, attributed to temporal feature-distribution shift: the LSTM
    latches onto SPECIFIC TIMESTEP POSITIONS in train (e.g., always uses
    day -1 momentum spike) that happen to be predictive in the train era
    but mean different things in the test era.

    Time masking forces the LSTM to predict CORRECTLY even when those
    specific timesteps are missing — the model must learn features that
    are aggregable across the WHOLE sequence, not concentrated on one
    time position. This produces representations that are invariant to
    "where in the recent window the signal appears", which is exactly
    the invariance needed to transfer across regimes where the same
    pattern plays out at different paces or positions.

    Why this is NOT one of the already-rejected mechanisms
    ------------------------------------------------------
      * Window warping (#334, REJECTED): preserves ALL temporal info,
        distorts only PACING. Time masking REMOVES info entirely at
        random positions. Different mechanism: warping = "same signal
        at different speed"; masking = "predict robustly without
        portion of recent context". Empirically window warping just
        rejected; masking is a complementary intervention that should
        not inherit warping's failure mode because the optimization
        constraint is qualitatively different.
      * Mixup (Entry 197 baseline, ACTIVE): convex combo of TWO
        sequences. Time masking operates on ONE sequence. They COMPOSE
        cleanly — mask first, then mixup the masked sequences.
      * Manifold Mixup (#328, REJECTED): mix at hidden layer. Different
        layer + multi-sample mechanism.
      * INPUT_NOISE_STD inside LSTMModel (always active): IID Gaussian
        per ELEMENT. Time masking zeros CONTIGUOUS BLOCKS of TIMESTEPS
        across all features — structured, sparse, large-magnitude
        perturbation vs IID small-magnitude noise.
      * input_dropout (in LSTMModel, baseline ON): per-ELEMENT
        Bernoulli zeroing. Time masking is BLOCK-level on the TIME
        axis. Critically different statistical structure: dropout's
        independence assumption breaks when correlated information
        across timesteps is dropped together; time masking REQUIRES
        the model to use other timesteps to reconstruct the missing
        context.
      * Curriculum / boundary-band exclusion (#304, #317, REJECTED):
        sample FILTERING by PnL. Time masking keeps every sample but
        perturbs its INPUT.
      * Confident-learning label cleaning (#332, REJECTED): removes
        likely-mislabeled samples. Time masking keeps all samples.
      * SS-pretrain / SupCon (#298, #319, REJECTED): pre-training
        stage. Time masking runs DURING supervised training.
      * Mean Teacher (#300, REJECTED): EMA-teacher consistency. No
        teacher copy here — single-pass single-model augmentation.
      * Label smoothing / soft labels (#47, #72, #321, REJECTED):
        label-side mechanisms. Time masking is INPUT-side and
        completely label-independent.
      * NEVER tried in this project's 334-attempt history (verified
        by grep on the trainer source for 'time_mask', 'spec_aug',
        'specaugment', 'time_mask_batch' — all returned no matches).

    Why this is SAFE
    ----------------
      - Saved model has IDENTICAL signature to Entry 197. Gate loader,
        live trader, scaler are all unchanged. Production inference
        does NOT mask (eval mode, single forward pass on un-masked
        input). Only training-time data perturbation.
      - CURATED_FEATURES untouched. Mask operates on already-scaled
        (B, T, F) arrays; feature semantics, ordering, count preserved.
      - lstm_model.py untouched.
      - Compute cost: 2 random integer draws + 1 numpy slice assign
        per masked sequence per mask. With B=256, T=20, n_masks=2,
        p=0.5, ~256 ops/batch — negligible overhead vs the LSTM
        forward pass.
      - Bounded distortion: with default mask_size_frac=0.15, n_masks=2,
        worst case zeroes 30% of timesteps (still 14 of 20 remaining),
        average case ~15-20%. Well within the LSTM's information-
        recovery capability — empirically NLP / audio models tolerate
        much higher mask ratios.
      - Per-sequence prob p=0.5: half of each batch is unmasked, so
        the un-perturbed distribution is always represented. The
        optimizer sees both clean and masked versions of the train
        distribution per step — keeps gradient direction stable while
        adding regularization pressure.
      - Degrades gracefully:
          * use_time_mask=False (sweep ablation): code path is
            bit-identical to the (now-window-warp-OFF) Entry 197
            baseline.
          * mask_size_frac=0 or n_masks=0: no-op (returns input
            unchanged).
          * T < 4: short sequences pass through unchanged (no
            meaningful mask possible).
      - Active on every training path (BCE / GCE / Brier / PnL /
        distill) because masking happens at batch-prep time, before
        any path-specific code reads X_batch. Independent of which
        loss family is active. Composes cleanly with mixup: masked
        sequences feed into the standard mixup pair-construction.

    Why two masks (n_masks=2) rather than one
    -----------------------------------------
    Park et al. (2019) found that TWO time masks per sequence outperformed
    one for speech recognition because it forces the model to recover
    from MULTIPLE missing windows simultaneously, which more closely
    matches inference-time conditions where any feature can be
    momentarily uninformative. With T=20 and mask_size_frac=0.15, two
    masks of size up to 3 timesteps means the model usually still has
    14+ timesteps of context — plenty for a 1-layer 48-hidden LSTM.

    Args:
      X_np: (B, T, F) numpy array of (already-scaled) input sequences.
      p: probability of masking each sequence (others pass through).
      mask_size_frac: max mask size as a fraction of T. Each mask is
                      sampled at int(U[1, mask_size_frac * T]).
      n_masks: number of independently sampled masks per sequence.
      rng: numpy RandomState for determinism.

    Returns: (B, T, F) numpy array, same shape and dtype as input.
    """
    if rng is None:
        rng = np.random.RandomState(20260503)
    B, T, F = X_np.shape
    if T < 4 or int(n_masks) <= 0:
        return X_np
    max_mask_size = max(1, int(round(T * float(mask_size_frac))))
    if max_mask_size <= 0:
        return X_np
    out = X_np.copy()
    for i in range(B):
        if rng.random() > float(p):
            continue
        for _ in range(int(n_masks)):
            # Per-mask size sampled uniformly in [1, max_mask_size] —
            # variable size mirrors SpecAugment's F/T parameters where
            # each application chooses a fresh random width.
            ms = int(rng.randint(1, max_mask_size + 1))
            if ms >= T:
                continue
            start = int(rng.randint(0, T - ms + 1))
            # Zero-fill = per-feature median after RobustScaler centering.
            # Encodes "no informative deviation at this timestep" rather
            # than the model's INPUT_NOISE_STD Gaussian which adds
            # spurious extreme values.
            out[i, start:start + ms, :] = 0.0
    return out


def _compute_time_decay_weights(dates_train, decay=1.0, verbose=False):
    """Exponential CHRONOLOGICAL time-decay sample weighting (THIS ATTEMPT #316).

    Per-sample weight w_i = exp(decay * t_i_norm) where t_i_norm in [0, 1] is
    the normalized chronological position of sample i within
    [date_min, date_max]. With decay=1.0, the most recent train sample
    weighs e ~= 2.72x the oldest. Mean-normalized to 1 so total loss
    magnitude is preserved across runs.

    Why time-decay attacks the 37.5%-WR plateau / "epoch-0-best" pathology
    ---------------------------------------------------------------------
    The dominant failure mode across 315 attempts is "best epoch is 0-4
    with val_loss diverging immediately" — the condensed lessons attribute
    this to a temporal feature-distribution shift, not architectural
    overfitting. The model spreads gradient mass UNIFORMLY across train
    samples; samples from older market regimes dominate by count and pull
    weights toward an "average regime" that is not the regime where test
    actually lives.

    Time-decay reweighting tells the optimizer:
      "samples chronologically closest to the train cut-off matter MORE,
       because they are the ones whose distribution most closely matches
       the test window that immediately follows."
    This is parameter-free, monotone, smooth, and label-independent — the
    weight depends only on each sample's calendar position.

    Why this is NOT one of the rejected mechanisms
    ----------------------------------------------
      * Density-ratio covariate-shift (#295, REJECTED): learned weights
        from FEATURE distributions via an XGBoost domain classifier.
        Saturated immediately (domain AUC ~ 1.0 -> all weights pinned to
        clip floor, effectively uniform). Time-decay uses CHRONOLOGY
        directly — no learned model, no saturation, weights are bounded
        by exp(decay) by construction.
      * Curriculum learning (#304, REJECTED): hard-thresholded sample
        FILTERING by |PnL|. Time-decay is SOFT continuous reweighting —
        every sample contributes; only the gradient share shifts.
      * Mean Teacher (#300, REJECTED): EMA-teacher consistency loss —
        a different mechanism entirely; does not reweight samples.
      * SS pretraining (#298, REJECTED): two-stage backbone pretext task.
        Time-decay is single-stage and orthogonal.

    Composition with existing machinery
    -----------------------------------
    Returns a (N,) numpy array intended to be MULTIPLIED into the existing
    `shift_weights_np` slot. That slot already feeds the BCE per-sample
    loss as `total_weight = class_weight * shift_mix`, so time-decay
    weights compose multiplicatively with class rebalancing (pos_weight)
    without touching the optimizer, the loss family, or the model.

    Args:
      dates_train: array-like, length N, aligned 1:1 with X_train. Items
                   may be pandas Timestamps, datetime, np.datetime64, or
                   ISO strings. Sorting is NOT required — weights depend
                   only on each sample's absolute calendar position.
      decay: float >= 0. Recommended range [0.5, 2.0]; default 1.0 gives
             ~2.7x newest:oldest spread. decay=0 is a uniform no-op.
      verbose: print summary stats.

    Returns (weights, info_dict). On error returns uniform ones + a
    failure reason in info — training degrades to the un-weighted
    baseline rather than erroring out.
    """
    dates = np.asarray(dates_train)
    if len(dates) == 0:
        return np.ones(0, dtype=np.float32), {'applied': False, 'reason': 'empty dates'}
    try:
        t = np.asarray(pd.to_datetime(dates).astype(np.int64), dtype=np.float64)
    except Exception:
        # Fallback: rank by sort order — preserves chronological monotonicity
        # without requiring datetime parsing to succeed.
        order = np.argsort(np.argsort(dates))
        t = order.astype(np.float64)
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max <= t_min:
        return (np.ones(len(t), dtype=np.float32),
                {'applied': False, 'reason': 'degenerate dates (single timestamp)'})
    t_norm = (t - t_min) / (t_max - t_min)
    weights = np.exp(float(decay) * t_norm)
    weights = weights * (len(weights) / max(weights.sum(), 1e-9))
    weights = weights.astype(np.float32)

    info = {
        'applied': True,
        'decay': float(decay),
        'min': float(weights.min()),
        'median': float(np.median(weights)),
        'max': float(weights.max()),
        'std': float(weights.std()),
        'mean': float(weights.mean()),
        'newest_to_oldest_ratio': float(np.exp(float(decay))),
        'n_samples': int(len(weights)),
    }
    if verbose:
        print(f'  Time-decay weights: decay={decay}, '
              f'min={info["min"]:.3f} median={info["median"]:.3f} '
              f'max={info["max"]:.3f} mean={info["mean"]:.3f}')
        print(f'    Newest:oldest weight ratio = {info["newest_to_oldest_ratio"]:.2f} '
              f'(parameter-free, derived from chronology)')
    return weights, info


def _compute_stationarity_mask(X_train_scaled, X_val_scaled, features,
                                alpha=10.0, floor=0.3, verbose=False):
    """Per-feature KS-distance stationarity mask (THIS ATTEMPT #326).

    For each input feature i, compute the Kolmogorov–Smirnov 2-sample
    distance KS_i between its TRAIN and VAL distributions (on the
    decision-point timestep, i.e. the last step of each sequence).
    Features whose distribution shifts substantially from train→val
    are likely also non-stationary across val→test, so the model
    cannot rely on them to generalize.

    Mask formula:
        m_i = max(floor, exp(-alpha * KS_i^2))

    Defaults (alpha=10.0, floor=0.3) give:
        KS=0.00 -> m=1.000  (full weight, no shift)
        KS=0.10 -> m=0.905  (mild shift, ~9% downweight)
        KS=0.20 -> m=0.670  (moderate, 33% downweight)
        KS=0.30 -> m=0.407  (significant, 59% downweight)
        KS=0.50 -> m=0.300  (clamped at floor)

    The mask is applied multiplicatively to the (already-scaled) input
    features (broadcast over the time dimension), and the scaler's
    `scale_` is divided by the mask in place so that downstream
    `scaler.transform()` calls — at the WF gate, in production, and
    during recalibration — automatically reproduce the same masked
    feature space. The saved scaler is still a valid RobustScaler
    object (identical pickle format); only its `scale_` array differs
    from what the un-masked baseline would produce.

    Why this attacks the 37.5%-WR plateau / "epoch-0-best" pathology
    ----------------------------------------------------------------
    The dominant 325-attempt failure mode is "best epoch is 0–4 with
    val_loss diverging immediately" — the condensed lessons explicitly
    attribute this to "a feature-distribution temporal shift, not LSTM
    overfitting" (persistent problem #1) and list "Adding a feature-
    stationarity filter (drop any feature whose train/val KS-distance
    exceeds a threshold) automatically rather than hand-curating" as
    untried option (5).

    The KS-mask intervenes at the most upstream layer possible: the
    INPUT itself. Features that drift across the train/val boundary
    are the very features that drive the val-loss collapse — they
    embed shortcut signals that fit train but mismatch val. By
    multiplicatively shrinking those features at the model's input,
    we shift the optimizer's gradient pressure away from them and
    toward features whose distributions are stable.

    Critically, this is computed PER WALK-FORWARD SPLIT against that
    split's own train/val pair. Different splits have different
    distribution-drift profiles (a feature that's stable in one regime
    may shift in another), so the mask self-adapts per split rather
    than imposing a single global feature ranking.

    Why this is NOT one of the already-rejected mechanisms
    ------------------------------------------------------
      * #295 covariate-shift density-ratio (REJECTED): SAMPLE-level
        weighting via a learned XGBoost domain classifier. Saturated
        with domain AUC ≈ 1.0 → all weights pinned to clip floor →
        effectively uniform. Stationarity mask is FEATURE-level,
        not sample-level: it does not learn weights, it computes them
        from a closed-form KS test, and it caps the per-feature
        downweight at `floor` (0.3) so no feature is ever zeroed out.
      * #296 DANN (REJECTED): adversarial gradient-reversal alignment
        of train/val FEATURES via a domain classifier added to the
        forward pass. Trains a parametric adversary; failure mode was
        the same domain-AUC saturation. Stationarity mask has zero
        learned parameters and zero adversarial training; it's a
        deterministic rescaling of inputs.
      * #298 SS-pretrain (REJECTED): unsupervised next-step pretext
        task on the LSTM backbone. Pre-training stage. Stationarity
        mask is INPUT-LAYER and runs alongside training, not before.
      * #316 chronological time-decay (REJECTED): sample-level
        weighting by calendar position. Stationarity mask is
        feature-level, no sample weights; orthogonal mechanism.
      * #317 boundary-band sample exclusion (REJECTED): hard SAMPLE
        filtering by |pnl - 4%|. Throws out samples. Stationarity
        mask keeps every sample but recalibrates per-feature weight.
      * #318 recent-window decision-head FT (REJECTED): two-stage
        with frozen backbone. Stationarity mask is single-stage,
        operates on inputs not on FC weights.
      * #319 SupCon (REJECTED): cross-symbol contrastive pretraining.
        Stage-0 representation learning. Stationarity mask runs
        in-place during regular supervised training.
      * #320 IRM (REJECTED): per-environment penalty on the gradient
        of a dummy-scaled loss. Optimizer-level mechanism. Stationarity
        mask is data-level, no penalty term, no environments.
      * Hand-curated feature subsets (#54, #60, #189, etc.): a HUMAN
        chose which features to drop. Stationarity mask is fully
        AUTOMATIC and per-split — exactly the recommendation from
        the condensed lessons ("automatically rather than hand-
        curating").
      * NOT in the project's 325-attempt history (verified by
        exhaustive grep on the trainer source for "KS_2", "ks_2samp",
        "stationarity", "ks_distance", and similar terms).

    Why this is SAFE
    ----------------
      - CURATED_FEATURES is unchanged: feature_eng.py still produces
        the same 19-feature input layout. The scaler still expects 19
        features. No alignment break.
      - lstm_model.py is unchanged: the LSTMModel input_size, layer
        topology, and forward path are identical. Production inference
        sees the same model class.
      - Saved model has BIT-IDENTICAL signature to Entry 197: state_dict
        keys, shapes, dtype, attribute set are all unchanged.
      - Saved scaler is BIT-IDENTICAL in format (pickled RobustScaler);
        only the numerical values of `scale_` differ. Loading and
        `.transform()` work unchanged because `scale_` is just a
        per-feature divisor.
      - Inference behavior is consistent end-to-end: training uses
        masked features → saved scaler reproduces those masked features
        on test/live data → model evaluates on the same distribution
        it trained on. No train/inference skew.
      - Compute cost: 19 KS tests × ≤ N samples each. Negligible
        (<< 1 second per WF split) compared to LSTM training.
      - Degrades gracefully:
          * When all features stationary (KS≈0): m≈1, mask is a no-op.
          * When use_stationarity_mask=False: mask not computed,
            scaler unmodified, training is bit-identical to the
            (now r_drop-OFF) Entry 197 baseline.
      - Floor of 0.3 prevents any feature from being effectively
        removed; the model can still recover some signal from
        non-stationary features if they truly help.
      - Per-WF-split: each split computes its own mask. A feature
        flagged in split 4 may not be flagged in split 2 — the mask
        captures regime-specific drift.

    Args:
      X_train_scaled: (N_tr, T, F) numpy array, AFTER RobustScaler.
      X_val_scaled:   (N_val, T, F) numpy array, AFTER RobustScaler.
      features:       length-F list of feature names (for diagnostics).
      alpha:          steepness of mask falloff. 10.0 gives:
                        KS=0.10 → 0.905, KS=0.30 → 0.407.
      floor:          minimum mask value (default 0.3 = 70% downweight cap).
      verbose:        print mask diagnostics.

    Returns (mask, info_dict). Mask is a (F,) float32 array. On any
    error (e.g. scipy unavailable, degenerate distributions), returns
    an all-ones mask so training falls through to the un-masked path.
    """
    F = X_train_scaled.shape[2]
    try:
        from scipy import stats as _scstats
    except Exception as exc:
        return (np.ones(F, dtype=np.float32),
                {'applied': False, 'reason': f'scipy unavailable: {exc}'})

    # Use the decision-point timestep (last step of each sequence) as the
    # per-sample observation. This matches what the LSTM's final hidden
    # state is conditioned on, and gives one clean observation per sample
    # rather than mixing within-sequence variation with between-period
    # variation.
    train_last = X_train_scaled[:, -1, :]
    val_last = X_val_scaled[:, -1, :]

    if len(train_last) < 50 or len(val_last) < 50:
        return (np.ones(F, dtype=np.float32),
                {'applied': False,
                 'reason': f'samples too few (train={len(train_last)}, val={len(val_last)})'})

    ks_distances = np.zeros(F, dtype=np.float64)
    for i in range(F):
        try:
            d, _ = _scstats.ks_2samp(train_last[:, i], val_last[:, i])
            ks_distances[i] = float(d)
        except Exception:
            ks_distances[i] = 0.0

    mask = np.exp(-float(alpha) * ks_distances ** 2)
    mask = np.maximum(mask, float(floor)).astype(np.float32)

    info = {
        'applied': True,
        'alpha': float(alpha),
        'floor': float(floor),
        'mean_mask': float(mask.mean()),
        'median_mask': float(np.median(mask)),
        'min_mask': float(mask.min()),
        'min_mask_feature': str(features[int(np.argmin(mask))]),
        'max_ks': float(ks_distances.max()),
        'max_ks_feature': str(features[int(np.argmax(ks_distances))]),
        'n_features_below_0.7': int((mask < 0.7).sum()),
        'n_features_at_floor': int((mask <= float(floor) + 1e-6).sum()),
        'feature_ks': {str(features[i]): float(ks_distances[i]) for i in range(F)},
        'feature_mask': {str(features[i]): float(mask[i]) for i in range(F)},
    }

    if verbose:
        print(f'  Stationarity mask: alpha={alpha}, floor={floor}')
        print(f'    Mean mask weight: {info["mean_mask"]:.3f}, '
              f'median: {info["median_mask"]:.3f}, '
              f'min: {info["min_mask"]:.3f}')
        print(f'    Most non-stationary feature: '
              f'{info["max_ks_feature"]} '
              f'(KS={info["max_ks"]:.3f}, mask={mask[int(np.argmax(ks_distances))]:.3f})')
        # Top-5 most-downweighted features for diagnostics
        sorted_idx = np.argsort(mask)
        print(f'    Top-5 downweighted features (per this split):')
        for k in range(min(5, F)):
            idx = int(sorted_idx[k])
            print(f'      {features[idx]:<32s}  KS={ks_distances[idx]:.3f}, '
                  f'mask={mask[idx]:.3f}')
        print(f'    Features with mask < 0.7 (>30% downweight): '
              f'{info["n_features_below_0.7"]}/{F}')

    return mask, info


def _pnl_smooth_labels(pnl, threshold=MIN_PROFIT_PCT, scale=0.02,
                       clip_eps=1e-4):
    """Compute PnL-distance-based soft labels (THIS ATTEMPT #321).

    For each sample with realized PnL p, returns
        y_soft = sigmoid((p - threshold) / scale)
    bounded into [clip_eps, 1 - clip_eps] for numerical stability of BCE.

    Properties (with default threshold=0.04, scale=0.02):
      - At p = threshold (exactly the fence)        : y_soft = 0.500
      - At p = threshold + scale  (one e-fold above): y_soft = 0.731
      - At p = threshold + 2*scale                  : y_soft = 0.881
      - At p = threshold + 5*scale                  : y_soft = 0.993
      - At p = threshold - scale                    : y_soft = 0.269
      - At p = -0.03 (clear stop-loss)              : y_soft = 0.029

    Why sigmoid (not linear interpolation, not Gaussian)
    ----------------------------------------------------
      - Sigmoid is bounded in (0, 1), so the loss is well-defined and the
        gradient never explodes — both BCE and downstream calibration
        assume probabilistic targets.
      - Sigmoid is monotone in PnL: ranking by realized magnitude is
        preserved exactly. A bigger winner has a higher target than a
        smaller winner; a bigger loser has a smaller target than a
        smaller loser. No information about magnitude order is discarded.
      - Sigmoid SATURATES on both tails. Going from +15% to +20% PnL
        only moves y_soft from 0.9999 to 1.0000 — the model learns that
        ANY clear winner is a 1, not that bigger winners are exponentially
        better. This avoids the "regress on outlier PnL" trap that pure
        magnitude-weighted losses (#199, REJECTED) fell into.
      - The sigmoid's INFLECTION is precisely AT the binary fence
        (pnl = MIN_PROFIT_PCT), where the binary label is most
        ambiguous. This is the location with maximum information value
        for relabeling.

    Why scale = 0.02 (= half of MIN_PROFIT_PCT)
    --------------------------------------------
      - At |p - threshold| > 3*scale = 6%, y_soft is within 0.05 of 0
        or 1 — the smoothing is effectively transparent for clear
        winners (PnL > +10%) and clear losers (PnL < -2%).
      - At |p - threshold| <= scale = 2%, y_soft sits in (0.27, 0.73) —
        the entire fence-line band [+2%, +6%] is moderately uncertain.
      - The scale is wide enough to actually attenuate the fence-line
        gradient (not a tiny epsilon that doesn't move it) but narrow
        enough that clear evidence still produces sharp targets.

    Returns a numpy float32 array of shape pnl.shape.
    """
    pnl_arr = np.asarray(pnl, dtype=np.float64)
    z = (pnl_arr - float(threshold)) / max(float(scale), 1e-9)
    y_soft = 1.0 / (1.0 + np.exp(-z))
    y_soft = np.clip(y_soft, clip_eps, 1.0 - clip_eps)
    return y_soft.astype(np.float32)


def _finetune_on_recent(model, X_train_scaled, y_train, dates_train,
                        n_recent_days=30, epochs=8, lr=2e-4,
                        batch_size=256, pos_weight_cap=10.0, verbose=False):
    """Fine-tune ONLY the FC head on the most-recent N days of train (#318).

    Stage 2 of two-stage training. Stage 1 (the main loop + SWA) sees the
    full train period; Stage 2 sees ONLY the last n_recent_days. The LSTM
    backbone is frozen via requires_grad_(False), so the encoder's learned
    features stay fixed; only the small FC head (~hidden_size+1 params)
    updates against a class-balanced BCE loss on the recent slice.

    Why recent-window FT attacks the 37.5%-WR plateau / "epoch-0-best"
    pathology
    ------------------------------------------------------------------
    The dominant failure mode is temporal feature-distribution shift
    between train and test. The 30 days closest to the train cut-off
    have a distribution that is closest to the test window (which begins
    immediately after). Re-fitting the decision boundary on that
    near-test slice — without disturbing the encoder learned over the
    full train history — adapts the classifier to the regime that test
    will actually inhabit. Time-decay (#316, REJECTED) tried to do this
    via SOFT re-weighting in a single pass; recent FT does it via HARD
    staging (full-train then recent-only).

    Why this is NOT one of the rejected mechanisms
    ----------------------------------------------
      * Time-decay sample weighting (#316, REJECTED): re-weighted ALL
        train samples in a single pass. Recent FT is a SECOND STAGE
        that sees ONLY the recent slice — hard localization.
      * Curriculum learning (#304, REJECTED): filtered batches by |PnL|
        magnitude. Recent FT filters by CALENDAR position, no PnL
        thresholding at all.
      * Boundary-band exclusion (#317, REJECTED): dropped fence-line
        samples by PnL. Recent FT keeps every sample inside its window.
      * Self-supervised pretraining (#298, REJECTED): unsupervised
        pretext task BEFORE supervised. Recent FT runs AFTER supervised
        using the same supervised loss family (BCE).
      * Mean Teacher (#300, REJECTED): EMA-teacher consistency loss —
        different mechanism; no recent-window concept.
      * Test-time fine-tuning on most-recent N days: explicitly listed
        as UNTRIED in the condensed lessons. This is the legitimate
        in-train proxy (we use the most-recent N TRAIN days, never
        peeking at unseen test data).

    Why this is SAFE
    ----------------
      - Only ~hidden_size+1 = 49 floats (FC layer) update. The LSTM
        backbone (thousands of params) is frozen. Catastrophic
        forgetting of learned features is structurally impossible
        because they don't update.
      - Fixed low LR (2e-4) and few epochs (8). With a few hundred
        recent samples and a 49-param head, expected weight movement
        is small and bounded.
      - Saved model has the IDENTICAL signature to Entry 197 — gate
        loader, live trader, scaler are all unchanged. Only the FC
        weights end up at a slightly different location.
      - Degrades gracefully: when use_recent_finetune=False, dates_train
        is None, or n_recent_samples < 100, this function is a no-op
        and the trained model is bit-identical to the SWA-only baseline.
      - bias calibration runs AFTER recent FT, so the precision-targeted
        threshold is computed against the recent-FT-shifted predictions.
        This composes cleanly: FT shifts decision boundary toward the
        regime closest to test, then calibration tunes the FINAL
        threshold against the FULL val period.
    """
    if dates_train is None or len(dates_train) == 0:
        return {'applied': False, 'reason': 'no dates_train'}

    try:
        dates_arr = pd.to_datetime(np.asarray(dates_train))
    except Exception as exc:
        return {'applied': False, 'reason': f'date parse failed: {exc}'}

    cutoff = dates_arr.max() - pd.Timedelta(days=int(n_recent_days))
    mask = np.asarray(dates_arr > cutoff, dtype=bool)
    n_recent = int(mask.sum())
    if n_recent < 100:
        return {'applied': False, 'reason': f'only {n_recent} recent samples'}

    X_recent = X_train_scaled[mask]
    y_recent = y_train[mask]

    pos_rate_recent = float(y_recent.mean())
    if pos_rate_recent <= 0.0 or pos_rate_recent >= 1.0:
        return {'applied': False,
                'reason': f'degenerate recent labels (pos_rate={pos_rate_recent:.3f})'}
    pos_weight_recent = float(min((1.0 - pos_rate_recent)
                                  / max(pos_rate_recent, 1e-6), pos_weight_cap))

    device = DEVICE
    model.to(device)

    # Freeze every parameter, then re-enable grads only on the FC layer.
    # Guarantees the LSTM backbone (and any attention) never updates —
    # recent FT can ONLY shift the decision boundary.
    for p in model.parameters():
        p.requires_grad_(False)
    if isinstance(model.fc, nn.Linear):
        fc_layer = model.fc
    else:
        fc_layer = model.fc[-1]
    for p in fc_layer.parameters():
        p.requires_grad_(True)
    fc_params = [p for p in fc_layer.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(fc_params, lr=lr, weight_decay=0.0)
    criterion = nn.BCELoss(reduction='none')

    X_recent_t = torch.tensor(X_recent, dtype=torch.float32, device=device)
    y_recent_t = torch.tensor(y_recent, dtype=torch.float32, device=device)

    initial_loss = None
    final_loss = None
    n_steps_total = 0

    # train() mode keeps any input_dropout / INPUT_NOISE / hidden dropout
    # active — they act as regularizers on the small recent window so
    # the FC head doesn't overfit the few hundred samples it sees.
    model.train()
    for ep in range(int(epochs)):
        perm = torch.randperm(len(X_recent_t), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 8:
                continue
            x_b = X_recent_t[idx]
            y_b = y_recent_t[idx]
            optimizer.zero_grad()
            pred = model(x_b)
            class_w = y_b * pos_weight_recent + (1.0 - y_b)
            loss = (criterion(pred, y_b) * class_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fc_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            n_steps_total += 1
        avg = epoch_loss / max(n_batches, 1)
        if ep == 0:
            initial_loss = avg
        final_loss = avg
        if verbose and (ep % 2 == 0 or ep == int(epochs) - 1):
            print(f'    Recent-FT epoch {ep}: loss={avg:.4f} '
                  f'(n_batches={n_batches})')

    model.eval()
    # Hygiene: re-enable grads on every parameter so any subsequent code
    # (or the saved model on reload) is in the standard state.
    for p in model.parameters():
        p.requires_grad_(True)

    info = {
        'applied': True,
        'n_recent_samples': int(n_recent),
        'n_recent_days': int(n_recent_days),
        'epochs': int(epochs),
        'lr': float(lr),
        'initial_loss': float(initial_loss) if initial_loss is not None else None,
        'final_loss': float(final_loss) if final_loss is not None else None,
        'pos_rate_recent': pos_rate_recent,
        'pos_weight_recent': pos_weight_recent,
        'n_optimizer_steps': int(n_steps_total),
    }
    if verbose:
        print(f'  Recent-FT done: {n_recent} samples in last '
              f'{n_recent_days} days, {epochs} epochs, '
              f'loss {initial_loss:.4f} -> {final_loss:.4f}')
    return info


class _GradReverse(torch.autograd.Function):
    """Gradient-reversal layer (Ganin & Lempitsky 2015, arXiv:1409.7495).

    Identity on the forward pass; multiplies the incoming gradient by -alpha
    on the backward pass. Placed between a shared feature extractor and an
    auxiliary domain classifier, it turns the discriminator's "minimize
    domain loss" into the feature extractor's "maximize domain loss" —
    i.e., produce features that the discriminator CANNOT tell apart.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def _grad_reverse(x, alpha):
    return _GradReverse.apply(x, alpha)


class _DomainClassifier(nn.Module):
    """Small MLP discriminating train (label=0) vs val (label=1) features.

    Kept intentionally shallow — if it outpaces the feature extractor's
    adversarial pushback the GRL gradient becomes noise. Outputs a raw
    logit; paired with BCEWithLogitsLoss for numeric stability.
    """

    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _lstm_features(model, x):
    """Pre-FC pooled features from LSTMModel, preserving the autograd graph.

    Mirrors model.forward() but stops before the FC layer so DANN can route
    the features through a gradient-reversal layer and a domain classifier
    without touching the shared model class. Honors seq_normalize, input
    dropout, training-time input/hidden noise, use_attention pooling, and
    output_norm exactly as the production forward path does.
    """
    if model._seq_normalize.item():
        x = model._normalize_sequence(x)
    x = model.input_dropout(x)
    if model.training and model.INPUT_NOISE_STD > 0:
        x = x + torch.randn_like(x) * model.INPUT_NOISE_STD
    if getattr(model, 'use_mlp', False):
        agg = model._aggregate_sequence(x)
        return model.mlp_hidden(agg)
    out, _ = model.lstm(x)
    if model.training and model.hidden_noise_std > 0:
        out = out + torch.randn_like(out) * model.hidden_noise_std
    if model.use_attention:
        T = out.size(1)
        w = torch.exp(torch.linspace(-2.0, 0.0, T, device=out.device))
        w = w / w.sum()
        out = (out * w.view(1, -1, 1)).sum(dim=1)
        out = model.dropout(out)
    else:
        out = model.dropout(out[:, -1, :])
    if hasattr(model, 'output_norm'):
        out = model.output_norm(out)
    return out


def _head_forward(model, features):
    """Apply FC + temperature-scaled sigmoid on pre-FC features."""
    logit = model.fc(features).squeeze(-1)
    return model.sigmoid(logit / model._output_temp)


def _pretrain_lstm_backbone(model, X_scaled_list, epochs=25, batch_size=256,
                            lr=1e-3, verbose=False):
    """Self-supervised pretraining: autoregressive next-step feature prediction.

    Pretext task
    ------------
    For each sequence (B, T, F), the LSTM consumes all T timesteps and its
    hidden state h[:, t, :] is asked to predict features[:, t+1, :] via a
    temporary Linear(hidden_size, input_size) decoder. Loss is MSE on
    positions 0..T-2 vs targets at positions 1..T-1. The decoder is
    discarded afterward — only model.lstm weights are carried forward
    into supervised training.

    Why this addresses the "epoch-0-best" pathology
    ------------------------------------------------
    With random init, the LSTM has no temporal prior; noisy 11% binary
    labels converge training into label-specific shortcuts within 1-2
    epochs. Pretraining a proxy task that requires capturing HOW features
    evolve (not WHICH labels) gives the supervised phase a stable
    regime-invariant backbone to fine-tune on. Classification then only
    needs to learn a decision boundary on top of features that already
    encode market dynamics.

    Generalization argument
    -----------------------
    Next-step structure exists in every regime. The pretext signal is
    consistent across train → val → test time, so features useful for
    pretraining transfer to any test window. Unlike adversarial domain
    alignment (DANN #296), this ADDS information rather than REMOVING
    it — it doesn't require features to look identical across eras, only
    to capture evolution within each era.

    Safety properties
    -----------------
    - No outcome labels touched during pretraining → no forward leakage.
    - Val is in-sample for model selection, so pretraining on train+val
      is legitimate (just as supervised BCE uses val for early stopping).
    - LSTMModel.forward regularizers (input_dropout, INPUT_NOISE_STD,
      sequence_normalize) are BYPASSED during pretraining — those are
      classification-specific noise injection; pretraining operates on
      clean scaled features so the decoder can actually learn.

    Returns an info dict for logging. Silent no-op in MLP mode
    (num_layers=0) since there's no recurrent backbone to pretrain.
    """
    if getattr(model, 'use_mlp', False) or not hasattr(model, 'lstm'):
        return {'applied': False, 'reason': 'MLP mode (no LSTM to pretrain)'}

    device = DEVICE
    model.to(device)
    input_size = model.input_size
    hidden_size = model.hidden_size

    X_all = np.concatenate([np.asarray(arr) for arr in X_scaled_list], axis=0)
    if X_all.size == 0 or X_all.shape[1] < 2:
        return {'applied': False, 'reason': 'insufficient sequence length'}
    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    n_samples = X_all_t.size(0)

    decoder = nn.Linear(hidden_size, input_size).to(device)
    params = list(model.lstm.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()

    model.train()
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() == 0:
                continue
            x_batch = X_all_t[idx]
            # Feed clean scaled features straight to model.lstm, bypassing
            # LSTMModel.forward's classification-time regularizers.
            h, _ = model.lstm(x_batch)
            pred = decoder(h[:, :-1, :])
            target = x_batch[:, 1:, :]
            loss = mse(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg_mse = total_loss / max(n_batches, 1)
        history.append(avg_mse)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f'  SS-pretrain epoch {epoch}: next-step MSE={avg_mse:.4f}')

    model.eval()
    initial = float(history[0]) if history else None
    final = float(history[-1]) if history else None
    reduction = float((initial - final) / initial) if initial and initial > 0 else 0.0

    if verbose:
        print(f'  SS-pretrain complete: MSE {initial:.4f} -> {final:.4f} '
              f'({reduction*100:.1f}% reduction over {epochs} epochs, '
              f'{n_samples} sequences)')

    return {
        'applied': True,
        'epochs': epochs,
        'n_samples': int(n_samples),
        'initial_mse': initial,
        'final_mse': final,
        'mse_reduction': reduction,
    }


def _pretrain_lstm_supcon(model, X_train_scaled, y_train, dates_train,
                          epochs=12, batch_size=512, lr=1e-3,
                          temperature=0.1, projection_dim=32,
                          min_per_day=8, max_per_day=128,
                          verbose=False):
    """Cross-symbol Supervised Contrastive (SupCon) pretraining (THIS ATTEMPT).

    Mechanism
    ---------
    Pretrains the LSTM backbone on a SAME-DAY contrastive task BEFORE the
    supervised BCE phase. Within each training day's cross-section:
      - Run all eligible-day samples through the LSTM backbone to produce
        pre-FC pooled features.
      - Project to a 32-dim L2-normalized embedding via a temporary
        projection head (2-layer MLP, discarded after pretraining).
      - Apply SupCon loss (Khosla et al. 2020, arXiv:2004.11362):
            L_i = -1/|P(i)| * sum_{p in P(i)} log(
                  exp(z_i · z_p / T) /
                  sum_{a != i} exp(z_i · z_a / T) )
        where P(i) = {j != i : same-day, same-label as i}.
        Pulls embeddings of same-LABEL samples within the same day
        together; pushes opposite-label apart.

    Why SupCon attacks the 37.5%-WR plateau / "epoch-0-best" pathology
    -----------------------------------------------------------------
    The dominant failure mode is temporal feature-distribution shift:
    the model learns era-specific shortcuts within 1-2 epochs of BCE on
    11% positive rate, then val_loss diverges immediately. SupCon attacks
    this BEFORE BCE ever runs:
      1. RELATIVE within-day structure is regime-invariant by construction.
         Even in a flat regime, SOME stocks outperform others on any given
         day. The contrastive task ("which features make winners cluster
         apart from losers WITHIN this day") cannot exploit era-specific
         shortcuts because the contrast is local in time.
      2. The encoder is initialized into a region of weight space where
         winners and losers are already linearly separable in embedding
         space. The supervised BCE phase then only needs to learn a
         decision boundary on top of features that already encode the
         distinguishing structure.
      3. NO ABSOLUTE label dependence — the loss only cares about pairs
         (same vs different label, within the same day). Label noise from
         the +4% fence-line samples is partially absorbed because a noisy
         "winner" sample that's actually a loser still contributes a
         coherent contrastive signal: it pulls toward other "winners" of
         the same day (regardless of their absolute PnL), and the
         repulsion from losers averages over many pairs.

    Why this is NOT one of the already-rejected mechanisms
    ------------------------------------------------------
      * Autoregressive SS pretrain (#298, REJECTED): unsupervised next-step
        FEATURE reconstruction. Uses no labels. SupCon uses labels and
        within-day STRUCTURE explicitly.
      * Bootstrap deep-ensemble distillation (#314, REJECTED): K
        independently-trained ensemble averaged at OUTPUT level. SupCon
        is single-model REPRESENTATION pretraining.
      * Pairwise / listwise daily ranking (#303, #310, REJECTED): operates
        at the LOSS-on-output level (compares scalar predictions). SupCon
        operates at the EMBEDDING level (compares feature vectors), and
        runs BEFORE the supervised loss starts.
      * SelectiveNet (#312, REJECTED): learned abstain head on outputs.
        SupCon is upstream of any output head.
      * Mean Teacher (#300, REJECTED): EMA-teacher consistency on outputs.
        SupCon's contrastive structure does not require a teacher copy.
      * DANN (#296, REJECTED): adversarial alignment of train/val FEATURE
        distributions. SupCon does not use val features at all; it
        organizes train features by within-day label structure.
      * Curriculum / boundary-band exclusion (#304, #317, REJECTED):
        sample FILTERING by PnL magnitude. SupCon contributes every
        eligible-day sample to the loss; no filtering.
      * Time-decay / recent-window FT (#316, #318, REJECTED): chronological
        re-weighting. SupCon weighs every day equally on its own
        cross-section — the contrast is INTRA-day, not across time.
      * Listed as UNTRIED in the condensed lessons:
        "(5) Cross-symbol contrastive learning to learn which patterns
              generalize beyond a single ticker's idiosyncrasies."

    Why this is SAFE
    ----------------
      - Only model.lstm parameters update; the projection head is on a
        SEPARATE optimizer and discarded after pretraining. The model's
        FC layer and (optional) attention/output_norm parameters are
        UNTOUCHED — they enter supervised training at their fresh-init
        values, with the LSTM backbone now better-organized.
      - Saved model has IDENTICAL signature to Entry 197. Gate loader,
        live trader, scaler are all unchanged. Production inference is
        bit-identical.
      - Compute cost: ~12 epochs of LSTM forward+backward on training
        data, no decoder forward pass per timestep (one pooled feature
        per sequence). Comfortably within the 30-min WF budget.
      - Degrades gracefully: when use_supcon_pretrain=False, dates_train
        is None, or no day has >= min_per_day samples with both classes,
        this function is a no-op.

    Args:
      model: LSTMModel instance (modified in place).
      X_train_scaled: (N, T, F) numpy array, already scaled.
      y_train: (N,) numpy array of binary labels.
      dates_train: (N,) array of date identifiers (strings or datetimes).
      epochs: number of pretraining passes.
      batch_size: target #samples per packed batch (multiple days).
      lr: Adam LR for projection head + LSTM backbone.
      temperature: SupCon temperature (lower = sharper contrast).
      projection_dim: dim of L2-normalized contrastive embeddings.
      min_per_day: skip days with fewer samples than this.
      max_per_day: cap to avoid OOM on extremely-dense days.
    """
    if getattr(model, 'use_mlp', False) or not hasattr(model, 'lstm'):
        return {'applied': False, 'reason': 'MLP mode (no LSTM to pretrain)'}
    if dates_train is None:
        return {'applied': False, 'reason': 'no dates_train'}

    dates_arr = np.asarray(dates_train)
    if len(dates_arr) != len(X_train_scaled):
        return {'applied': False, 'reason': 'dates_train length mismatch'}

    device = DEVICE
    model.to(device)
    hidden_size = model.hidden_size

    # Build date -> [indices] map. Only keep days with >= min_per_day samples
    # AND at least one sample of each class. Without both classes the SupCon
    # loss contribution is zero (no positives or no negatives).
    unique_dates, date_inv = np.unique(dates_arr, return_inverse=True)
    y_arr = np.asarray(y_train).astype(np.float32)
    date_to_indices = {}
    for i, d in enumerate(date_inv):
        date_to_indices.setdefault(int(d), []).append(i)

    eligible = []
    for d, idxs in date_to_indices.items():
        if len(idxs) < min_per_day:
            continue
        n_pos = int((y_arr[idxs] > 0.5).sum())
        n_neg = len(idxs) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        eligible.append((d, idxs))

    if len(eligible) < 2:
        return {'applied': False, 'reason': f'only {len(eligible)} eligible days'}

    proj = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, projection_dim),
    ).to(device)

    params = list(model.lstm.parameters()) + list(proj.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    X_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_arr, dtype=torch.float32, device=device)

    history = []
    rng_master = np.random.RandomState(42)

    for ep in range(int(epochs)):
        rng = np.random.RandomState(rng_master.randint(2**31 - 1))
        order = list(range(len(eligible)))
        rng.shuffle(order)

        # Pack consecutive eligible days into a single forward pass until
        # the cumulative sample count reaches batch_size. Each "step"
        # processes K days totalling ~batch_size samples; SupCon loss is
        # computed per-day and averaged. This amortizes the LSTM forward
        # pass and keeps the per-day cross-section structure exact.
        epoch_loss = 0.0
        n_steps = 0
        n_pairs_total = 0
        buf_indices = []
        buf_day_offsets = []  # [(start, end, n_pos, n_neg), ...]

        def _flush(buf_indices, buf_day_offsets):
            nonlocal epoch_loss, n_steps, n_pairs_total
            if not buf_indices or not buf_day_offsets:
                return
            idx_t = torch.tensor(buf_indices, dtype=torch.long, device=device)
            X_batch = X_t[idx_t]
            y_batch = y_t[idx_t]

            # Backbone forward — bypass classification-time noise/dropout
            # so the contrastive signal is on clean features.
            h, _ = model.lstm(X_batch)
            feat = h[:, -1, :]
            z = proj(feat)
            z = nn.functional.normalize(z, p=2, dim=-1)

            day_losses = []
            for (start, end, _np, _nn) in buf_day_offsets:
                n_d = end - start
                if n_d < min_per_day:
                    continue
                z_d = z[start:end]
                y_d = y_batch[start:end]
                # Pairwise similarity scaled by temperature. Stabilize by
                # subtracting the per-row max BEFORE the logsumexp — keeps
                # exp(sim) bounded by 1 so we never hit overflow with
                # cosine similarity in [-1, 1] / T.
                sim = torch.matmul(z_d, z_d.t()) / max(temperature, 1e-3)
                sim = sim - sim.max(dim=1, keepdim=True).values.detach()
                eye = torch.eye(n_d, device=device, dtype=torch.bool)
                # Positive pairs: same label, not self.
                pos_mask = (y_d.unsqueeze(0) == y_d.unsqueeze(1)) & ~eye
                non_self = (~eye).float()
                # log-softmax denominator over all NON-SELF contrasts only.
                # Multiply exp(sim) by non_self mask before summing — avoids
                # the NaN from -inf*0 on the diagonal.
                exp_sim = torch.exp(sim) * non_self
                log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
                log_prob = sim - log_denom
                n_pos_per = pos_mask.sum(dim=1).float()
                # Skip rows with no positives (shouldn't happen given the
                # eligibility check, but defensive against edge cases).
                valid_rows = n_pos_per > 0
                if int(valid_rows.sum().item()) == 0:
                    continue
                pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1)
                per_sample = -(pos_log_prob[valid_rows]
                               / n_pos_per[valid_rows].clamp(min=1.0))
                day_losses.append(per_sample.mean())
                n_pairs_total += int(pos_mask.sum().item())

            if not day_losses:
                return
            loss = torch.stack(day_losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_steps += 1

        model.train()
        for o in order:
            d, idxs = eligible[o]
            if len(idxs) > max_per_day:
                # Stratified subsample preserving the within-day class
                # ratio — SupCon needs both classes present.
                idxs_arr = np.asarray(idxs)
                pos_mask = y_arr[idxs_arr] > 0.5
                pos_idx = idxs_arr[pos_mask]
                neg_idx = idxs_arr[~pos_mask]
                target_pos = max(2, int(max_per_day * len(pos_idx)
                                        / len(idxs_arr)))
                target_neg = max(2, max_per_day - target_pos)
                pos_pick = rng.choice(pos_idx,
                                      size=min(target_pos, len(pos_idx)),
                                      replace=False)
                neg_pick = rng.choice(neg_idx,
                                      size=min(target_neg, len(neg_idx)),
                                      replace=False)
                day_indices = list(pos_pick) + list(neg_pick)
            else:
                day_indices = list(idxs)
            n_pos = int((y_arr[day_indices] > 0.5).sum())
            n_neg = len(day_indices) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            start_off = len(buf_indices)
            buf_indices.extend(day_indices)
            buf_day_offsets.append((start_off, len(buf_indices), n_pos, n_neg))
            if len(buf_indices) >= batch_size:
                _flush(buf_indices, buf_day_offsets)
                buf_indices = []
                buf_day_offsets = []
        _flush(buf_indices, buf_day_offsets)

        avg = epoch_loss / max(n_steps, 1)
        history.append(avg)
        if verbose and (ep % 3 == 0 or ep == int(epochs) - 1):
            print(f'  SupCon epoch {ep}: loss={avg:.4f} '
                  f'({n_steps} steps, {n_pairs_total} positive pairs)')

    model.eval()
    initial = float(history[0]) if history else None
    final = float(history[-1]) if history else None
    reduction = float((initial - final) / initial) if initial and initial > 0 else 0.0

    if verbose:
        print(f'  SupCon pretrain complete: loss {initial:.4f} -> {final:.4f} '
              f'({reduction*100:.1f}% reduction over {epochs} epochs, '
              f'{len(eligible)} eligible days)')

    return {
        'applied': True,
        'epochs': int(epochs),
        'n_eligible_days': len(eligible),
        'temperature': float(temperature),
        'projection_dim': int(projection_dim),
        'initial_loss': initial,
        'final_loss': final,
        'loss_reduction': reduction,
        'min_per_day': int(min_per_day),
        'max_per_day': int(max_per_day),
    }


class _DateBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that packs whole training-date groups into each batch.

    Rationale: pairwise-ranking loss needs dense per-day cross-sections in
    each batch. With 49K samples spread over ~600 dates, a fully-random
    batch of 256 averages 0.4 samples/date — almost never yielding the
    >=6 samples + >=1 winner + >=1 loser needed for a ranking pair. By
    packing whole dates (~60-70 samples/date) consecutively until the
    batch is at least `batch_size` large, each yielded batch covers
    ~3-5 dates with full same-day cohorts intact.

    The sampler yields LISTS of indices (PyTorch BatchSampler protocol),
    not single indices, so DataLoader(batch_sampler=sampler) feeds each
    list directly to collate_fn.
    """

    def __init__(self, date_ids, batch_size, shuffle=True, seed=0):
        self.date_ids = np.asarray(date_ids)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        # Build date -> [indices] mapping
        self.date_to_indices = {}
        for i, d in enumerate(self.date_ids):
            self.date_to_indices.setdefault(int(d), []).append(i)
        self.unique_dates = sorted(self.date_to_indices.keys())
        self._epoch = 0
        # Precompute length (approximate, within 1 of the actual yield count)
        total = len(self.date_ids)
        self._len = max(1, (total + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        dates = list(self.unique_dates)
        if self.shuffle:
            rng.shuffle(dates)
        batch = []
        for d in dates:
            indices = list(self.date_to_indices[d])
            if self.shuffle:
                rng.shuffle(indices)
            batch.extend(indices)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return self._len


def _daily_rank_loss(pred, y, date_ids, min_per_day=6, margin=0.05):
    """Pairwise margin-hinge ranking loss on daily cross-sections.

    For each unique date in the batch with >=min_per_day samples and at
    least one winner (y>=0.5) AND one loser, form ALL (winner, loser)
    pairs and compute
      pair_loss = mean over pairs of clamp(margin - (score_w - score_l), 0)
    Loss returned is the mean across qualifying days (each day gets
    equal weight regardless of its pair count, so high-volatility days
    don't dominate).

    Hinge-margin rather than RankNet log-sigmoid is used because:
      (a) Winners only need to clear losers by `margin` in probability
          space — no asymptotic pressure to push scores further apart
          after separation is achieved. This keeps scores compatible
          with the downstream 0.6 classification threshold.
      (b) Zero gradient once the pair is cleanly separated. With 9%
          base rate the pair count is dominated by winners-vs-losers
          across many losers; a log-sigmoid would accumulate gradient
          indefinitely on the tail.

    Returns a 0-d tensor (possibly requiring grad) or None if no day
    in the batch qualifies.
    """
    unique_dates = torch.unique(date_ids)
    loss_terms = []
    n_pairs_total = 0
    for d in unique_dates:
        mask = date_ids == d
        n_d = int(mask.sum().item())
        if n_d < min_per_day:
            continue
        y_d = y[mask]
        s_d = pred[mask]
        pos_mask = y_d > 0.5
        n_pos = int(pos_mask.sum().item())
        n_neg = n_d - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        pos_s = s_d[pos_mask]
        neg_s = s_d[~pos_mask]
        # diff[i, j] = score_winner_i - score_loser_j
        diff = pos_s.unsqueeze(1) - neg_s.unsqueeze(0)
        pair_loss = torch.clamp(margin - diff, min=0.0).mean()
        loss_terms.append(pair_loss)
        n_pairs_total += n_pos * n_neg
    if not loss_terms:
        return None, 0
    return torch.stack(loss_terms).mean(), n_pairs_total


def _listnet_topk_loss(pred, pnl, date_ids, min_per_day=6, top_k_frac=0.2,
                        temperature=0.5):
    """Listwise top-K softmax ranking loss on per-day cross-sections.

    For each day with >= min_per_day samples:
      1. Logit-transform sigmoid predictions to unbounded scores.
      2. Predicted distribution p_i = softmax(logit_i / T) over the day.
      3. Target distribution q_i = 1/K for the K samples with highest
         realized PnL that day, else 0.
      4. Cross-entropy L_day = -sum(q_i * log(p_i)).
    Loss is mean across qualifying days.

    Using continuous PnL for target construction (rather than binary labels)
    eliminates the 4%-fence-line noise inherent to pairwise ranking and
    produces a gradient signal that scales with the MARGIN between winners
    and losers, not just their sign.

    Top-K truncation (Cao et al. 2008) focuses the learning pressure on
    the head of the ranking — exactly where the live 0.6 threshold
    operates after calibration.

    Args:
      pred: (B,) sigmoid scores in (0,1).
      pnl:  (B,) realized PnL per sample (continuous).
      date_ids: (B,) integer date identifiers.
      min_per_day: skip days with fewer samples.
      top_k_frac: fraction of each day's stocks to use as positive
                  target set (clamped to >= 1).
      temperature: softmax temperature; T<1 sharpens, T>1 flattens.

    Returns (loss_0d_tensor, n_qualifying_days) or (None, 0) if no day
    in the batch qualifies.
    """
    unique_dates = torch.unique(date_ids)
    losses = []
    n_days = 0
    for d in unique_dates:
        mask = date_ids == d
        n_d = int(mask.sum().item())
        if n_d < min_per_day:
            continue
        s_d = pred[mask].clamp(1e-6, 1 - 1e-6)
        pnl_d = pnl[mask]
        # Unique top-K selection requires at least as many distinct PnLs as K.
        # Ties in pnl (common: multiple stocks all hit SL at -3%) resolve
        # arbitrarily in torch.topk — acceptable because the loss is symmetric
        # in the target set members.
        logit_d = torch.log(s_d / (1.0 - s_d))
        log_p = torch.log_softmax(logit_d / max(float(temperature), 1e-3), dim=0)
        k = max(1, int(n_d * top_k_frac))
        _, top_idx = torch.topk(pnl_d, k)
        # Cross-entropy with uniform-on-top-K target, closed form:
        #   L = -(1/k) * sum_{i in top-K} log_p[i]
        per_day = -log_p[top_idx].mean()
        losses.append(per_day)
        n_days += 1
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), n_days


class SelectionHead(nn.Module):
    """Selection function g(x) for SelectiveNet-style training (#312).

    A small MLP on the LSTM's pre-FC pooled features returning g ∈ (0,1)
    per sample. Trained jointly with the classifier f(x) such that
    f(x) becomes most accurate on samples where g(x) selects, while
    g(x) learns to identify the regime-invariant trust region.

    Discarded after training — only the LSTMModel (with f(x)) is saved.
    Production inference, the WF gate's loader, and live trading code
    are bit-identical regardless of whether SelectiveNet was used.
    """

    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        h_mid = max(hidden_size // 2, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, h_mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_mid, 1),
        )

    def forward(self, h):
        return torch.sigmoid(self.net(h)).squeeze(-1)


def train_model(X_train, y_train, X_val, y_val, n_features, features,
                hidden_size=48, num_layers=1, dropout=0.4, lr=5e-4,
                batch_size=256, max_epochs=200, patience=20, verbose=True,
                early_sl_train=None, early_sl_val=None,
                survival_weight=1.0, fp_weight=1.0,
                use_attention=False,
                sequence_normalize=False,
                mixup_alpha=0.3,
                swa_window=5,
                # Covariate-shift density-ratio reweighting (#295, rejected).
                # Now OFF by default in favor of adversarial DANN adaptation
                # below — the density-ratio approach saturated because the
                # domain classifier scored AUC≈1.0, driving all weights to
                # the clip bounds and reducing to near-uniform reweighting.
                use_covariate_shift=False,
                # XGBoost teacher distillation — default OFF. Entries 291/292
                # (XGB and tree-ensemble distillation) were rejected on the
                # 7-split gate. Sweep can still enable for ablation.
                use_xgb_distill=False,
                distill_alpha=0.7,
                # PnL-optimization loss (Entry 290). Default OFF — the
                # cost-aware differentiable objective was rejected on the
                # 7-split gate. Retained here so sweep can re-enable via
                # use_pnl_loss=True for ablation. WF gate unconditionally
                # passes pnl_train/pnl_val, so we now require an explicit
                # opt-in flag to route there — otherwise SAM + pure BCE
                # (Entry 197 recipe) is used as intended.
                use_pnl_loss=False,
                pnl_train=None, pnl_val=None,
                pnl_loss_scale=20.0, pnl_budget_lambda=5.0,
                pnl_target_selectivity=0.10, pnl_commission=0.011,
                pnl_clip_low=-0.05, pnl_clip_high=0.20,
                # Self-supervised LSTM backbone pretraining (#298, REJECTED).
                # Now OFF by default — replaced by Mean Teacher consistency
                # training (see mt_* args below) as the active structural
                # mechanism for this attempt. Kept as opt-in knob for
                # sweep-mode ablation.
                use_ss_pretrain=False,
                ss_pretrain_epochs=25,
                ss_pretrain_lr=1e-3,
                # Mean Teacher consistency training (#300, REJECTED).
                # Now OFF by default — replaced by pairwise daily-ranking
                # loss as the active structural mechanism for this attempt.
                # Kept as opt-in knob for sweep-mode ablation.
                use_mean_teacher=False,
                mt_ema_decay=0.99,
                mt_lambda_max=0.5,
                mt_rampup_epochs=5,
                # Domain-Adversarial Neural Network (Ganin & Lempitsky 2015).
                # Default OFF — #296 was rejected on the 7-split gate, and
                # stacking it with SS pretraining would fight the pretrained
                # representations (DANN compresses features for separability,
                # pretraining expands them for predictive richness).
                use_dann=False,
                dann_lambda_max=0.3,
                dann_gamma=10.0,
                dann_val_batch_size=256,
                # Legacy kwargs retained so callers (sweep, WF gate) don't break.
                rank_lambda=0.3, rank_pnl_gap=0.02,
                dann_enabled=False, dann_buckets=4,
                # Pairwise daily-ranking loss (#303, REJECTED). The WF gate
                # still passes daily_rank_enabled=True; we leave the default
                # True so that behavior is preserved verbatim and this attempt
                # (curriculum) can compose on top rather than silently
                # overwriting the gate's call. Ranking gracefully degrades
                # under curriculum filtering (days that drop below
                # min_per_day contribute no pairs).
                dates_train=None,
                daily_rank_enabled=True, daily_rank_lambda=0.5,
                daily_rank_min_per_day=6, daily_rank_margin=0.05,
                # PnL-magnitude CURRICULUM LEARNING (#304, REJECTED).
                # Default now OFF — the 7-split gate rejected the curriculum
                # approach. Kept as an opt-in knob for sweep-mode ablation.
                use_curriculum=False,
                curriculum_strong_win=0.10,
                curriculum_strong_loss=-0.02,
                curriculum_strong_epochs=3,
                curriculum_relax_epochs=5,
                curriculum_min_batch=16,
                # GENERALIZED CROSS ENTROPY (THIS ATTEMPT — Zhang & Sabuncu
                # 2018, arXiv:1805.07836). Drop-in replacement for BCE on
                # the supervised-classification path. q=0.7 is the paper's
                # recommended default — tradeoff between CE's fast
                # convergence (q→0) and MAE's label-noise robustness (q→1).
                # Gates:
                #   - Active only on the BCE path (no effect when
                #     use_pnl_loss or use_xgb_distill override the loss).
                #   - Composes freely with mixup, pos_weight, covariate-
                #     shift weights, daily-rank auxiliary loss, and all
                #     other opt-in mechanisms.
                use_gce=False,
                gce_q=0.7,
                # Multi-quantile PnL distributional regression (THIS ATTEMPT).
                # Adds a training-only auxiliary head on top of the shared
                # LSTM features that predicts multiple quantiles of the
                # future PnL via pinball loss. Forces the representation
                # to encode full-distribution magnitude info, not just the
                # binary 4%-threshold decision. Aux head is NOT saved to
                # the LSTMModel state_dict — production inference is
                # unchanged.
                use_quantile_aux=False,
                quantile_aux_weight=0.4,
                quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9),
                quantile_pnl_clip_low=-0.05,
                quantile_pnl_clip_high=0.20,
                # TEMPORAL-CONSISTENCY SUB-SEQUENCE REGULARIZATION (#309).
                # Default flipped to False for THIS ATTEMPT so the listwise
                # ranking change is isolated. Kept opt-in for sweep
                # ablation.
                use_temporal_consist=False,
                consist_weight=0.3,
                consist_trunc=3,
                consist_rampup_epochs=3,
                # LISTWISE TOP-K SOFTMAX RANKING (#310, REJECTED).
                # Default flipped True → False for THIS ATTEMPT so the
                # multi-fold selection change is isolated from the
                # known-failed listnet mechanism. Sweep can re-enable
                # via use_listnet_rank=True for ablation.
                use_listnet_rank=False,
                listnet_top_k_frac=0.2,
                listnet_temperature=0.5,
                # MULTI-FOLD WORST-CASE VAL_LOSS SELECTION (#311, REJECTED).
                # Default flipped True → False for THIS ATTEMPT (#312) so the
                # SelectiveNet abstention head is isolated from the just-
                # rejected selection-side mechanism. Sweep can re-enable
                # via use_multifold_select=True for ablation.
                use_multifold_select=False,
                n_val_folds=4,
                multifold_min_size=64,
                multifold_aggregator='median',
                # SelectiveNet abstain head (#312, REJECTED). Default flipped
                # True -> False for THIS ATTEMPT (#316) so the time-decay
                # weighting change is isolated from the just-rejected
                # SelectiveNet mechanism. Sweep can re-enable via
                # use_selective_head=True for ablation.
                use_selective_head=False,
                # Coverage target: fraction of training samples g(x) should
                # select on average. 0.5 keeps the gradient signal from
                # selective_risk meaningful (not too narrow, not too broad).
                # Lower values (e.g., 0.1) would push g toward the live
                # 0.6-threshold flagging rate but starve selective_risk of
                # samples; higher values dilute the trust-region signal.
                selective_target_coverage=0.5,
                # Coverage-violation penalty multiplier (Geifman & El-Yaniv
                # default = 32). Quadratic on shortfall: max(0, c*-mean(g))²
                # so penalty=8 when mean(g)=0, penalty=0 when mean(g)>=c*.
                selective_lambda=32.0,
                # Mixing weight between auxiliary CE and selective head.
                # Total = (1-α)·BCE + α·(selective_risk + λ·cov_penalty).
                # 0.5 is the paper's default — equal pressure from both
                # objectives so f(x) stays globally honest while gaining
                # selective sharpness on the trust region.
                selective_alpha=0.5,
                # EXPONENTIAL CHRONOLOGICAL TIME-DECAY SAMPLE WEIGHTING
                # (THIS ATTEMPT #316). Per-sample weight w_i = exp(decay *
                # t_i_norm) where t_i_norm in [0,1] is the chronological
                # position of sample i in the train window. With decay=1.0
                # the most-recent sample weighs e≈2.72x the oldest. Mean-
                # normalized to 1, multiplied into shift_weights_np so it
                # composes multiplicatively with class rebalancing without
                # changing the loss family or model architecture.
                #
                # Defaults ON so the WF gate (which doesn't pass this kwarg)
                # picks up the structural change automatically. Active only
                # on the BCE path AND only when dates_train is supplied
                # (the WF gate always passes dates_train).
                # Time-decay weighting (#316, REJECTED). Now default OFF
                # to isolate the THIS ATTEMPT (#318) recent-window FT
                # change. Sweep can re-enable via use_time_decay=True
                # for ablation.
                use_time_decay=False,
                time_decay_factor=1.0,
                # Recent-window FT (#318, REJECTED). Default flipped True ->
                # False so it's isolated from THIS ATTEMPT (#319) SupCon
                # pretraining. Sweep can re-enable via use_recent_finetune.
                use_recent_finetune=False,
                recent_ft_days=30,
                recent_ft_epochs=8,
                recent_ft_lr=2e-4,
                # CROSS-SYMBOL SUPERVISED CONTRASTIVE PRETRAINING (THIS
                # ATTEMPT #319). Stage-0 pretraining of the LSTM backbone
                # before supervised BCE: pulls same-day, same-label
                # embeddings together; pushes opposite-label apart. Forces
                # the encoder into a regime-invariant representation
                # before the noisy 11%-positive BCE signal kicks in.
                # See _pretrain_lstm_supcon for full rationale.
                #
                # Default ON so the WF gate (which doesn't pass this kwarg)
                # picks up the structural change automatically. Active only
                # when dates_train is supplied AND there are >= 2 days with
                # >= supcon_min_per_day samples and both classes present.
                # Degrades gracefully to the (recent-FT-OFF) Entry 197
                # baseline otherwise.
                # SupCon pretraining (#319, REJECTED). Default flipped True ->
                # False so the THIS ATTEMPT (#320) IRM change is isolated
                # from the just-rejected SupCon mechanism. Sweep can re-
                # enable via use_supcon_pretrain=True for ablation.
                use_supcon_pretrain=False,
                supcon_epochs=12,
                supcon_lr=1e-3,
                supcon_temperature=0.1,
                supcon_projection_dim=32,
                supcon_min_per_day=8,
                supcon_max_per_day=128,
                # INVARIANT RISK MINIMIZATION (THIS ATTEMPT #320 — Arjovsky
                # et al. 2019, arXiv:1907.02893). Listed as untried approach
                # (2) in the condensed lessons. Partitions train into K
                # chronological environments and adds a per-env penalty:
                #     pen_e = || d/dw  BCE(w * f(x), y) ||^2  at w=1
                # over each env e. Total loss = ERM + lam * mean_e(pen_e).
                # Unlike Group DRO (#313, REJECTED) which optimizes worst-
                # case per-env loss, IRM forces the OPTIMAL CLASSIFIER to
                # be the same across environments — i.e., the learned
                # representation is one where invariant features dominate.
                # This is the textbook structural attack on the temporal-
                # distribution-shift root cause of "epoch-0-best".
                #
                # Default ON so the WF gate (which doesn't pass these
                # kwargs) picks up the structural change automatically.
                # Active only on BCE path (xgb_distill / pnl_loss override
                # the loss family entirely) AND when dates_train is
                # supplied AND when each environment has >= irm_min_per_env
                # samples. Degrades gracefully to the (now SupCon-OFF)
                # Entry 197 baseline otherwise.
                # IRM (#320, REJECTED). Default flipped True -> False so the
                # THIS ATTEMPT (#321) PnL-distance soft-label change is isolated
                # from the just-rejected IRM mechanism. Sweep can re-enable via
                # use_irm=True for ablation.
                use_irm=False,
                irm_n_envs=4,
                irm_lambda_max=100.0,
                irm_warmup_epochs=15,
                irm_min_per_env=16,
                # PNL-DISTANCE SOFT LABEL SMOOTHING (THIS ATTEMPT #321).
                # Replaces the binary classification target y = 1[pnl > 4%] with
                # a CONTINUOUS soft label
                #     y_soft = sigmoid((pnl - MIN_PROFIT_PCT) / pnl_smooth_scale)
                # so labels become DATA-DRIVEN by realized PnL magnitude rather
                # than a hard fence at +4%. With pnl_smooth_scale=0.02:
                #   - pnl = +0.04 (the threshold)         -> y_soft = 0.500 (max uncertainty)
                #   - pnl = +0.06 (1*scale above)         -> y_soft = 0.731
                #   - pnl = +0.10 (clear winner)          -> y_soft = 0.953
                #   - pnl = +0.15 (target hit)            -> y_soft = 0.996
                #   - pnl = +0.03 (fence-line below)      -> y_soft = 0.378
                #   - pnl = -0.03 (clear stop-loss)       -> y_soft = 0.029
                # Fence-line samples (PnL in 3-5%) — historically a label-noise
                # source — are pulled toward 0.5 and contribute LESS gradient,
                # while clear winners and clear losers stay at sharp targets.
                #
                # Why this attacks the 37.5%-WR plateau / "epoch-0-best" pathology
                # ----------------------------------------------------------------
                # The condensed lessons identify the +4% fence-line as a major
                # source of effective label noise: a +4.1% trade and a +3.9%
                # trade have near-identical trajectories but receive opposite
                # binary labels. Hard BCE on these noisy boundary samples drives
                # an UNBOUNDED gradient that dominates the optimizer in the
                # first 1-2 epochs (the "epoch-0-best" effect). Soft labels
                # reduce the gradient on uncertain samples without REMOVING
                # them: at p_pred=0.5 and y_soft=0.62, |dL/dp| = 0.24 — only
                # 12% of the magnitude a hard label would impose at the same
                # uncertainty. Clean winners and clean losers retain near-full
                # gradient. The optimizer therefore learns from CLEAR evidence
                # in early epochs and only later, when it has settled, gets
                # contributions from ambiguous fence-line samples.
                #
                # Why this is NOT one of the already-rejected mechanisms
                # ------------------------------------------------------
                #   * #199 PnL-WEIGHTED BCE (REJECTED): scaled per-sample loss
                #     by |pnl| but kept HARD binary labels. A +3.9% trade was
                #     still labeled 0; a +4.1% trade was labeled 1; only their
                #     loss magnitudes differed. The fence-line CLASSIFICATION
                #     noise was untouched. This change attacks the LABEL ITSELF.
                #   * #305 GCE (REJECTED): clipped gradient magnitude in
                #     prediction space (q=0.7 caps |dL/dp|) but kept hard
                #     binary targets. Symmetric clipping; no use of pnl info.
                #     This change is ASYMMETRIC by sample (each sample's label
                #     depends on its OWN realized PnL distance from the fence).
                #   * #317 BOUNDARY-BAND EXCLUSION (REJECTED): hard-DROPPED
                #     samples in the |pnl - 4%| < band region. Throws away
                #     gradient signal entirely from a quarter of the dataset.
                #     This change KEEPS every sample but downweights uncertain
                #     ones smoothly via the sigmoid roll-off — no information
                #     is discarded; only its certainty is recalibrated.
                #   * #47 / #72 LABEL SMOOTHING (REJECTED): UNIFORM smoothing
                #     y -> 0.95 / 0.05. Same offset on every sample regardless
                #     of evidence strength. Compresses scores below 0.6
                #     because the maximum target is 0.95. This change keeps
                #     y_soft in (0, 1) as the FULL sigmoid range, with sharp
                #     targets near 0/1 for clean samples and 0.5-ish for
                #     fence-liners — the score distribution stays uncompressed.
                #   * #306 QUANTILE AUX (REJECTED): added a SEPARATE regression
                #     head for {p10,p25,p50,p75,p90}, kept binary main loss.
                #     This change replaces the MAIN loss target itself.
                #   * #304 CURRICULUM (REJECTED): hard sample FILTERING by
                #     |pnl| in early epochs, full set later. Discrete schedule.
                #     This change is continuous-magnitude weighting via the
                #     sigmoid roll-off — no discrete schedule, no filtering.
                #
                # Composition with the rest of the trainer
                # ----------------------------------------
                #   - Mixup interpolates BOTH y (binary) and y_soft linearly.
                #     pos_weight class re-balancing uses the BINARY y (the
                #     positive-class FRACTION, not the loss target). The BCE
                #     loss target is y_soft.
                #   - SWA + precision-targeted bias calibration both run on
                #     the resulting model unchanged. Validation precision is
                #     still computed against the BINARY y_val (the gate's
                #     ground-truth fence is unchanged).
                #   - daily_rank is suppressed when this mechanism is active
                #     so the THIS ATTEMPT change is isolated from the rejected
                #     #303/#310 ranking variants which the WF gate still
                #     passes via daily_rank_enabled=True.
                #
                # Default ON so the WF gate (which doesn't pass these kwargs)
                # picks up the structural change automatically. Active only on
                # the BCE path AND when pnl_train is supplied. Degrades
                # gracefully to the (now IRM-OFF) Entry 197 baseline otherwise.
                # PnL-distance soft labels (#321, REJECTED). Default flipped
                # True -> False so the THIS ATTEMPT (#322) Sortino aux change
                # is isolated from the just-rejected soft-label mechanism.
                # Sweep can re-enable via use_pnl_smooth_labels=True for
                # ablation.
                use_pnl_smooth_labels=False,
                pnl_smooth_scale=0.02,
                # Sortino aux (#322, REJECTED). Default flipped True -> False
                # so the THIS ATTEMPT (#323) gradient-noise change is isolated
                # from the just-rejected Sortino mechanism. Sweep can re-enable
                # via use_sortino_aux=True for ablation.
                use_sortino_aux=False,
                sortino_aux_weight=0.4,
                sortino_target_selectivity=0.10,
                sortino_budget_lambda=2.0,
                sortino_commission=0.011,
                sortino_eps=1e-4,
                sortino_warmup_epochs=2,
                # ANNEALED GRADIENT NOISE INJECTION (THIS ATTEMPT #323 —
                # Neelakantan et al. 2015, arXiv:1511.06807). After
                # loss.backward() and before optimizer.step(), Gaussian
                # noise xi_t ~ N(0, sigma_t^2 I) is added to every
                # parameter's accumulated gradient. The noise std anneals
                # polynomially:
                #     sigma_t = grad_noise_eta /
                #               (1 + step_post_warmup) ** grad_noise_gamma
                # with eta=0.01 and gamma=0.55 (Neelakantan et al. defaults).
                #
                # The first grad_noise_warmup_steps (default 200, ~1 epoch
                # of typical batch counts) inject NO noise so the
                # optimizer first establishes a sensible gradient
                # direction. The polynomial decay 1/(1+t)^0.55 keeps
                # sigma meaningful for the first ~1000 steps before
                # fading toward 0 — which is exactly when the model is
                # most prone to committing to a sharp shortcut.
                #
                # See module docstring for the FLAT-MINIMUM-BIAS,
                # EPOCH-0-ESCAPE, and BAYESIAN-POSTERIOR-APPROXIMATION
                # mechanisms by which this attacks the 322-attempt
                # plateau and the cross-split-WR-variance failure mode.
                #
                # Gradient noise (#323, REJECTED). Default flipped
                # True -> False so THIS ATTEMPT (#324) EV-targeted
                # bias calibration is isolated from the just-rejected
                # gradient-noise mechanism. Sweep can re-enable via
                # use_grad_noise=True for ablation.
                use_grad_noise=False,
                grad_noise_eta=0.01,
                grad_noise_gamma=0.55,
                grad_noise_warmup_steps=200,
                # EV-targeted FC bias calibration (#324, REJECTED).
                # Default flipped True -> False so THIS ATTEMPT (#325) R-Drop
                # is isolated from the just-rejected EV-calibration mechanism.
                # Sweep can re-enable via use_ev_calibration=True for ablation.
                use_ev_calibration=False,
                ev_commission=0.011,
                # ===== R-DROP REGULARIZED DROPOUT (THIS ATTEMPT #325) =====
                # Liang et al. 2021, "R-Drop: Regularized Dropout for Neural
                # Networks", NeurIPS 2021 (arXiv:2106.14448). Active only on
                # the BCE path AND when dropout > 0 (otherwise the two forward
                # passes produce identical predictions and the KL term is a
                # no-op). Default ON so the WF gate picks it up automatically.
                #
                # Mechanism: per batch, run model(X_mix) TWICE with the model
                # in train() mode (different dropout masks). Predictions p1,
                # p2 differ in proportion to the model's dropout sensitivity.
                # The total loss adds a symmetric KL divergence between p1
                # and p2:
                #     L = 0.5 * (BCE(p1, y) + BCE(p2, y))
                #         + r_drop_alpha * (KL(p1||p2) + KL(p2||p1)) / 2
                # KL(p||q) for binary distributions is
                #     p*log(p/q) + (1-p)*log((1-p)/(1-q)).
                #
                # Why R-Drop attacks the 37.5%-WR plateau / epoch-0-best
                # ------------------------------------------------------
                # The dominant 324-attempt failure is "best epoch is 0-2 with
                # val_loss diverging immediately". The condensed lessons
                # attribute this to the model committing to a sharp
                # train-era feature shortcut within 1-2 epochs. R-Drop forces
                # predictions to be invariant under dropout perturbation:
                # features that the model genuinely USES survive both dropout
                # masks (their downstream contribution is robust); features
                # that are random shortcuts produce divergent predictions
                # (and accumulate KL loss). The optimizer is pushed toward
                # using ROBUST features — the ones that survive dropout
                # perturbation — which by construction are the features that
                # carry the most consistent signal. Robustness within an
                # input distribution is a known proxy for robustness across
                # related distributions, attacking the cross-split-WR
                # variance directly.
                #
                # Why this is NOT one of the already-rejected mechanisms
                # ------------------------------------------------------
                #   * Mean Teacher (#300, REJECTED): EMA-teacher consistency.
                #     TEACHER and STUDENT are different network instances
                #     (one is an EMA of the other). R-Drop uses the SAME
                #     network instance with different DROPOUT REALIZATIONS;
                #     no second model exists. Mean Teacher's failure mode
                #     ("teacher = student at init -> early signal near zero")
                #     does NOT apply: with dropout=0.4 the two R-Drop
                #     forward passes produce meaningfully different outputs
                #     from EPOCH 0.
                #   * Temporal-consistency (#309, REJECTED): full vs
                #     truncated SEQUENCE. Same model, different INPUTS. R-Drop
                #     uses the same model AND same input — only the dropout
                #     mask differs. Different mechanism; the gradient signal
                #     is on dropout sensitivity, not sequence-length
                #     sensitivity.
                #   * Bootstrap-bagging deep-ensemble (#314, REJECTED): K
                #     independently-trained ensemble distilled at OUTPUT
                #     level. R-Drop is single-model, single-stage; KL is
                #     applied DURING training to TWO forward passes of the
                #     same model, not to ensemble means.
                #   * SAM (#294, REJECTED): worst-case ascent in a fixed
                #     ρ-ball. Deterministic gradient direction. R-Drop's
                #     extra signal is RANDOM (Bernoulli dropout masks),
                #     unbiased, and trivially cheap.
                #   * Annealed gradient noise (#323, REJECTED): N(0, σ²) on
                #     PARAMETER GRADIENTS. R-Drop's perturbation is on the
                #     ACTIVATIONS via dropout — different layer of the
                #     system. Different theoretical objective.
                #   * Mixup (Entry 197 baseline, ACTIVE): mixes INPUT pairs
                #     and INTERPOLATES labels. R-Drop operates on a single
                #     (mixup'd) input; the regularization is on the model's
                #     OUTPUT distribution, not its input.
                #   * Label smoothing (#47, #72, REJECTED): UNIFORM
                #     y -> 0.95/0.05. Compresses the score distribution
                #     below 0.6. R-Drop keeps the BCE TARGET unchanged
                #     (still y_mix); only the KL of p1 vs p2 is added.
                #     Score distribution is preserved.
                #   * R-Drop is NOT in this project's 324-attempt history
                #     (verified by exhaustive grep on the trainer source).
                #
                # Why this is SAFE
                # ----------------
                #   - Saved model has IDENTICAL signature to Entry 197.
                #     Gate loader, live trader, scaler are all unchanged.
                #     Production inference runs in eval() mode (dropout
                #     OFF) with a SINGLE forward pass — no R-Drop machinery
                #     leaks into inference.
                #   - Compute cost: 2x forward pass per training step.
                #     Backward pass remains 1x because both forwards share
                #     the autograd graph. Comfortably within the 30-min
                #     WF-split budget.
                #   - Linear ramp-up of the KL weight over the first
                #     r_drop_rampup_epochs lets BCE establish a decision
                #     surface before the KL term pulls on it. Avoids the
                #     "two random forward passes early in training have
                #     huge KL → huge gradient → divergence" failure mode.
                #   - r_drop_alpha=5.0 is calibrated for our problem:
                #     with pos_weight≈10, the BCE term is ~3-7 per sample
                #     (heavily class-balanced); KL is ~0.04-0.20 per
                #     sample. alpha=5.0 makes KL contribute ~10-30% of
                #     total loss — meaningful regularization without
                #     drowning BCE.
                #   - Degrades gracefully: when use_r_drop=False (sweep
                #     ablation), dropout=0, or non-BCE path, the code is
                #     bit-identical to the (now ev-calibration-OFF)
                #     baseline.
                #   - Fixed seed (42) for the LSTM keeps run-to-run noise
                #     bounded; the dropout masks themselves are
                #     determined by the global PyTorch RNG seeded at
                #     trainer entry.
                #
                # Companion change for clean isolation
                # ------------------------------------
                #   * daily_rank_active is suppressed when r_drop_active is
                #     True. The WF gate passes daily_rank_enabled=True;
                #     without this guard, the rejected pairwise (#303) /
                #     listnet (#310) rankers would silently activate
                #     alongside R-Drop, muddling the ablation.
                #
                # Reference: Liang et al. (2021). "R-Drop: Regularized
                # Dropout for Neural Networks." NeurIPS.
                # R-Drop (#325, REJECTED). Default flipped True -> False so
                # THIS ATTEMPT (#326) feature-stationarity KS-mask is
                # isolated from the just-rejected R-Drop mechanism. Sweep
                # can re-enable via use_r_drop=True for ablation.
                use_r_drop=False,
                r_drop_alpha=5.0,
                r_drop_rampup_epochs=3,
                # ===== FEATURE-STATIONARITY KS-MASK (THIS ATTEMPT #326) =====
                # Per-feature multiplicative input mask derived from the
                # train/val Kolmogorov-Smirnov 2-sample distance. Mask
                # m_i = max(floor, exp(-alpha * KS_i^2)) is applied to
                # the scaled inputs (broadcast over time) AND baked into
                # scaler.scale_ so inference at the WF gate, in production,
                # and during recalibration all transparently reproduce the
                # same masked feature space. Per-WF-split: each split
                # computes its own mask against its own train/val pair.
                #
                # Default ON so the WF gate (which doesn't pass these
                # kwargs) picks up the structural change automatically.
                # Active on every training path (BCE, GCE, PnL-loss,
                # XGB-distill) because the mask runs once at scaling time
                # and is independent of the loss family.
                #
                # Companion change for clean isolation: daily_rank_active
                # is suppressed when stationarity_active is True so the
                # rejected pairwise (#303) / listnet (#310) rankers don't
                # silently activate alongside this attempt.
                #
                # See _compute_stationarity_mask docstring for the full
                # rationale, the comparison against #295 (covariate-shift
                # density-ratio), #296 (DANN), and the other rejected
                # mechanisms, and the safety arguments around the
                # 'CURATED_FEATURES untouchable' constraint.
                use_stationarity_mask=False,
                stationarity_alpha=10.0,
                stationarity_floor=0.3,
                # ===== POST-SWA TEMPERATURE SCALING (THIS ATTEMPT #327) =====
                # Guo et al. (2017), "On Calibration of Modern Neural Networks",
                # ICML. After SWA averaging — before precision/EV-targeted FC
                # bias calibration — fit a single scalar T to minimize
                # binary-cross-entropy NLL on the val set, with logits
                # transformed as sigmoid(z / T). Updates model._output_temp
                # in place; the saved buffer carries through to the WF gate's
                # loader and to production inference because the model
                # already wraps every forward pass with sigmoid(logit / T).
                #
                # Why temperature scaling attacks the cross-split-WR-variance
                # plateau and the "best epoch is 0-4" pathology
                # ----------------------------------------------------------
                # Across 326 attempts the persistent failure modes have been
                # (i) val_loss diverges by epoch 0-4 (temporal feature
                # distribution shift) and (ii) WR variance across splits is
                # 12-26% even on configurations that average above base
                # rate. Both leave a model whose RANKING of stocks is
                # informative but whose CONFIDENCE values are systematically
                # miscalibrated — Guo et al. proved this is the rule, not
                # the exception, for modern NNs trained with cross-entropy.
                # The bias calibration that runs next computes
                #     b_shift = T * (logit_target - logit_qraw)
                # so the magnitude of the shift, and therefore the
                # robustness of the resulting threshold under regime drift,
                # depends on T being CORRECT. The model ships with
                # output_temperature=0.5 (a sharpening prior chosen to
                # rescue 0-trade configurations from sub-0.6 score
                # collapse). When the model has already learned a healthy
                # decision surface, that aggressive sharpening over-extends
                # confidence into miscalibrated regions; the bias shift
                # then has the wrong magnitude. Fitting T per WF split
                # against that split's own val NLL automatically dampens
                # over-confidence in noisy regimes (T -> larger) and
                # restores sharpness when val NLL prefers it (T -> smaller),
                # giving the bias-calibration step a probability scale
                # that is locally honest. This is the textbook way to put
                # the model's score distribution back on a meaningful
                # interpretive footing without changing what the model
                # learned.
                #
                # Why this is NOT one of the already-rejected mechanisms
                # ------------------------------------------------------
                #   * Selectivity calibration (#289, post-hoc): shifts FC
                #     bias only; treats T as fixed at init. Temperature
                #     scaling fits T FIRST so the bias-shift formula has
                #     the right denominator. They compose cleanly.
                #   * Precision-targeted bias calibration (#308, ACTIVE):
                #     same — only the bias shifts. T stays at init. This
                #     change adds T as a free parameter UPSTREAM of bias
                #     calibration. After T is fit, precision-targeted
                #     calibration runs unchanged on the temperature-scaled
                #     score distribution.
                #   * EV-targeted bias calibration (#324, REJECTED): also
                #     only shifts FC bias. Same as #308 in the T treatment.
                #   * Label smoothing (#47, #72, REJECTED): changed the
                #     BCE TARGETS during training (y -> 0.95/0.05).
                #     Temperature scaling changes the OUTPUT INTERPRETATION
                #     after training; the BCE targets and the saved model
                #     weights are bit-identical to the (now-stationarity-
                #     OFF) baseline. Score distribution is NOT compressed
                #     toward 0.5 the way label smoothing did — only its
                #     spread is rescaled monotonically.
                #   * Multi-fold val select (#311, REJECTED): changed
                #     SELECTION metric (worst/median over K folds).
                #     Temperature scaling does not touch selection at all.
                #   * SAM, R-Drop, gradient-noise (#294, #325, #323,
                #     REJECTED): all OPTIMIZER-level perturbations during
                #     training. Temperature scaling is a CLOSED-FORM
                #     post-hoc 1D fit on val. Zero training-time changes;
                #     zero new randomness; zero new gradient flow.
                #   * NEVER tried in the project's 326-attempt history
                #     (verified by exhaustive grep on this trainer source
                #     for 'temperature_scal', 'temp_scale', '_output_temp.fill_',
                #     'nll_temp', 'TemperatureScaling', 'guo' — all returned
                #     no matches).
                #
                # Why this is SAFE
                # ----------------
                #   - lstm_model.py is unchanged: model already wraps every
                #     forward pass with sigmoid(logit / _output_temp). We
                #     only update the BUFFER VALUE.
                #   - Saved model has BIT-IDENTICAL signature to Entry 197.
                #     state_dict keys, shapes, dtype are unchanged. The
                #     gate's loader, the live trader's loader, and the
                #     inference path all transparently pick up the new T.
                #   - Pure monotone post-hoc 1D fit. Score ORDERING is
                #     preserved exactly; only the spread of confidences
                #     changes. The model's ranking ability — which the
                #     daily-rank, listnet, multifold-select, and similar
                #     ablations all targeted — is COMPLETELY UNAFFECTED.
                #   - Compute cost: one val forward pass to extract logits,
                #     then a 1D bounded scalar minimization (scipy
                #     minimize_scalar with method='bounded'). << 1 second
                #     per WF split.
                #   - Degrades gracefully:
                #       * Val too small (< MIN_VAL_FOR_CALIBRATION): no-op.
                #       * NLL improvement < 1% relative: no-op (T stays
                #         at the init value of 0.5 from LSTMModel).
                #       * scipy unavailable: implements a coarse grid
                #         fallback in pure numpy.
                #       * use_temperature_scaling=False (sweep ablation):
                #         bit-identical to the (now-stationarity-OFF)
                #         baseline.
                #
                # Companion changes for clean isolation
                # -------------------------------------
                #   * use_stationarity_mask default flipped True -> False
                #     (since #326 was just rejected). The mask is still
                #     available as an opt-in for sweep-mode ablation.
                #   * daily_rank_active is suppressed when temp_scaling
                #     is the active mechanism (default ON). The WF gate
                #     passes daily_rank_enabled=True; without this guard,
                #     the rejected pairwise (#303) / listnet (#310)
                #     rankers would silently activate alongside this
                #     attempt's mechanism, muddling the ablation.
                #   * All other rejected-mechanism opt-ins remain default
                #     OFF.
                #
                # Reference: Guo, Pleiss, Sun, Weinberger (2017).
                # "On Calibration of Modern Neural Networks." ICML.
                # Temperature scaling (#327, REJECTED). Default flipped True ->
                # False so THIS ATTEMPT (#328) Manifold Mixup is isolated from
                # the just-rejected post-hoc temperature fit. Sweep can re-enable
                # via use_temperature_scaling=True for ablation.
                use_temperature_scaling=False,
                temperature_min=0.3,
                temperature_max=5.0,
                temperature_min_nll_improvement=0.01,
                # ===== MANIFOLD MIXUP (THIS ATTEMPT #328) =====
                # Verma et al. 2019, "Manifold Mixup: Better Representations
                # by Interpolating Hidden States", ICML (arXiv:1806.05236).
                # Per-batch coin flip: with probability manifold_mixup_p,
                # mix at the LSTM's pre-FC pooled hidden representation
                # instead of at the input. Both passes use the SAME mixup_alpha
                # beta distribution, but the layer at which interpolation
                # happens differs. Verma et al. proved this concentrates
                # representations into "tight clusters", flattens decision
                # boundaries, and substantially improves out-of-distribution
                # generalization vs input mixup alone.
                #
                # Why manifold mixup attacks the 37.5%-WR plateau / "epoch-0-best"
                # ----------------------------------------------------------------
                # The dominant 327-attempt failure is "best epoch is 0-2 with
                # val_loss diverging immediately" — the model commits to a
                # sharp train-era feature shortcut within 1-2 epochs. Input
                # mixup (Entry 197 baseline) flattens decision boundaries in
                # INPUT space; that mostly regularizes which input PATTERNS
                # the model fits. But the sharp-shortcut pathology lives in
                # HIDDEN space — the LSTM's pooled representation collapses
                # winning vs losing patterns onto distinct "clusters" too
                # aggressively, leaving thin decision regions that don't
                # transfer across regimes.
                #
                # Mixing at the hidden layer forces these clusters to be
                # SMOOTHLY interpolatable. A point halfway between a "winner"
                # and "loser" representation must yield a halfway-prediction;
                # that constraint is incompatible with the sharp jagged
                # decision surfaces the optimizer would otherwise commit to
                # within 2 epochs. The hidden geometry becomes flatter,
                # cluster boundaries shift outward, and predictions on
                # never-before-seen test-era patterns interpolate within
                # the learned manifold rather than falling into shortcut
                # crevices.
                #
                # Why this is NOT one of the already-rejected mechanisms
                # ------------------------------------------------------
                #   * Input mixup (Entry 197 baseline, ACTIVE): mixes RAW
                #     INPUT sequences. Manifold mixup mixes the LSTM's
                #     pre-FC POOLED REPRESENTATION. Different layer of the
                #     network; complementary regularization. We keep input
                #     mixup with probability (1-p) per batch and switch to
                #     manifold mixup with probability p, matching Verma
                #     et al.'s recommended hybrid.
                #   * R-Drop (#325, REJECTED): two forward passes with
                #     different DROPOUT MASKS, KL between predictions.
                #     Manifold mixup is ONE forward pass, but mixed at
                #     a deeper layer. No KL term, no second forward pass.
                #   * SupCon (#319, REJECTED): cross-symbol contrastive
                #     pretraining BEFORE supervised. Manifold mixup runs
                #     INSIDE supervised training as a per-batch decision.
                #   * Mean Teacher (#300, REJECTED): EMA-teacher consistency
                #     loss. Different mechanism — no second model.
                #   * SAM (#294), gradient noise (#323), R-Drop (#325):
                #     all OPTIMIZER/PARAMETER-level perturbations. Manifold
                #     mixup is DATA-level (specifically, REPRESENTATION-
                #     level) augmentation. Different layer of the system.
                #   * Label smoothing (#47, #72, REJECTED): UNIFORM
                #     y -> 0.95/0.05. Manifold mixup keeps targets faithful
                #     to the convex combination of labels; the score
                #     distribution is NOT compressed toward 0.5.
                #   * Temporal-consistency (#309, REJECTED): full vs
                #     truncated SEQUENCE — same model, different INPUTS.
                #     Manifold mixup uses the same input length, mixes
                #     internal HIDDEN representations.
                #   * NEVER tried in this project's 327-attempt history
                #     (verified by grep on the trainer source for
                #     'manifold', 'manifold_mixup', '_lstm_features.*lam'
                #     — all returned no matches).
                #
                # Why this is SAFE
                # ----------------
                #   - Saved model has IDENTICAL signature to Entry 197.
                #     Gate loader, live trader, scaler are all unchanged.
                #     Production inference runs in eval() mode with a
                #     SINGLE forward pass — manifold-mixup machinery only
                #     activates during training and only when the per-batch
                #     coin flip selects it.
                #   - Compute cost: identical to Entry 197. One forward
                #     pass per batch; the only change is WHICH layer's
                #     output gets mixed.
                #   - Degrades gracefully: when use_manifold_mixup=False
                #     (sweep ablation) OR mixup_alpha=0 OR any other
                #     advanced mechanism (DANN/IRM/R-Drop/etc.) is active,
                #     the code path is bit-identical to the (now temp-
                #     scaling-OFF) Entry 197 baseline.
                #   - Per-batch coin flip with seed-controlled np.random
                #     keeps run-to-run variance bounded.
                #
                # Companion changes for clean isolation
                # -------------------------------------
                #   * use_temperature_scaling default flipped True -> False
                #     (since #327 was just rejected).
                #   * daily_rank_active is suppressed when manifold_mixup
                #     is the active mechanism (default ON). The WF gate
                #     passes daily_rank_enabled=True; without this guard,
                #     the rejected pairwise (#303) / listnet (#310)
                #     rankers would silently activate alongside manifold
                #     mixup, muddling the ablation.
                #   * All other rejected-mechanism opt-ins remain default
                #     OFF.
                #
                # Reference: Verma, Lamb, Beckham, Najafi, Mitliagkas,
                # Lopez-Paz, Bengio (2019). "Manifold Mixup: Better
                # Representations by Interpolating Hidden States." ICML.
                use_manifold_mixup=False,
                manifold_mixup_p=0.5,
                # ===== DUAL-TARGET MULTI-TASK AUXILIARY HEAD (THIS ATTEMPT #330) =====
                # Adds a small auxiliary classification head — Linear(hidden_size, 1)
                # — that shares the LSTM's pre-FC pooled features with the main
                # classifier and predicts the binary outcome
                #     y_target = 1[pnl >= target_aux_threshold]   (default +15%)
                # via pos-weighted BCE. Total loss = main_BCE + target_aux_weight *
                # aux_BCE. The aux head is created locally in train_model, joined
                # to the main optimizer so its gradient flows back into the LSTM
                # backbone, but is NOT registered to the LSTMModel state_dict and
                # NEVER saved. Production inference (the gate's loader, the live
                # trader) is bit-identical to Entry 197.
                #
                # Why aim the aux at +15% specifically: that is the trading system's
                # ACTUAL REWARD TARGET (TARGET_PCT in labels.py) — the live system
                # only books a full +15% win when the trade hits this level before
                # SL/trailing/timeout. The main classification label is an arbitrary
                # +4% threshold (MIN_PROFIT_PCT) chosen to capture commission-net
                # winners; the AUX label is what the deployment cares about. Forcing
                # shared features to predict both pushes the LSTM to encode magnitude
                # ordering, not just the marginal +4% fence.
                #
                # Why this attacks the 37.5%-WR plateau / cross-split-WR variance
                # ----------------------------------------------------------------
                # Across 329 attempts the dominant failure modes have been (i) the
                # model finds a sharp shortcut at the +4% fence within 1-2 epochs,
                # (ii) cross-split WR variance stays high (12-26%) because the +4%
                # threshold's margin is tiny and noisy across regimes. The +15%
                # target is structurally MORE STABLE across regimes:
                #   * It's far from the SL boundary, so labels are unambiguous —
                #     a stock either ran +15% or it didn't, with no near-fence
                #     mislabeling band.
                #   * It rewards features capturing MOMENTUM/BREAKOUT magnitude,
                #     which generalizes across bull/chop/drawdown regimes better
                #     than features capturing marginal-winner discrimination.
                #   * It's CORRELATED but not IDENTICAL with the +4% task — every
                #     +15% winner is also a +4% winner, but not vice versa. The
                #     joint loss therefore pushes shared features toward signals
                #     that distinguish "merely profitable" from "actually hit
                #     reward target", giving the live 0.6 threshold sharper edge.
                #
                # Why this is NOT one of the already-rejected mechanisms
                # ------------------------------------------------------
                #   * #199 PnL-weighted BCE (REJECTED): scaled per-sample MAIN
                #     loss by |pnl|. Same task, same threshold, just reweighted.
                #     This change adds a SEPARATE TASK at a DIFFERENT THRESHOLD;
                #     main BCE is unchanged.
                #   * #290 PnL-optimization loss (REJECTED): replaced BCE with
                #     -E[pred·(pnl-c)] — single head, profit-maximizing objective.
                #     Here BCE is preserved; the aux head adds another BCE on a
                #     binarized PnL outcome, not a continuous-profit objective.
                #   * #306 / #308 quantile aux (REJECTED): predicted continuous
                #     PnL quantiles via pinball regression. Magnitude regression,
                #     no threshold. This change is BINARY classification at the
                #     +15% reward target — sharp gradient at the boundary,
                #     robust to PnL outliers.
                #   * #321 PnL-distance soft labels (REJECTED): replaced the
                #     binary y with sigmoid((pnl - 0.04) / scale). Single head,
                #     soft target. This change keeps the binary y unchanged for
                #     the main head and ADDS a second head with a different
                #     binary target.
                #   * #322 Sortino aux (REJECTED): differentiable Sortino ratio
                #     on soft-selected trades. Batch-level statistic. This change
                #     is per-sample BCE.
                #   * #312 SelectiveNet abstain head (REJECTED): aux head g(x)
                #     learned WHEN to abstain on the main task. This change's
                #     aux head predicts a DIFFERENT TASK (+15% target hit), not
                #     a selection function over the main task.
                #   * Multi-horizon multi-task labels (5d ∧ 10d ∧ 20d) explicitly
                #     UNTRIED in the condensed lessons. This change is the
                #     same-horizon multi-threshold variant: same 10-day max-hold
                #     window, two thresholds (+4% and +15%). Doesn't require
                #     recomputing sequences for different lookaheads — uses the
                #     existing pnl array directly.
                #
                # Why this is SAFE
                # ----------------
                #   - Aux head is a local nn.Linear(hidden_size, 1), built inside
                #     train_model and discarded at function exit. It is NOT
                #     registered to the LSTMModel and is NOT in model.state_dict().
                #     Saved model (h5 file) has BIT-IDENTICAL signature to
                #     Entry 197 — gate loader, live trader, scaler are all
                #     unchanged.
                #   - feat_shared extraction is already an established pattern
                #     (qr_active / sel_active trigger it). Adding target_aux as
                #     a new trigger reuses that machinery: one forward pass,
                #     two heads, single backward.
                #   - Composition with mixup: y_target is computed from
                #     pnl_batch_raw on the fly and mixed with the same lam used
                #     for y_mix. Linear interpolation of two binary labels
                #     gives a valid soft target in [0, 1] — handled identically
                #     to how y_mix is handled in BCE.
                #   - Composition with class re-balancing: aux task has its own
                #     pos_weight (computed from y_target_train rate), capped at
                #     15. Independent of the main task's pos_weight, so the
                #     two losses each get correctly-balanced gradient at their
                #     respective base rates.
                #   - Compute cost: one extra Linear(48, 1) forward + backward
                #     per batch. << 1% overhead vs the LSTM. The single forward
                #     pass through the LSTM is shared.
                #   - Degrades gracefully when use_target_aux=False (sweep
                #     ablation), pnl_train is None, or pnl_train length doesn't
                #     match X_train. Falls through to the (now-manifold-OFF)
                #     Entry 197 baseline.
                #
                # Companion changes for clean isolation
                # -------------------------------------
                #   * use_manifold_mixup default flipped True -> False (since
                #     #328 was just rejected).
                #   * daily_rank_active is suppressed when target_aux_active is
                #     True. The WF gate passes daily_rank_enabled=True; without
                #     this guard, the rejected pairwise (#303) / listnet (#310)
                #     rankers would silently activate alongside this attempt's
                #     mechanism, muddling the ablation.
                #   * All other rejected-mechanism opt-ins remain default OFF.
                # Dual-target multi-task aux head (#330, REJECTED). Default
                # flipped True -> False so THIS ATTEMPT (#331) Brier-score
                # classification loss is isolated from the just-rejected
                # multi-task aux mechanism. Sweep can re-enable via
                # use_target_aux=True for ablation.
                use_target_aux=False,
                target_aux_weight=0.3,
                # ===== BRIER-SCORE CLASSIFICATION LOSS (THIS ATTEMPT #331) =====
                # Replaces per-sample BCE / GCE with MSE on probabilities
                # (Brier score, Brier 1950; "Verification of forecasts
                # expressed in terms of probability"). Active only on the
                # BCE path (use_xgb_distill=False and use_pnl_loss=False).
                # Default ON so the WF gate picks it up automatically.
                #
                # Mechanism: per_sample = (pred - target)^2 instead of
                #   per_sample = -y*log(pred) - (1-y)*log(1-pred)
                # Composes unchanged with class_weight (pos_weight on the
                # binary positive class), shift_weights (covariate-shift),
                # mixup (linear interpolation of soft targets in [0,1]),
                # and PnL-distance soft labels (continuous targets).
                #
                # Why Brier attacks the "epoch-0-best" pathology
                # ----------------------------------------------
                # The dominant 330-attempt failure mode is "best epoch is
                # 0-2 with val_loss diverging immediately" — the model
                # commits to a sharp train-era feature shortcut within
                # 1-2 epochs of an 11%-positive label distribution. The
                # condensed lessons attribute this to a feature-distribution
                # temporal shift; but the IMMEDIACY of the divergence
                # (epoch 0-2, not 5-10) reveals a loss-geometry component
                # too: BCE has UNBOUNDED gradient magnitude. For y=1 with
                # pred=epsilon, the gradient w.r.t. pred is -1/epsilon, an
                # arbitrarily large pull. A single mislabeled fence-line
                # sample (pnl in 3-5%, label noise band) drives a HUGE
                # gradient into the LSTM to fit that one sample — exactly
                # the "memorize shortcut features" behavior. The optimizer
                # does this in the FIRST TWO EPOCHS because the gradient
                # mass on noisy examples is so disproportionate.
                #
                # Brier (MSE on probabilities) has BOUNDED LINEAR gradient:
                # for any sample, |dL/dp| = 2|p - y| <= 2. Confident-wrong
                # predictions don't get arbitrarily-large gradient — they
                # get the same scale of pull as moderately-confident wrong
                # predictions. The optimizer needs MULTIPLE consistent
                # examples to shift the decision boundary, not just one
                # noisy fence-line sample. This is exactly the "ignore
                # individual mislabeled samples, learn the robust signal"
                # behavior past attempts tried to engineer via curriculum
                # (#304), sample reweighting (#199, #261), and noise-robust
                # losses (#305 GCE) — all rejected, but Brier is a STRICTLY
                # PROPER scoring rule with a fundamentally different loss
                # geometry, not a parameterized softening of BCE.
                #
                # Why Brier is NOT one of the already-rejected mechanisms
                # -------------------------------------------------------
                #   * BCE (Entry 197 baseline, ACTIVE): unbounded gradient
                #     on confident-wrong predictions. -log(p) -> infinity
                #     as p -> 0 for y=1. Brier's (p-y)^2 is bounded by 1.
                #   * GCE q=0.7 (#305, REJECTED): parameterized between CE
                #     (q->0) and MAE (q->1). Bounded gradient via clipping
                #     in prediction space, but still concave-up curvature
                #     (CE-like). Brier is QUADRATIC: gradient is LINEAR
                #     in error, so small errors get small gradient and
                #     large errors get large but bounded gradient — a
                #     genuinely different loss family, not a parameterized
                #     softening of CE.
                #   * MAE / L1 on probabilities: |dL/dp| = constant 1
                #     regardless of error magnitude. Brier's |dL/dp| =
                #     2|p-y| scales with error — confident-correct gets
                #     small gradient, confident-wrong gets large gradient.
                #     Different optimization dynamics; MAE plateaus on
                #     fence-line samples, Brier discriminates them.
                #   * Label smoothing (#47, #72, REJECTED): MOVES the BCE
                #     TARGET from 0/1 toward 0.5. Score distribution
                #     compresses below 0.6 because max target is 0.95.
                #     Brier KEEPS the target at 0/1 (or y_soft if PnL-
                #     distance soft labels are on); only the loss FUNCTION
                #     changes. Score distribution stays uncompressed.
                #   * R-Drop (#325, REJECTED): added a KL-divergence term
                #     between two dropout-perturbed forward passes. Brier
                #     is a SINGLE-FORWARD-PASS REPLACEMENT of the
                #     classification loss; no new term, no second pass.
                #   * SAM (#294, REJECTED): worst-case parameter ascent.
                #     Brier is a data-loss change, not an optimizer change.
                #   * Manifold Mixup (#328, REJECTED): mixed at the hidden
                #     layer. Brier doesn't touch the forward path.
                #   * Sortino aux (#322, REJECTED): batch-level risk-
                #     adjusted return aux loss. Brier replaces the MAIN
                #     per-sample loss; no new aux term.
                #   * NEVER tried in this project's 330-attempt history
                #     (verified by grep on the trainer source for
                #     'brier', 'mse_loss', '(pred.*-.*y).*\\*\\*.*2',
                #     'MSELoss' — all returned no matches in the
                #     classification-loss code path).
                #
                # Why Brier is a strictly proper scoring rule (theory)
                # ----------------------------------------------------
                # A scoring rule s(p, y) is strictly proper if it is
                # uniquely minimized in expectation when p equals the true
                # conditional probability P(Y=1|X). For Brier:
                #   E[ (p - Y)^2 | X ] = (p - P(Y=1|X))^2
                #                       + P(Y=1|X) * (1 - P(Y=1|X))
                # The second term is constant in p; the first is uniquely
                # minimized at p = P(Y=1|X). So the gradient pushes p
                # directly toward the true conditional probability — by
                # construction, Brier produces calibrated probabilities.
                # BCE is also strictly proper but its gradient geometry
                # rewards extreme confidence; Brier's quadratic geometry
                # rewards CALIBRATED confidence. Calibrated probabilities
                # threshold more consistently across the 7 WF splits —
                # which is exactly what the wf_std=0.07-0.25 failure mode
                # demands.
                #
                # Why Brier is SAFE
                # -----------------
                #   - Saved model has IDENTICAL signature to Entry 197.
                #     Gate loader, live trader, scaler are all unchanged.
                #     Production inference is bit-identical regardless of
                #     which loss the model was trained with.
                #   - Compute cost: identical. (pred - target)^2 vs
                #     -y*log(pred) - (1-y)*log(1-pred) — both are
                #     elementwise scalar operations, same FLOP count.
                #   - Per-sample shape matches BCE's reduction='none'
                #     output, so class_weight, shift_weights, daily_rank
                #     auxiliary loss, R-Drop machinery, and all other
                #     mechanisms compose unchanged.
                #   - mixup-blended targets y_mix in [0, 1] handle natively
                #     via the quadratic formula — convex combinations of
                #     0 and 1 give intermediate targets, and (p - y_mix)^2
                #     is well-defined for any y_mix in [0, 1].
                #   - PnL-distance soft labels (when use_pnl_smooth_labels
                #     is on) produce y_soft in (0, 1) which Brier handles
                #     identically to mixup targets — no special handling
                #     needed.
                #   - Class re-balancing via pos_weight is multiplicative
                #     on per_sample, so high-pos_weight on Brier still
                #     correctly amplifies positive-class gradient mass.
                #   - Degrades gracefully:
                #       * use_brier_loss=False (sweep ablation): code is
                #         bit-identical to the (now target_aux-OFF)
                #         baseline.
                #       * Non-BCE path (xgb_distill, pnl_loss): Brier
                #         doesn't fire; those loss families are unchanged.
                #
                # Companion changes for clean isolation
                # -------------------------------------
                #   * use_target_aux default flipped True -> False (since
                #     #330 was just rejected).
                #   * daily_rank_active is suppressed when brier_active is
                #     True. The WF gate passes daily_rank_enabled=True;
                #     without this guard, the rejected pairwise (#303) /
                #     listnet (#310) rankers would silently activate
                #     alongside Brier, muddling the ablation.
                #   * All other rejected-mechanism opt-ins remain default
                #     OFF (R-Drop, GCE, label smoothing, SAM, etc.).
                #
                # Reference: Brier (1950). "Verification of forecasts
                # expressed in terms of probability." Monthly Weather
                # Review 78(1): 1-3.
                #
                # Brier-score loss (#331, REJECTED). Default flipped True ->
                # False so THIS ATTEMPT (#332) confident-learning label
                # cleaning is isolated from the just-rejected Brier
                # mechanism. Sweep can re-enable via use_brier_loss=True
                # for ablation.
                use_brier_loss=False,
                # +10% strong-winner threshold (default). Realized PnL is bounded by
                # the trading system's trailing-stop machinery — full +15% target
                # captures are exceedingly rare in the realized-PnL distribution
                # because the trailing stop at 50% of peak gain locks in profits
                # well below the +15% reward target. Empirically pnl in this dataset
                # tops out near +0.139, with pnl >= 0.10 occurring ~4.4% of the time
                # — ~3x rarer than the +4% main task (~10.8%) and a good auxiliary
                # signal: the +10% slice represents trades that actually rode
                # momentum hard, the kind of high-quality winner we want our
                # production threshold (0.6) to favor.
                target_aux_threshold=0.10,
                # Confident-learning label cleaning (#332, REJECTED). Default
                # flipped True -> False so THIS ATTEMPT (#334) window warping
                # is isolated from the just-rejected label-cleaning mechanism.
                # Sweep can re-enable via use_label_cleaning=True for ablation.
                use_label_cleaning=False,
                label_cleaning_n_folds=3,
                label_cleaning_noise_frac=0.10,
                # ===== WINDOW WARPING TIME-SERIES AUGMENTATION (THIS ATTEMPT #334) =====
                # Per-sequence temporal distortion: pick a random window inside
                # each sequence, resample at a different rate (speedup or
                # slowdown), then re-stretch the whole sequence to length T.
                # Listed as UNTRIED option (6) in the condensed lessons:
                #   "Time-series augmentations (jitter, scaling, time-warping,
                #    window-warping) instead of plain Gaussian noise."
                #
                # Theoretical basis: Um et al. 2017, "Data Augmentation of
                # Wearable Sensor Data for Parkinson's Disease Monitoring
                # using Convolutional Neural Networks" (NIPS-W). Window
                # warping was the most effective single-source augmentation
                # for time-series classification in their study.
                #
                # Why window warping attacks the cross-split-WR-variance plateau
                # -------------------------------------------------------------
                # Stocks move at variable paces across regimes. A momentum
                # signature that takes 4 days to play out in a fast regime
                # can stretch to 7 days in a slow regime. The LSTM trained
                # on raw sequences sees only ONE specific timing per pattern;
                # window warping exposes it to plausible alternate-pace
                # versions of the SAME pattern, forcing pace-invariant
                # features. Pace-invariance is structurally what should
                # transfer across the 7 WF splits, each with different
                # volatility and momentum dynamics.
                #
                # Why this is NOT one of the already-rejected mechanisms
                # ------------------------------------------------------
                #   * INPUT_NOISE_STD (always active in LSTMModel): IID
                #     Gaussian per-timestep. Window warping is STRUCTURED,
                #     SMOOTH, and TEMPORALLY COHERENT — different family.
                #   * Mixup (Entry 197 baseline, ACTIVE): convex combo of
                #     TWO sequences. Window warping operates on ONE sequence.
                #     They COMPOSE — warp first, then mixup the warped pair.
                #   * Manifold mixup (#328, REJECTED): mix at hidden layer.
                #     Multi-sample mechanism vs window warp's single-sample.
                #   * SupCon (#319, REJECTED): cross-symbol contrastive
                #     pretraining stage. Window warping runs INSIDE
                #     supervised training as on-the-fly per-batch aug.
                #   * Curriculum / boundary-band exclusion (#304, #317,
                #     REJECTED): hard SAMPLE filtering by PnL. Window
                #     warping keeps every sample.
                #   * Label smoothing / soft labels (#47, #72, #321,
                #     REJECTED): label-side mechanisms. Window warping is
                #     INPUT-side and label-independent.
                #   * NEVER tried in this project's 333-attempt history
                #     (verified by grep on the trainer source for
                #     'window_warp', 'time_warp', 'temporal_distort' —
                #     all returned no matches).
                #
                # Why this is SAFE
                # ----------------
                #   - Saved model has IDENTICAL signature to Entry 197.
                #     Gate loader, live trader, scaler are all unchanged.
                #     Production inference does NOT warp.
                #   - CURATED_FEATURES untouched. Operates on already-scaled
                #     (B, T, F) batches; feature semantics preserved exactly.
                #   - lstm_model.py untouched.
                #   - Compute cost: per-sample numpy interp; ~5ms per
                #     batch_size=256, T=20, F=19 batch. << 1% overhead.
                #   - Bounded distortion: scale_range=[0.5, 2.0],
                #     window_frac=0.2 (4 of 20 timesteps). Max temporal
                #     displacement ~3 timesteps — well within the LSTM's
                #     receptive field.
                #   - Per-sequence prob p=0.5 keeps half of each batch
                #     un-warped, so the un-augmented distribution is always
                #     represented.
                #   - Degrades gracefully: use_window_warp=False (sweep
                #     ablation) makes the path bit-identical to the
                #     (now-label-cleaning-OFF) Entry 197 baseline.
                #   - Active on every training path (BCE/Brier/PnL/distill)
                #     because warping runs at batch-prep time, before any
                #     path-specific code.
                #
                # Companion change for clean isolation
                # ------------------------------------
                #   * use_label_cleaning default flipped True -> False (#332
                #     just rejected).
                #   * daily_rank_active is suppressed when window_warp_active
                #     is True. The WF gate passes daily_rank_enabled=True;
                #     without this guard, the rejected pairwise (#303) /
                #     listnet (#310) rankers would silently activate
                #     alongside this attempt's mechanism, muddling the
                #     ablation.
                #   * All other rejected-mechanism opt-ins remain default
                #     OFF.
                use_window_warp=False,
                window_warp_p=0.5,
                window_warp_window_frac=0.2,
                window_warp_scale_low=0.5,
                window_warp_scale_high=2.0,
                # ===== SPECAUGMENT-STYLE TIME MASKING (THIS ATTEMPT #335) =====
                # Park et al. 2019 (Interspeech). Per-sequence stochastic
                # blanking of n_masks contiguous time windows of variable
                # size up to time_mask_size_frac * T. Default ON so the WF
                # gate (which doesn't pass these kwargs) picks up the
                # structural change automatically. See _time_mask_batch
                # docstring for the full rationale and comparisons to all
                # rejected mechanisms.
                use_time_mask=True,
                time_mask_p=0.5,
                time_mask_size_frac=0.15,
                time_mask_n_masks=2):
    """Train LSTM. Returns (model, scaler, metrics_dict).

    Two training paths:
      (A) PnL-optimization (active when pnl_train AND pnl_val provided):
          loss = -scale * mean(p * (pnl - commission))
                 + budget_lambda * (mean(p) - target_selectivity) ** 2
          Directly maximizes expected realized profit with a selectivity
          budget. Mixup mixes pnl alongside x.
      (B) BCE fallback (when pnl arrays absent): Entry 197 recipe —
          pos-weight class-balanced BCE with mixup.

    Model selection (both paths):
      - Track best val_loss each epoch (same loss family as training)
      - SWA: average last swa_window val_loss-improving snapshots
      - Final weights: SWA average if >=2 snapshots exist, else best state
      - Post-training FC-bias calibration so top target_selectivity of val
        propensities lands at LIVE_THRESHOLD=0.6 (#289)
    """
    device = DEVICE

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)

    if verbose:
        for i, feat in enumerate(features):
            col = X_train_scaled.reshape(-1, n_features)[:, i]
            p5, p95 = np.percentile(col, [5, 95])
            if (p95 - p5) < 0.05:
                print(f"  WARNING: '{feat}' compressed after scaling (5-95% range = {p95-p5:.4f})")

    # ===== CONFIDENT-LEARNING LABEL CLEANING (THIS ATTEMPT #332) =====
    # OOF XGBoost identifies likely-mislabeled training samples; the top
    # noise_frac is removed before any downstream computation. All
    # per-sample arrays (y, dates, pnl, early_sl) are filtered in
    # lockstep so derived arrays (covariate-shift weights, time-decay
    # weights, env IDs, soft labels) automatically reference the
    # cleaned dataset. See _label_cleaning_oof_xgb for full rationale.
    if use_label_cleaning:
        keep_mask_lc, label_cleaning_info = _label_cleaning_oof_xgb(
            X_train_scaled, y_train,
            n_folds=int(label_cleaning_n_folds),
            noise_frac=float(label_cleaning_noise_frac),
            verbose=verbose)
        if label_cleaning_info.get('applied', False):
            X_train_scaled = X_train_scaled[keep_mask_lc]
            y_train = np.asarray(y_train)[keep_mask_lc]
            if dates_train is not None:
                dates_train = np.asarray(dates_train)[keep_mask_lc]
            if pnl_train is not None:
                pnl_train = np.asarray(pnl_train)[keep_mask_lc]
            if early_sl_train is not None:
                early_sl_train = np.asarray(early_sl_train)[keep_mask_lc]
    else:
        label_cleaning_info = {'applied': False, 'reason': 'disabled by flag'}

    # ===== FEATURE-STATIONARITY KS-MASK (THIS ATTEMPT #326) =====
    # Compute the per-feature train/val KS distance and derive the
    # multiplicative downweighting mask. Apply it to the scaled inputs
    # AND bake it into the scaler so test/live inference automatically
    # produces the same masked feature space (no train/inference skew).
    # See _compute_stationarity_mask docstring for the full rationale.
    if use_stationarity_mask:
        stationarity_mask, stationarity_info = _compute_stationarity_mask(
            X_train_scaled, X_val_scaled, features,
            alpha=float(stationarity_alpha),
            floor=float(stationarity_floor),
            verbose=verbose)
        if stationarity_info.get('applied'):
            mask_3d = stationarity_mask.reshape(1, 1, -1).astype(np.float32)
            X_train_scaled = (X_train_scaled * mask_3d).astype(np.float32)
            X_val_scaled = (X_val_scaled * mask_3d).astype(np.float32)
            # Bake into scaler. RobustScaler.transform returns
            # (x - center_) / scale_; we want it to return
            # mask * (x - center_) / scale_ at inference, which means
            # the new scale_ should be (old scale_) / mask.
            # Guard against zero in mask via the floor (>=0.3 by default).
            scaler.scale_ = (scaler.scale_ / stationarity_mask).astype(scaler.scale_.dtype)
    else:
        stationarity_info = {'applied': False, 'reason': 'disabled by flag'}

    torch.manual_seed(42)
    np.random.seed(42)

    model = LSTMModel(input_size=n_features, hidden_size=hidden_size,
                      num_layers=num_layers, dropout=dropout,
                      use_attention=use_attention,
                      sequence_normalize=sequence_normalize).to(device)

    # Stage 0a: cross-symbol Supervised Contrastive (SupCon) pretraining of
    # the LSTM backbone. THIS ATTEMPT #319. Pulls same-day, same-label
    # embeddings together; pushes opposite-label apart. Forces a regime-
    # invariant representation before the noisy 11%-positive BCE phase
    # starts. See _pretrain_lstm_supcon for full rationale.
    if use_supcon_pretrain:
        supcon_info = _pretrain_lstm_supcon(
            model, X_train_scaled, y_train, dates_train,
            epochs=supcon_epochs,
            batch_size=batch_size,
            lr=supcon_lr,
            temperature=supcon_temperature,
            projection_dim=supcon_projection_dim,
            min_per_day=supcon_min_per_day,
            max_per_day=supcon_max_per_day,
            verbose=verbose)
    else:
        supcon_info = {'applied': False, 'reason': 'disabled by flag'}

    # Stage 0b: self-supervised pretraining of the LSTM backbone. Runs BEFORE
    # supervised training so the classifier fine-tunes from a pretrained
    # temporal encoder, not from random init. See _pretrain_lstm_backbone.
    if use_ss_pretrain:
        ss_pretrain_info = _pretrain_lstm_backbone(
            model, [X_train_scaled, X_val_scaled],
            epochs=ss_pretrain_epochs,
            batch_size=batch_size,
            lr=ss_pretrain_lr,
            verbose=verbose)
    else:
        ss_pretrain_info = {'applied': False, 'reason': 'disabled by flag'}

    pos_rate = y_train.mean()
    if pos_rate == 0 or pos_rate == 1:
        return None, None, {'error': 'degenerate labels'}

    pos_weight = float(min((1.0 - pos_rate) / max(pos_rate, 1e-6), 10.0))

    # Decide training path. XGB distillation takes precedence when enabled,
    # overriding any pnl_train/pnl_val that may have been passed.
    xgb_train_soft = None
    xgb_val_soft = None
    if use_xgb_distill:
        xgb_train_soft, xgb_val_soft, _ = _train_xgb_teacher(
            X_train_scaled, y_train, X_val_scaled, y_val, verbose=verbose)
        if xgb_train_soft is None:
            use_xgb_distill = False  # teacher unavailable → fall back

    # PnL path only runs when explicitly opted in — the WF gate passes
    # pnl_train/pnl_val unconditionally, so requiring use_pnl_loss=True
    # keeps BCE (Entry 197 recipe) as the default path.
    use_pnl_loss = (use_pnl_loss and (not use_xgb_distill)
                    and (pnl_train is not None) and (pnl_val is not None))

    # BCE path gates the two domain-adaptation strategies. DANN takes
    # precedence — it and the density-ratio weights address the same
    # pathology (train/val distribution shift) and stacking them would
    # apply the correction twice.
    bce_path_active = (not use_xgb_distill) and (not use_pnl_loss)
    dann_active = bool(use_dann) and bce_path_active

    # PnL-distance soft labels (THIS ATTEMPT #321). Active only on the BCE
    # path AND when caller supplied pnl_train. The WF gate passes pnl_train
    # unconditionally, so this picks up automatically. Computes y_soft from
    # PnL up front; the binary y is still used for class re-balancing
    # (pos_weight) and for daily_rank's pair construction. The BCE TARGET
    # becomes y_soft instead of y.
    pnl_smooth_active = (bool(use_pnl_smooth_labels) and bce_path_active
                         and (pnl_train is not None)
                         and (len(np.asarray(pnl_train)) == len(X_train_scaled)))
    if pnl_smooth_active:
        y_train_soft_np = _pnl_smooth_labels(
            np.asarray(pnl_train, dtype=np.float32),
            threshold=MIN_PROFIT_PCT,
            scale=float(pnl_smooth_scale))
        # Diagnostic stats: how much of train sits in the "uncertain" band
        # (y_soft in [0.3, 0.7]) — that's the chunk that gets gradient
        # attenuated. Below 5% means smoothing is essentially transparent;
        # above 30% means we're aggressively softening labels.
        uncertain_band = np.logical_and(y_train_soft_np >= 0.3,
                                         y_train_soft_np <= 0.7)
        clear_win_band = y_train_soft_np >= 0.9
        clear_loss_band = y_train_soft_np <= 0.1
        # Cross-check: how does the soft-label MEAN compare to the binary
        # positive rate? Big divergence (>10pp) means the threshold is
        # mis-calibrated relative to the smooth distribution.
        pnl_smooth_info = {
            'applied': True,
            'threshold': float(MIN_PROFIT_PCT),
            'scale': float(pnl_smooth_scale),
            'binary_pos_rate': float(np.asarray(y_train).mean()),
            'soft_target_mean': float(y_train_soft_np.mean()),
            'frac_uncertain_0.3_0.7': float(uncertain_band.mean()),
            'frac_clear_win_>=0.9': float(clear_win_band.mean()),
            'frac_clear_loss_<=0.1': float(clear_loss_band.mean()),
            'soft_min': float(y_train_soft_np.min()),
            'soft_max': float(y_train_soft_np.max()),
            'soft_median': float(np.median(y_train_soft_np)),
        }
        if verbose:
            print(f'  PnL-distance soft labels: threshold={MIN_PROFIT_PCT:.2%}, '
                  f'scale={pnl_smooth_scale:.4f}')
            print(f'    Binary pos_rate={pnl_smooth_info["binary_pos_rate"]:.3f}, '
                  f'soft target mean={pnl_smooth_info["soft_target_mean"]:.3f} '
                  f'(diff={pnl_smooth_info["soft_target_mean"]-pnl_smooth_info["binary_pos_rate"]:+.3f})')
            print(f'    Uncertain band [0.3, 0.7]: {pnl_smooth_info["frac_uncertain_0.3_0.7"]:.1%} of train, '
                  f'clear-win >=0.9: {pnl_smooth_info["frac_clear_win_>=0.9"]:.1%}, '
                  f'clear-loss <=0.1: {pnl_smooth_info["frac_clear_loss_<=0.1"]:.1%}')
    else:
        y_train_soft_np = None
        pnl_smooth_info = {
            'applied': False,
            'reason': ('disabled by flag' if not use_pnl_smooth_labels
                       else ('non-BCE path' if not bce_path_active
                             else ('no pnl_train' if pnl_train is None
                                   else 'pnl_train length mismatch'))),
        }

    if use_covariate_shift and bce_path_active and not dann_active:
        shift_weights_np, shift_info = _compute_covariate_shift_weights(
            X_train_scaled, X_val_scaled, verbose=verbose)
    else:
        shift_weights_np = np.ones(len(X_train_scaled), dtype=np.float32)
        shift_info = {'applied': False,
                      'reason': ('superseded by DANN' if dann_active
                                 else 'disabled by flag or non-BCE path')}

    # Exponential CHRONOLOGICAL time-decay sample weighting (THIS ATTEMPT #316).
    # Multiplied into shift_weights_np so it composes with any covariate-shift
    # weights (which are off by default) and feeds through the existing
    # per-sample BCE pipeline unchanged. The two are conceptually orthogonal:
    # covariate-shift uses LEARNED FEATURE-DENSITY ratios (rejected #295);
    # time-decay uses CALENDAR POSITION directly (this attempt). When both are
    # active, weights compose multiplicatively and are re-mean-normalized
    # together so total loss magnitude stays at unity.
    td_active = (bool(use_time_decay) and bce_path_active
                 and (dates_train is not None) and len(np.asarray(dates_train)) == len(X_train_scaled))
    if td_active:
        td_weights_np, td_info = _compute_time_decay_weights(
            dates_train, decay=float(time_decay_factor), verbose=verbose)
        if td_info.get('applied'):
            combined = shift_weights_np * td_weights_np
            combined = combined * (len(combined) / max(combined.sum(), 1e-9))
            shift_weights_np = combined.astype(np.float32)
        else:
            td_active = False
    else:
        td_info = {
            'applied': False,
            'reason': ('disabled by flag' if not use_time_decay
                       else ('non-BCE path' if not bce_path_active
                             else ('no dates_train' if dates_train is None
                                   else 'dates_train length mismatch'))),
        }

    # Domain classifier head — trained jointly with the LSTM. Sees pre-FC
    # features through a gradient-reversal layer; its gradient is negated
    # (and scaled by alpha) before reaching the LSTM, so the LSTM learns
    # to make train and val indistinguishable while the classifier does
    # its best to tell them apart. Not saved in the LSTMModel state_dict
    # (it's training-only auxiliary machinery).
    domain_clf = None
    dann_info = {'applied': False, 'reason': 'not active'}
    if dann_active:
        domain_clf = _DomainClassifier(hidden_size, dropout=max(0.1, dropout * 0.5)).to(device)
        # Val sample pool for adversarial pairing. Rebuilt each epoch at
        # batch time; we hold the full tensor on device to avoid repeated
        # host→device copies inside the tight inner loop.
        X_val_dev = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
        dann_info = {
            'applied': True,
            'lambda_max': float(dann_lambda_max),
            'gamma': float(dann_gamma),
            'val_pool_size': len(X_val_scaled),
        }

    # Mean Teacher setup. Active only on the BCE path (the default, and the
    # only path kept on by default). The teacher is a deep copy of the
    # freshly initialized student — since both start identical, the early
    # consistency signal is near-zero (λ ramp-up further suppresses early-
    # training pressure), and the EMA gradually diverges from the student
    # as SGD updates arrive. Teacher params are frozen (no_grad) and the
    # teacher is held in eval() so it sees clean inputs (no input_dropout,
    # no INPUT_NOISE, no hidden_noise) — canonical Mean Teacher design.
    mt_active = bool(use_mean_teacher) and bce_path_active and (not dann_active)
    mt_info = {
        'applied': False,
        'reason': ('disabled by flag' if not use_mean_teacher
                   else ('non-BCE path' if not bce_path_active
                         else 'superseded by DANN' if dann_active else 'n/a')),
    }
    if mt_active:
        import copy as _copy
        model_ema = _copy.deepcopy(model).to(device)
        for _p in model_ema.parameters():
            _p.requires_grad_(False)
        model_ema.eval()
        mt_info = {
            'applied': True,
            'ema_decay': float(mt_ema_decay),
            'lambda_max': float(mt_lambda_max),
            'rampup_epochs': int(mt_rampup_epochs),
        }
        if verbose:
            print(f'  Mean Teacher: ema_decay={mt_ema_decay}, '
                  f'lambda_max={mt_lambda_max}, rampup_epochs={mt_rampup_epochs}')
    else:
        model_ema = None

    # MULTI-FOLD VAL SELECTION (THIS ATTEMPT). Decided here so we can
    # disable ranking when multifold is on, isolating the structural
    # change. Active only on the BCE path; non-BCE paths (xgb_distill,
    # pnl_loss) override the loss family entirely and don't compose
    # cleanly with this selection metric.
    multifold_active = bool(use_multifold_select) and bce_path_active

    # Daily pairwise-ranking activation. Requires BCE path, daily_rank_enabled,
    # dates_train. WF gate satisfies all three by default. ALSO requires:
    # (a) multifold OFF (#311), and (b) SelectiveNet OFF (this attempt #312).
    # Without these guards, the WF gate's daily_rank_enabled=True kwarg would
    # silently re-introduce the rejected pairwise/listnet ranking mechanisms
    # alongside whatever new structural change is being evaluated, muddling
    # the ablation. The current default config (multifold off + selective on)
    # therefore disables ranking — explicit override required to re-enable.
    # NOTE: `_sel_will_activate` is a forward-reference flag — the actual
    # SelectionHead is constructed later (after qr_info), but the activation
    # condition is identical and computed here cheaply.
    _sel_will_activate = bool(use_selective_head) and bce_path_active
    # Sortino aux activation (#322, REJECTED — now default OFF).
    sortino_active = (bool(use_sortino_aux) and bce_path_active
                      and (pnl_train is not None)
                      and (len(np.asarray(pnl_train)) == len(X_train_scaled)))
    # ANNEALED GRADIENT NOISE INJECTION (THIS ATTEMPT #323).
    # Active only on the BCE path. Each parameter's gradient is perturbed by
    # Gaussian noise xi_t ~ N(0, sigma_t^2 I) with sigma_t annealed
    # polynomially in step count post-warmup (Neelakantan et al. 2015).
    # See module docstring for the FLAT-MINIMUM-BIAS, EPOCH-0-ESCAPE, and
    # BAYESIAN-POSTERIOR-APPROXIMATION mechanisms by which this attacks
    # the 322-attempt plateau and the cross-split-WR-variance failure mode.
    # Computed here so we can disable daily_rank when grad_noise is active
    # — keeps THIS ATTEMPT isolated from the rejected pairwise (#303) /
    # listnet (#310) rankers that the WF gate's daily_rank_enabled=True
    # kwarg would otherwise silently re-enable.
    grad_noise_active = bool(use_grad_noise) and bce_path_active
    # R-Drop activation (THIS ATTEMPT #325). Active only on BCE path AND
    # when dropout > 0 (otherwise the two forward passes are identical and
    # the KL is identically zero — no gradient signal). The float() guard
    # against dropout==0 is critical: a 0-dropout model with R-Drop on
    # would still incur the 2x forward-pass cost without any regularization
    # benefit.
    r_drop_active = (bool(use_r_drop) and bce_path_active
                     and float(dropout) > 0.0)
    # FEATURE-STATIONARITY KS-MASK active flag (THIS ATTEMPT #326). Mirrors
    # the activation logic of _compute_stationarity_mask above — used here
    # only to suppress daily_rank when this attempt's mechanism fired, so
    # the rejected pairwise (#303) / listnet (#310) rankers don't silently
    # activate alongside it via the WF gate's daily_rank_enabled=True kwarg.
    stationarity_active = (bool(use_stationarity_mask)
                           and stationarity_info.get('applied', False))
    # Temperature scaling activation flag (THIS ATTEMPT #327). Mirrors
    # the precondition logic of _temperature_scale — used here only to
    # suppress daily_rank when this attempt's mechanism is the active
    # default, so the rejected pairwise (#303) / listnet (#310) rankers
    # don't silently activate alongside it via the WF gate's
    # daily_rank_enabled=True kwarg.
    temp_scaling_active = (bool(use_temperature_scaling) and bce_path_active
                           and len(X_val_scaled) >= MIN_VAL_FOR_CALIBRATION
                           and y_val is not None)
    # MANIFOLD MIXUP activation flag (THIS ATTEMPT #328). Active only on
    # BCE path AND when no other advanced mechanism is fighting for the
    # mixup slot — DANN, qr_aux, SelectiveNet, R-Drop all rely on either
    # an extracted-feature forward path (which conflicts with manifold
    # mixup's own feature extraction) or a second forward pass per batch
    # (which would double the manifold-mixup cost). Mutual exclusion
    # keeps the ablation clean. Mixup_alpha must be > 0 — otherwise the
    # mix is degenerate (lam=1) and there's nothing to mix.
    manifold_mixup_active = (bool(use_manifold_mixup) and bce_path_active
                              and float(mixup_alpha) > 0.0
                              and not bool(use_dann)
                              and not bool(use_quantile_aux)
                              and not bool(use_selective_head)
                              and not bool(use_r_drop)
                              and not bool(use_temporal_consist)
                              and not bool(use_mean_teacher)
                              and not bool(use_sortino_aux)
                              and not bool(use_grad_noise))
    # ===== DUAL-TARGET MULTI-TASK AUX HEAD activation flag (THIS ATTEMPT #330)
    # Active only on BCE path AND when caller supplied pnl_train (the WF gate
    # always passes pnl_train). Composes with feat_shared extraction (which
    # is already triggered by qr_active / sel_active) — see the head-build
    # block below and the loss-composition block in the training loop.
    target_aux_active = (bool(use_target_aux) and bce_path_active
                         and (pnl_train is not None)
                         and (len(np.asarray(pnl_train)) == len(X_train_scaled)))
    # Build the binary auxiliary target up front so the per-batch loop can
    # slice it in lockstep with X_train. y_target_train_np[i] = 1 iff sample
    # i hit the trading system's reward target (+15% by default), 0 otherwise.
    # Skipped if active flag flipped off, which happens when label
    # distribution is degenerate (no positives or all positives) — in that
    # case the aux loss would have no useful gradient.
    if target_aux_active:
        y_target_train_np = (
            np.asarray(pnl_train, dtype=np.float32)
            >= float(target_aux_threshold)).astype(np.float32)
        target_pos_rate = float(y_target_train_np.mean())
        if target_pos_rate <= 0.0 or target_pos_rate >= 1.0:
            target_aux_active = False
            target_pos_weight = 1.0
            target_aux_info = {
                'applied': False,
                'reason': f'degenerate target labels (rate={target_pos_rate:.4f})',
            }
        else:
            # Cap aux pos_weight at 15 (vs main task's cap of 10) — the +15%
            # base rate is typically ~3-5% in the SET data, ~3x rarer than
            # the +4% main task. A higher cap is safe because the aux task's
            # gradient enters the main loss multiplied by target_aux_weight
            # (default 0.3), so even at the cap the effective per-sample
            # gradient is < 5x the main task's positive gradient.
            target_pos_weight = float(min(
                (1.0 - target_pos_rate) / max(target_pos_rate, 1e-6), 15.0))
            target_aux_info = {
                'applied': True,
                'threshold': float(target_aux_threshold),
                'aux_weight': float(target_aux_weight),
                'pos_rate': target_pos_rate,
                'pos_weight': target_pos_weight,
                'main_pos_rate': float(np.asarray(y_train).mean()),
                'n_target_winners': int(y_target_train_np.sum()),
                'n_train': int(len(y_target_train_np)),
            }
            if verbose:
                print(f'  Dual-target aux head (THIS ATTEMPT #330): '
                      f'threshold={target_aux_threshold:.3f}, '
                      f'aux_weight={target_aux_weight}')
                print(f'    Aux target pos_rate={target_pos_rate:.3f} '
                      f'(n={int(y_target_train_np.sum())}/{len(y_target_train_np)}), '
                      f'aux pos_weight={target_pos_weight:.2f}')
                print(f'    Main task pos_rate='
                      f'{float(np.asarray(y_train).mean()):.3f} '
                      f'@ +4% — aux head adds gradient toward features that '
                      f'predict the trading system\'s ACTUAL reward target.')
    else:
        y_target_train_np = None
        target_pos_weight = 1.0
        target_aux_info = {
            'applied': False,
            'reason': ('disabled by flag' if not use_target_aux
                       else ('non-BCE path' if not bce_path_active
                             else ('no pnl_train' if pnl_train is None
                                   else 'pnl_train length mismatch'))),
        }
    # ===== BRIER-SCORE LOSS activation flag (THIS ATTEMPT #331) =====
    # Active only on the BCE path. When active, replaces the per-sample
    # BCE / GCE loss with MSE on probabilities (Brier score). Composes
    # unchanged with class_weight, shift_weights, mixup, PnL-distance
    # soft labels, and all auxiliary losses (R-Drop, daily_rank, etc.).
    # See the use_brier_loss docstring for the full rationale.
    brier_active = bool(use_brier_loss) and bce_path_active
    brier_info = {
        'applied': bool(brier_active),
        'reason': ('active' if brier_active
                   else ('disabled by flag' if not use_brier_loss
                         else 'non-BCE path')),
    }
    if brier_active and verbose:
        print(f'  Brier-score loss (THIS ATTEMPT #331): MSE on probabilities')
        print(f'    per_sample = (pred - target)^2 instead of BCE/GCE.')
        print(f'    Bounded gradient |dL/dp| = 2|p-y| <= 2; strictly proper'
              f' scoring rule.')

    # ===== WINDOW WARPING activation flag (THIS ATTEMPT #334) =====
    # Active on every training path (BCE / GCE / Brier / PnL / distill)
    # because warping happens at batch-prep time, before any
    # path-specific code reads X_batch. Independent of which loss family
    # is active. Composes cleanly with mixup: warped sequences feed into
    # the standard mixup pair-construction.
    window_warp_active = bool(use_window_warp)
    window_warp_info = {
        'applied': bool(window_warp_active),
        'reason': ('active' if window_warp_active
                   else 'disabled by flag'),
        'p': float(window_warp_p) if window_warp_active else None,
        'window_frac': float(window_warp_window_frac) if window_warp_active else None,
        'scale_low': float(window_warp_scale_low) if window_warp_active else None,
        'scale_high': float(window_warp_scale_high) if window_warp_active else None,
    }
    if window_warp_active and verbose:
        print(f'  Window warping (THIS ATTEMPT #334): p={window_warp_p}, '
              f'window_frac={window_warp_window_frac}, '
              f'scale_range=[{window_warp_scale_low}, {window_warp_scale_high}]')
        print(f'    Per-sequence: pick random window, resample at scale ~U[low,high], '
              f'splice + re-stretch to T.')
        print(f'    Pace-invariant features → cross-regime generalization.')
    # Per-batch RNG for window warping; seeded so run-to-run variance is
    # bounded and composes with the global torch.manual_seed(42).
    window_warp_rng = np.random.RandomState(20260429) if window_warp_active else None

    # ===== SPECAUGMENT-STYLE TIME MASKING activation flag (THIS ATTEMPT #335)
    # Active on every training path (BCE / GCE / Brier / PnL / distill)
    # because masking happens at batch-prep time, before any path-specific
    # code reads X_batch. Independent of which loss family is active.
    # Composes cleanly with mixup: masked sequences feed into the standard
    # mixup pair-construction.
    time_mask_active = bool(use_time_mask)
    time_mask_info = {
        'applied': bool(time_mask_active),
        'reason': ('active' if time_mask_active else 'disabled by flag'),
        'p': float(time_mask_p) if time_mask_active else None,
        'mask_size_frac': float(time_mask_size_frac) if time_mask_active else None,
        'n_masks': int(time_mask_n_masks) if time_mask_active else None,
    }
    if time_mask_active and verbose:
        print(f'  Time masking (THIS ATTEMPT #335): p={time_mask_p}, '
              f'mask_size_frac={time_mask_size_frac}, '
              f'n_masks={time_mask_n_masks}')
        print(f'    Per-sequence: with prob {time_mask_p} blank '
              f'{time_mask_n_masks} random contiguous time-windows of size up to '
              f'{int(round(20 * time_mask_size_frac))} of seq_len.')
        print(f'    SpecAugment (Park et al. 2019). Forces position-invariant '
              f'temporal features → cross-regime generalization.')
    # Per-batch RNG for time masking; seeded so run-to-run variance is
    # bounded and composes with the global torch.manual_seed(42).
    time_mask_rng = np.random.RandomState(20260503) if time_mask_active else None

    daily_rank_active = (bool(daily_rank_enabled) and bce_path_active
                         and (dates_train is not None)
                         and (not multifold_active)
                         and (not _sel_will_activate)
                         and (not td_active)
                         and (not pnl_smooth_active)
                         and (not sortino_active)
                         and (not grad_noise_active)
                         and (not r_drop_active)
                         and (not stationarity_active)
                         and (not temp_scaling_active)
                         and (not manifold_mixup_active)
                         and (not target_aux_active)
                         and (not brier_active)
                         and (not label_cleaning_info.get('applied', False))
                         and (not window_warp_active)
                         and (not time_mask_active))
    listnet_active = False
    # MANIFOLD MIXUP info dict (THIS ATTEMPT #328).
    manifold_mixup_info = {
        'applied': bool(manifold_mixup_active),
        'reason': ('active' if manifold_mixup_active
                   else ('disabled by flag' if not use_manifold_mixup
                         else ('non-BCE path' if not bce_path_active
                               else ('mixup_alpha=0 (no-op)'
                                     if float(mixup_alpha) <= 0.0
                                     else 'superseded by other advanced mechanism')))),
        'mixup_alpha': float(mixup_alpha),
        'manifold_mixup_p': float(manifold_mixup_p) if manifold_mixup_active else None,
    }
    if manifold_mixup_active and verbose:
        print(f'  Manifold Mixup (THIS ATTEMPT #328): mixup_alpha={mixup_alpha}, '
              f'manifold_mixup_p={manifold_mixup_p}')
        print(f'    Per-batch coin flip: with prob {manifold_mixup_p} mix at '
              f'pre-FC pooled hidden layer, otherwise input mixup (Entry 197).')
        print(f'    No new parameters, no second forward pass, no extra compute.')

    # Per-batch coin-flip RNG for manifold mixup. Seeded so run-to-run
    # variance is bounded; composes with the global torch.manual_seed(42)
    # for the LSTM init.
    manifold_mixup_rng = np.random.RandomState(20260428)

    daily_rank_info = {
        'applied': False,
        'reason': ('superseded by SpecAugment-style time masking (this attempt #335)'
                   if time_mask_active else
                   ('superseded by window warping (rejected #334, default OFF)'
                   if window_warp_active else
                   ('superseded by confident-learning label cleaning (rejected #332, default OFF)'
                   if label_cleaning_info.get('applied', False) else
                   ('superseded by Brier-score loss (rejected #331, default OFF)'
                   if brier_active else
                   ('superseded by dual-target aux head'
                   if target_aux_active else
                   ('superseded by Manifold Mixup'
                   if manifold_mixup_active else
                   ('superseded by post-SWA temperature scaling'
                   if temp_scaling_active else
                   ('superseded by feature-stationarity KS-mask'
                   if stationarity_active else
                   ('superseded by R-Drop regularized dropout'
                   if r_drop_active else
                   ('superseded by annealed gradient noise'
                   if grad_noise_active else
                   ('superseded by Sortino aux loss'
                    if sortino_active else
                   ('superseded by PnL-distance soft labels'
                    if pnl_smooth_active else
                   ('superseded by time-decay sample weighting' if td_active
                   else ('superseded by SelectiveNet abstain head' if _sel_will_activate
                         else ('superseded by multifold val select' if multifold_active
                               else ('non-BCE path' if not bce_path_active
                                     else ('disabled by flag' if not daily_rank_enabled
                                           else ('no dates_train' if dates_train is None else 'n/a')))))))))))))))))),
    }
    # R-Drop info dict (THIS ATTEMPT #325).
    r_drop_info = {
        'applied': bool(r_drop_active),
        'reason': ('active' if r_drop_active
                   else ('disabled by flag' if not use_r_drop
                         else ('non-BCE path' if not bce_path_active
                               else 'dropout=0 (no-op)'))),
    }
    if r_drop_active:
        r_drop_info.update({
            'alpha': float(r_drop_alpha),
            'rampup_epochs': int(r_drop_rampup_epochs),
            'dropout': float(dropout),
        })
        if verbose:
            print(f'  R-Drop: alpha={r_drop_alpha}, '
                  f'rampup_epochs={r_drop_rampup_epochs}, '
                  f'model_dropout={dropout}')
            print('    L = 0.5*(BCE(p1,y) + BCE(p2,y)) + alpha*sym_KL(p1,p2)')
            print('    Two forward passes per step in train() mode '
                  '(different dropout masks).')
    date_ids_train_np = None
    if daily_rank_active:
        # Map dates_train (strings or datetimes) to dense integer IDs.
        dates_arr = np.asarray(dates_train)
        unique_dates_tr, date_ids_train_np = np.unique(dates_arr, return_inverse=True)
        date_ids_train_np = date_ids_train_np.astype(np.int64)
        # Sanity: need >=2 unique dates AND enough samples per date on average.
        n_unique = len(unique_dates_tr)
        avg_per_day = len(dates_arr) / max(n_unique, 1)
        if n_unique < 2 or avg_per_day < float(daily_rank_min_per_day):
            daily_rank_active = False
            daily_rank_info = {
                'applied': False,
                'reason': (f'too few dates ({n_unique}) or sparse '
                           f'({avg_per_day:.1f}/day < min {daily_rank_min_per_day})'),
            }
            date_ids_train_np = None
        else:
            listnet_active = (bool(use_listnet_rank) and (pnl_train is not None)
                              and len(np.asarray(pnl_train)) == len(X_train_scaled))
            daily_rank_info = {
                'applied': True,
                'variant': 'listnet_topk' if listnet_active else 'pairwise_margin_hinge',
                'lambda': float(daily_rank_lambda),
                'min_per_day': int(daily_rank_min_per_day),
                'margin': float(daily_rank_margin),
                'listnet_top_k_frac': float(listnet_top_k_frac) if listnet_active else None,
                'listnet_temperature': float(listnet_temperature) if listnet_active else None,
                'n_unique_dates': int(n_unique),
                'avg_samples_per_date': float(avg_per_day),
            }
            if verbose:
                if listnet_active:
                    print(f'  Daily ranking loss: LISTNET TOP-K '
                          f'(lambda={daily_rank_lambda}, k_frac={listnet_top_k_frac}, '
                          f'T={listnet_temperature}, min_per_day={daily_rank_min_per_day}, '
                          f'unique_dates={n_unique}, avg_samples/day={avg_per_day:.1f})')
                else:
                    print(f'  Daily ranking loss: PAIRWISE MARGIN-HINGE '
                          f'(lambda={daily_rank_lambda}, margin={daily_rank_margin}, '
                          f'min_per_day={daily_rank_min_per_day}, '
                          f'unique_dates={n_unique}, avg_samples/day={avg_per_day:.1f})')

    # ===== Sortino aux info dict (#322, REJECTED) =====
    sortino_info = {
        'applied': bool(sortino_active),
        'reason': ('active' if sortino_active
                   else ('disabled by flag' if not use_sortino_aux
                         else ('non-BCE path' if not bce_path_active
                               else ('no pnl_train' if pnl_train is None
                                     else 'pnl_train length mismatch')))),
    }
    if sortino_active:
        sortino_info.update({
            'aux_weight': float(sortino_aux_weight),
            'target_selectivity': float(sortino_target_selectivity),
            'budget_lambda': float(sortino_budget_lambda),
            'commission': float(sortino_commission),
            'eps': float(sortino_eps),
            'warmup_epochs': int(sortino_warmup_epochs),
        })
        if verbose:
            print(f'  Sortino aux loss: weight={sortino_aux_weight}, '
                  f'target_sel={sortino_target_selectivity}, '
                  f'budget_lambda={sortino_budget_lambda}, '
                  f'commission={sortino_commission}, '
                  f'warmup={sortino_warmup_epochs} epochs')
            print('    L_total = BCE + sortino_aux_weight * (-Sortino + budget_penalty)')

    # ===== Annealed Gradient Noise info dict (THIS ATTEMPT #323) =====
    grad_noise_info = {
        'applied': bool(grad_noise_active),
        'reason': ('active' if grad_noise_active
                   else ('disabled by flag' if not use_grad_noise
                         else 'non-BCE path')),
    }
    if grad_noise_active:
        grad_noise_info.update({
            'eta': float(grad_noise_eta),
            'gamma': float(grad_noise_gamma),
            'warmup_steps': int(grad_noise_warmup_steps),
        })
        if verbose:
            sigma_at_1 = float(grad_noise_eta) / max(
                1.0, (1.0 + 1.0) ** float(grad_noise_gamma))
            sigma_at_1000 = float(grad_noise_eta) / max(
                1.0, (1.0 + 1000.0) ** float(grad_noise_gamma))
            print(f'  Annealed gradient noise: eta={grad_noise_eta}, '
                  f'gamma={grad_noise_gamma}, '
                  f'warmup_steps={grad_noise_warmup_steps}')
            print(f'    sigma decay: step=1 -> {sigma_at_1:.5f}, '
                  f'step=1000 -> {sigma_at_1000:.5f}, polynomial 1/(1+t)^gamma')
            print('    Mechanism: post-warmup, every parameter gradient gets '
                  'N(0, sigma_t^2 I) added before optimizer.step()')

    # ===== IRM environment partitioning (THIS ATTEMPT #320) =====
    # Build a chronological K-quartile partition of unique train dates,
    # then map each train sample to its environment ID. This is computed
    # independently of date_ids_train_np (which only exists when daily_rank
    # is active) so IRM can run with daily ranking disabled.
    irm_active = (bool(use_irm) and bce_path_active
                  and (dates_train is not None)
                  and (len(np.asarray(dates_train)) == len(X_train_scaled)))
    env_ids_train_np = None
    irm_info = {
        'applied': False,
        'reason': ('disabled by flag' if not use_irm
                   else ('non-BCE path' if not bce_path_active
                         else ('no dates_train' if dates_train is None
                               else 'dates_train length mismatch'))),
    }
    if irm_active:
        dates_arr_irm = np.asarray(dates_train)
        unique_sorted = np.sort(np.unique(dates_arr_irm))
        n_unique_dates = len(unique_sorted)
        K_envs = max(2, int(irm_n_envs))
        if n_unique_dates < K_envs:
            irm_active = False
            irm_info = {
                'applied': False,
                'reason': f'too few unique dates ({n_unique_dates}) for K={K_envs} envs',
            }
        else:
            # Chronological partition: each env gets a contiguous date range.
            # Last env absorbs any remainder (handled by min(.,K-1)).
            n_per = max(1, n_unique_dates // K_envs)
            date_to_env = {}
            for i, d in enumerate(unique_sorted):
                date_to_env[d] = min(i // n_per, K_envs - 1)
            env_ids_train_np = np.array(
                [date_to_env[d] for d in dates_arr_irm], dtype=np.int64)
            env_counts = np.bincount(env_ids_train_np, minlength=K_envs)
            min_env_count = int(env_counts.min()) if len(env_counts) else 0
            if min_env_count < int(irm_min_per_env):
                irm_active = False
                env_ids_train_np = None
                irm_info = {
                    'applied': False,
                    'reason': (f'min env size {min_env_count} < '
                               f'irm_min_per_env={int(irm_min_per_env)}'),
                    'env_counts': env_counts.tolist(),
                }
            else:
                # Per-env date-range diagnostics (oldest:newest) help confirm
                # the partition reflects real temporal separation. Sort
                # rather than np.min/max so this works with string-dtype
                # date arrays (e.g., "YYYY-MM-DD" in candles.db).
                env_date_ranges = []
                for e in range(K_envs):
                    e_idx = np.where(env_ids_train_np == e)[0]
                    if len(e_idx) > 0:
                        e_dates_sorted = np.sort(dates_arr_irm[e_idx])
                        env_date_ranges.append(
                            (str(e_dates_sorted[0])[:10],
                             str(e_dates_sorted[-1])[:10]))
                    else:
                        env_date_ranges.append(None)
                irm_info = {
                    'applied': True,
                    'n_envs': K_envs,
                    'env_counts': env_counts.tolist(),
                    'env_date_ranges': env_date_ranges,
                    'lambda_max': float(irm_lambda_max),
                    'warmup_epochs': int(irm_warmup_epochs),
                    'min_per_env': int(irm_min_per_env),
                    'n_unique_dates': int(n_unique_dates),
                }
                if verbose:
                    print(f'  IRM (Invariant Risk Minimization): K={K_envs} '
                          f'chronological envs, sizes={env_counts.tolist()}, '
                          f'lambda_max={irm_lambda_max}, '
                          f'warmup={irm_warmup_epochs} epochs')
                    for e, rng in enumerate(env_date_ranges):
                        if rng is not None:
                            print(f'    env {e}: {rng[0]} → {rng[1]} '
                                  f'({env_counts[e]} samples)')

    # Disable daily-rank when IRM is active to isolate the structural
    # change. Pairwise (#303) and listnet (#310) ranking variants both
    # failed; running them alongside IRM would muddle the ablation.
    if irm_active and daily_rank_active:
        daily_rank_active = False
        daily_rank_info = {
            'applied': False,
            'reason': 'superseded by IRM (this attempt)',
        }
        date_ids_train_np = None
        if verbose:
            print('    (daily-rank disabled — IRM is the active mechanism)')

    if use_xgb_distill:
        # y_soft = α * xgb_prob + (1-α) * y_hard.
        # Small hard-label anchor (1-α=0.3) keeps the LSTM from drifting if
        # the XGB teacher is biased on a particular split.
        y_train_soft = np.clip(
            distill_alpha * xgb_train_soft + (1.0 - distill_alpha) * y_train,
            1e-6, 1 - 1e-6).astype(np.float32)
        y_val_soft = np.clip(
            distill_alpha * xgb_val_soft + (1.0 - distill_alpha) * y_val,
            1e-6, 1 - 1e-6).astype(np.float32)

    if verbose:
        if use_xgb_distill:
            print(f'  XGB distillation loss: alpha={distill_alpha} '
                  f'(y_soft = {distill_alpha}*xgb + {1-distill_alpha:.2f}*y_hard) '
                  f'| train y_soft mean={y_train_soft.mean():.3f} '
                  f'Q90={float(np.quantile(y_train_soft, 0.9)):.3f}')
        elif use_pnl_loss:
            print(f'  PnL-optimization loss: scale={pnl_loss_scale}, '
                  f'budget_lambda={pnl_budget_lambda}, target_sel={pnl_target_selectivity}, '
                  f'commission={pnl_commission}')
        else:
            print(f'  Class-weighted BCE: pos_rate={pos_rate:.3f}, pos_weight={pos_weight:.2f}')
        if dann_active:
            print(f'  DANN: lambda_max={dann_lambda_max}, gamma={dann_gamma}, '
                  f'val_pool={len(X_val_scaled)}, domain_clf=[{hidden_size}->32->1]')

    # PnL-magnitude curriculum activation (THIS ATTEMPT). Active only on the
    # BCE path and only when pnl_train is supplied by the caller. We attach
    # pnl as the 5th tensor in the BCE-path dataset so per-batch curriculum
    # filtering has direct access to per-sample pnl without a separate lookup.
    curriculum_active = (bool(use_curriculum) and bce_path_active
                         and (pnl_train is not None))
    if curriculum_active:
        pnl_train_arr_cur = np.asarray(pnl_train, dtype=np.float32)
        if len(pnl_train_arr_cur) != len(X_train_scaled):
            # Caller mismatch — safer to disable than silently mis-mask.
            curriculum_active = False
            curriculum_info = {'applied': False,
                               'reason': 'pnl_train length mismatch'}
        else:
            total_curriculum_epochs = int(curriculum_strong_epochs) + int(curriculum_relax_epochs)
            n_strong = int(((pnl_train_arr_cur > curriculum_strong_win)
                            | (pnl_train_arr_cur < curriculum_strong_loss)).sum())
            curriculum_info = {
                'applied': True,
                'strong_win': float(curriculum_strong_win),
                'strong_loss': float(curriculum_strong_loss),
                'strong_epochs': int(curriculum_strong_epochs),
                'relax_epochs': int(curriculum_relax_epochs),
                'total_curriculum_epochs': total_curriculum_epochs,
                'min_batch': int(curriculum_min_batch),
                'n_strong_samples': n_strong,
                'n_total_samples': int(len(pnl_train_arr_cur)),
                'strong_fraction': float(n_strong / max(len(pnl_train_arr_cur), 1)),
            }
            if verbose:
                print(f'  Curriculum: strong_win>{curriculum_strong_win}, '
                      f'strong_loss<{curriculum_strong_loss}, '
                      f'strong_epochs={curriculum_strong_epochs}, '
                      f'relax_epochs={curriculum_relax_epochs}')
                print(f'    Strong samples in train: {n_strong}/'
                      f'{len(pnl_train_arr_cur)} '
                      f'({100.0 * n_strong / max(len(pnl_train_arr_cur), 1):.1f}%)')
    else:
        curriculum_info = {
            'applied': False,
            'reason': ('non-BCE path' if not bce_path_active
                       else ('no pnl_train' if pnl_train is None
                             else 'disabled by flag')),
        }
        pnl_train_arr_cur = None

    # Build datasets. Distillation path carries y_soft alongside y_hard;
    # PnL path carries pnl_batch; fallback carries only y_hard.
    if use_xgb_distill:
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_train_soft, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(y_val_soft, dtype=torch.float32),
        )
    elif use_pnl_loss:
        pnl_train_arr = np.clip(np.asarray(pnl_train, dtype=np.float32),
                                pnl_clip_low, pnl_clip_high)
        pnl_val_arr = np.clip(np.asarray(pnl_val, dtype=np.float32),
                              pnl_clip_low, pnl_clip_high)
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(pnl_train_arr, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(pnl_val_arr, dtype=torch.float32),
        )
    else:
        # BCE (default) path carries shift_weights as the third tensor,
        # date_ids as the fourth, and pnl as the fifth (curriculum path).
        # When daily rank is inactive the date_ids placeholder is 0 for
        # all samples. When curriculum is inactive we still pack pnl as
        # zeros so the dataset tuple shape is uniform — the curriculum
        # flag in the inner loop gates actual use.
        if date_ids_train_np is None:
            date_ids_tensor = torch.zeros(len(X_train_scaled), dtype=torch.long)
        else:
            date_ids_tensor = torch.tensor(date_ids_train_np, dtype=torch.long)
        # PnL tensor: prefer curriculum-path pnl (already sanitized) if set,
        # else use caller-supplied pnl_train directly so the listnet ranking
        # path has real magnitudes even when curriculum is disabled.
        # Zero-fill only when pnl_train is genuinely absent.
        if pnl_train_arr_cur is not None:
            pnl_tensor_bce = torch.tensor(pnl_train_arr_cur, dtype=torch.float32)
        elif pnl_train is not None and len(pnl_train) == len(X_train_scaled):
            pnl_tensor_bce = torch.tensor(np.asarray(pnl_train, dtype=np.float32),
                                          dtype=torch.float32)
        else:
            pnl_tensor_bce = torch.zeros(len(X_train_scaled), dtype=torch.float32)
        # IRM env IDs (#320, REJECTED — now default OFF). Chronological env
        # partition. When IRM is inactive we pack a zeros placeholder so the
        # dataset tuple shape is uniform — the irm_active flag in the inner
        # loop gates actual use.
        if env_ids_train_np is not None:
            env_ids_tensor = torch.tensor(env_ids_train_np, dtype=torch.long)
        else:
            env_ids_tensor = torch.zeros(len(X_train_scaled), dtype=torch.long)
        # PnL-distance soft labels (THIS ATTEMPT #321). When inactive, fall
        # back to the binary y as the soft target — this makes the BCE loss
        # bit-identical to the prior baseline whenever pnl_smooth_active is
        # False (non-BCE path or pnl_train missing).
        if y_train_soft_np is not None:
            y_soft_tensor = torch.tensor(y_train_soft_np, dtype=torch.float32)
        else:
            y_soft_tensor = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(shift_weights_np, dtype=torch.float32),
            date_ids_tensor,
            pnl_tensor_bce,
            env_ids_tensor,
            y_soft_tensor,
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
    # Date-stratified batch sampler only when daily ranking is active.
    # Otherwise use the plain shuffled DataLoader so we don't alter
    # Entry 197's sampling semantics on non-ranking paths.
    if bce_path_active and daily_rank_active:
        date_sampler = _DateBatchSampler(
            date_ids_train_np, batch_size=batch_size, shuffle=True, seed=42)
        train_loader = DataLoader(train_dataset, batch_sampler=date_sampler,
                                  pin_memory=(device.type == 'cuda'))
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=(device.type == 'cuda'))

    # Build chronological val folds for multi-fold worst-case selection.
    # Assumes X_val_scaled is sorted by date (the WF gate's val IS sorted —
    # it's carved by date_cutoff on already-sorted dates). Standalone
    # main() also passes sorted val from time_split. With non-sorted val
    # the folds become K random partitions, which still gives a useful
    # bootstrap-style robustness signal but with a different geometric
    # interpretation. Cheap to construct here; reused every epoch.
    fold_indices_list = []
    if multifold_active:
        n_val_total = len(X_val_scaled)
        K_req = max(2, int(n_val_folds))
        fold_size = n_val_total // K_req
        boundaries = [k * fold_size for k in range(K_req)] + [n_val_total]
        candidate_folds = [
            np.arange(boundaries[k], boundaries[k + 1], dtype=np.int64)
            for k in range(K_req)
        ]
        # Drop folds smaller than min size (e.g., last fold may inherit
        # remainder; if any fold is below threshold we shrink K).
        fold_indices_list = [
            f for f in candidate_folds if len(f) >= int(multifold_min_size)
        ]
        if len(fold_indices_list) < 2:
            multifold_active = False
            fold_indices_list = []
    multifold_info = {
        'applied': bool(multifold_active),
        'n_folds_requested': int(n_val_folds),
        'n_folds_used': len(fold_indices_list) if multifold_active else 0,
        'min_fold_size': int(multifold_min_size),
        'fold_sizes': [int(len(f)) for f in fold_indices_list],
        'n_val_total': len(X_val_scaled),
        'aggregator': str(multifold_aggregator).lower(),
        'reason': ('active' if multifold_active
                   else ('non-BCE path' if not bce_path_active
                         else ('val too small for K folds' if not fold_indices_list
                               else 'disabled by flag'))),
    }
    if verbose and multifold_active:
        print(f'  Multi-fold val select: K={len(fold_indices_list)}, '
              f'fold_sizes={[len(f) for f in fold_indices_list]}, '
              f'aggregator={str(multifold_aggregator).lower()}')

    # Multi-quantile PnL regression auxiliary head (THIS ATTEMPT).
    # Active only on the BCE path AND only when pnl_train is supplied.
    # Small Linear(hidden_size, n_quantiles) applied to pre-FC pooled
    # features. Head parameters join the main optimizer so gradient
    # from the pinball loss flows BACK into the LSTM backbone —
    # shaping the shared representation, not just the aux head.
    qr_active = (bool(use_quantile_aux) and bce_path_active
                 and (pnl_train is not None))
    q_head = None
    q_levels_tensor = None
    if qr_active:
        q_head = nn.Linear(hidden_size, len(quantile_levels)).to(device)
        q_levels_tensor = torch.tensor(list(quantile_levels), device=device,
                                       dtype=torch.float32)
    qr_info = {
        'applied': qr_active,
        'weight': float(quantile_aux_weight) if qr_active else None,
        'levels': list(quantile_levels) if qr_active else None,
        'clip_low': float(quantile_pnl_clip_low) if qr_active else None,
        'clip_high': float(quantile_pnl_clip_high) if qr_active else None,
        'reason': ('active' if qr_active
                   else ('disabled by flag' if not use_quantile_aux
                         else ('no pnl_train' if pnl_train is None
                               else 'non-BCE path'))),
    }
    if verbose and qr_active:
        print(f'  Quantile aux: weight={quantile_aux_weight}, '
              f'levels={list(quantile_levels)}, '
              f'head=Linear({hidden_size},{len(quantile_levels)}), '
              f'pnl_clip=[{quantile_pnl_clip_low},{quantile_pnl_clip_high}]')

    # SelectiveNet abstention head (THIS ATTEMPT #312).
    # Active only on the BCE path. Allocates a small MLP that learns
    # `g(x) ∈ (0,1)` on the same pre-FC features the classifier uses.
    # Joint training reshapes f(x)'s gradient: more pressure on samples
    # where g(x) is high (the model's learned trust region), without
    # collapsing the score distribution thanks to the auxiliary CE term.
    sel_active = bool(use_selective_head) and bce_path_active
    sel_head = None
    if sel_active:
        sel_head = SelectionHead(hidden_size, dropout=min(0.2, dropout)).to(device)
    sel_info = {
        'applied': sel_active,
        'target_coverage': float(selective_target_coverage) if sel_active else None,
        'lambda': float(selective_lambda) if sel_active else None,
        'alpha': float(selective_alpha) if sel_active else None,
        'reason': ('active' if sel_active
                   else ('disabled by flag' if not use_selective_head
                         else 'non-BCE path')),
    }
    if verbose and sel_active:
        print(f'  SelectiveNet abstain head: target_cov={selective_target_coverage}, '
              f'lambda={selective_lambda}, alpha={selective_alpha}, '
              f'head=Linear({hidden_size},{max(hidden_size//2,16)})->Linear(...,1)')

    # Temporal-consistency sub-sequence regularization (THIS ATTEMPT).
    # Requires BCE path (use_xgb_distill=False and use_pnl_loss=False)
    # and a truncation small enough that the truncated sequence still
    # has a meaningful length. Guard against pathological sequence
    # lengths (seq_len <= trunc+1 would produce a 1-step "sequence").
    consist_active = (bool(use_temporal_consist) and bce_path_active
                      and int(consist_trunc) > 0)
    consist_info = {
        'applied': consist_active,
        'weight': float(consist_weight) if consist_active else None,
        'trunc': int(consist_trunc) if consist_active else None,
        'rampup_epochs': int(consist_rampup_epochs) if consist_active else None,
        'reason': ('active' if consist_active
                   else ('disabled by flag' if not use_temporal_consist
                         else ('non-BCE path' if not bce_path_active
                               else 'trunc <= 0'))),
    }
    if verbose and consist_active:
        print(f'  Temporal consistency: weight={consist_weight}, '
              f'trunc={consist_trunc} (of seq_len), '
              f'rampup_epochs={consist_rampup_epochs}')

    # Plain Adam (no SAM). SAM (#294) returned wf_avg_wr=0 with 0 valid
    # splits and cost 2x compute per step. The structural fix for the
    # "epoch-0-best" pathology moves up-front: DANN adversarial feature
    # alignment, applied jointly with the classifier. Adam stays as the
    # inner workhorse. When DANN is active, the domain classifier's
    # parameters are included in the same optimizer so the GRL gradient
    # can flow unmodified from classifier → features → LSTM.
    # ===== DUAL-TARGET AUX HEAD construction (THIS ATTEMPT #330) =====
    # Linear(hidden_size, 1) on the LSTM's pre-FC pooled features, returning
    # a raw logit. We pair it with BCEWithLogitsLoss (numerically stable) and
    # apply pos_weight via per-sample reduction='none' BCE so the loss can
    # be added to the main task's loss without scale conflicts. Initialized
    # with bias 0.0 so untrained logits start at sigmoid(0)=0.5 — a neutral
    # prior that lets the aux task drift to its own pos_rate during training
    # (a +15% target hit is rarer than +4%, so the aux head should learn a
    # negative bias). Discarded at function exit; not in model.state_dict.
    target_aux_head = None
    if target_aux_active:
        target_aux_head = nn.Linear(hidden_size, 1).to(device)
        nn.init.constant_(target_aux_head.bias, 0.0)
    trainable_params = list(model.parameters())
    if domain_clf is not None:
        trainable_params = trainable_params + list(domain_clf.parameters())
    if q_head is not None:
        trainable_params = trainable_params + list(q_head.parameters())
    if sel_head is not None:
        trainable_params = trainable_params + list(sel_head.parameters())
    if target_aux_head is not None:
        trainable_params = trainable_params + list(target_aux_head.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=0.0)
    criterion = nn.BCELoss(reduction='none')
    domain_criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    # Structural activation: GCE only engages on the BCE (default) path.
    # When use_xgb_distill or use_pnl_loss divert to their own objectives,
    # GCE is not applied — those paths already diverge from the supervised
    # classification loss this attempt is modifying.
    gce_active = bool(use_gce) and bce_path_active
    gce_info = {
        'applied': gce_active,
        'q': float(gce_q) if gce_active else None,
        'reason': 'active' if gce_active else (
            'disabled by flag' if not use_gce else 'non-BCE path'),
    }
    if verbose and gce_active:
        print(f'  GCE loss: q={gce_q} (q→0 = CE, q→1 = MAE; '
              f'0.7 is Zhang & Sabuncu default for noisy-label training)')

    def _gce_loss(pred, y, q):
        """Generalized Cross Entropy (Zhang & Sabuncu 2018).

        For binary with soft targets y ∈ [0, 1] and predictions p ∈ (0, 1):
          L_q(p, y) = y · (1 - p^q)/q + (1 - y) · (1 - (1-p)^q)/q

        Returns per-sample loss with shape equal to pred — matches the
        contract of nn.BCELoss(reduction='none') so it can be composed
        multiplicatively with pos_weight and covariate-shift weights.

        Gradient properties vs BCE:
          - BCE ∂L/∂p for y=1 is -1/p  (unbounded as p→0)
          - GCE ∂L/∂p for y=1 is -p^(q-1)  (bounded, = -p^-0.3 at q=0.7)
        → Mislabeled samples produce a bounded, not exploding, gradient
          — this is the label-noise-robust property we want.
        """
        p = pred.clamp(1e-6, 1 - 1e-6)
        # Safe exponentiation on clamped p.
        pos_term = (1.0 - p.pow(q)) / q
        neg_term = (1.0 - (1.0 - p).pow(q)) / q
        return y * pos_term + (1.0 - y) * neg_term

    def _pnl_loss(pred, pnl_batch):
        # Interpret pred as soft entry propensity; reward per-sample = net PnL
        # after commission. Scaling brings gradient magnitude roughly in line
        # with BCE. Budget term keeps mean propensity near target_selectivity.
        reward = (pnl_batch - pnl_commission) * pnl_loss_scale
        profit_term = -(pred * reward).mean()
        budget_term = pnl_budget_lambda * (pred.mean() - pnl_target_selectivity).pow(2)
        return profit_term + budget_term

    def _sortino_aux_loss(pred, pnl_batch, commission, target_sel,
                           budget_lambda, eps):
        """Differentiable Sortino-ratio auxiliary loss (THIS ATTEMPT #322).

        Soft per-sample return r_i = pred_i * (pnl_i - commission).
        Sortino-style score = mean(r) / sqrt(mean(min(r,0)^2) + eps), where
        the denominator captures DOWNSIDE-ONLY variance — asymmetric in
        line with our asymmetric trading payoff (+15% target / -3% stop).
        Selectivity budget penalizes deviation of mean(pred) from
        target_sel to prevent the optimizer collapsing pred → 0 (which
        would make the score numerically indeterminate via 0/0) or
        saturating pred → 1 (which would maximize Sortino magnitude
        without separating classes).

        Returns LOSS = -sortino + budget_lambda * (mean(pred) - target)^2.
        Lower is better. Composed additively with BCE in the main loop.
        """
        r = pred * (pnl_batch - commission)
        mean_r = r.mean()
        downside = torch.clamp(-r, min=0.0)
        dd = torch.sqrt((downside * downside).mean() + eps)
        sortino = mean_r / (dd + eps)
        budget = budget_lambda * (pred.mean() - target_sel).pow(2)
        return -sortino + budget

    def _pinball_loss(pred_q, y, q_levels):
        """Multi-quantile pinball (aka quantile regression) loss.

        For each quantile τ in q_levels, the pinball loss on residual
        e = y - ŷ_τ is:
          ρ_τ(e) = max(τ·e, (τ-1)·e)   = τ·e when e>0, (τ-1)·e when e<0.

        Minimizing ρ_τ in expectation yields the τ-th conditional
        quantile of y given the features. Averaging over quantiles
        optimizes a proper scoring rule on the full CDF.

        Args:
          pred_q:   (B, n_q) predicted quantiles, UNORDERED.
          y:        (B,) target scalars (per-sample PnL, clipped).
          q_levels: (n_q,) quantile levels in (0, 1).

        Returns 0-d tensor (mean over samples and quantiles).
        """
        errors = y.unsqueeze(-1) - pred_q       # (B, n_q)
        return torch.maximum(q_levels * errors, (q_levels - 1.0) * errors).mean()

    def _distill_loss(pred, y_hard_batch, y_soft_batch):
        # BCE against the soft (blended) target. sample_weight uses the HARD
        # label only for class-imbalance rebalancing, independent of the soft
        # target — this separates two concerns:
        #   - WHAT we're predicting: the XGB-blended probability (y_soft)
        #   - HOW we weight per-class: pos_weight on the rare class (y_hard)
        # Using y_soft for weighting would double-apply XGB's scale_pos_weight.
        sample_weight = y_hard_batch * pos_weight + (1.0 - y_hard_batch)
        return (criterion(pred, y_soft_batch) * sample_weight).mean()

    # Single tracking: best val_loss (Entry 197 recipe). Precision@0.6 selector
    # removed — #288 proved it picks degenerate "never predict positive" states.
    best_val_loss = float('inf')
    best_loss_state = None
    patience_counter = 0
    # SWA snapshots: last swa_window val_loss-improving checkpoints, averaged
    # at the end for smoother boundaries than any single-epoch state.
    swa_states = []

    # Accumulators for ranking diagnostics over the whole epoch.
    rank_pair_count_epoch = 0
    rank_loss_sum_epoch = 0.0
    rank_batches_with_pairs = 0

    # Curriculum diagnostics accumulated across all epochs.
    curriculum_epoch_log = []

    # Gradient-noise injection state (THIS ATTEMPT #323).
    # global_step counts every optimizer.step() call across all epochs,
    # so polynomial noise decay 1/(1+t)^gamma is applied uniformly,
    # not reset per epoch. Diagnostics track noise vs gradient norm to
    # confirm the injection is meaningful (sigma << ||grad|| means no
    # effect; sigma ~ ||grad|| means optimizer is doing pure noise).
    grad_noise_step = 0
    grad_noise_diag = {
        'n_steps_injected': 0,
        'noise_norm_sum': 0.0,
        'grad_norm_sum': 0.0,
    }
    # Dedicated RNG for gradient noise — fixed seed 4242 keeps run-to-run
    # variance bounded (composes with torch.manual_seed(42) for the LSTM
    # init; the two seeds together fully determine the trajectory).
    grad_noise_gen = torch.Generator(device=DEVICE)
    grad_noise_gen.manual_seed(4242)

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        rank_pair_count_epoch = 0
        rank_loss_sum_epoch = 0.0
        rank_batches_with_pairs = 0

        # Per-epoch curriculum thresholds. Linear relaxation toward 0
        # during the `relax_epochs` window, then fully permissive.
        if curriculum_active:
            se = int(curriculum_strong_epochs)
            re = int(curriculum_relax_epochs)
            if epoch < se:
                cur_win_thr = float(curriculum_strong_win)
                cur_loss_thr = float(curriculum_strong_loss)
                cur_phase = 'strong'
            elif epoch < se + re:
                # progress: 0 at the start of relax, 1 at the end of relax.
                rp = (epoch - se + 1) / max(re, 1)
                rp = max(0.0, min(1.0, rp))
                cur_win_thr = float(curriculum_strong_win) * (1.0 - rp)
                cur_loss_thr = float(curriculum_strong_loss) * (1.0 - rp)
                cur_phase = 'relax'
            else:
                cur_win_thr = 0.0
                cur_loss_thr = 0.0
                cur_phase = 'full'
        else:
            cur_win_thr = 0.0
            cur_loss_thr = 0.0
            cur_phase = 'disabled'
        curriculum_batches_kept = 0
        curriculum_batches_skipped = 0
        curriculum_samples_kept = 0
        curriculum_samples_seen = 0

        for batch in train_loader:
            shift_batch = None
            date_ids_batch = None
            pnl_batch_raw = None
            env_ids_batch = None
            y_soft_pnl_batch = None
            if use_xgb_distill:
                X_batch, y_batch, y_soft_batch = batch
                y_soft_batch = y_soft_batch.to(device)
            elif use_pnl_loss:
                X_batch, y_batch, pnl_batch = batch
                pnl_batch = pnl_batch.to(device)
            else:
                (X_batch, y_batch, shift_batch, date_ids_batch,
                 pnl_batch_raw, env_ids_batch, y_soft_pnl_batch) = batch
                shift_batch = shift_batch.to(device)
                date_ids_batch = date_ids_batch.to(device)
                pnl_batch_raw = pnl_batch_raw.to(device)
                env_ids_batch = env_ids_batch.to(device)
                y_soft_pnl_batch = y_soft_pnl_batch.to(device)
            # ===== WINDOW WARPING (THIS ATTEMPT #334) =====
            # Apply per-sequence temporal distortion BEFORE the .to(device)
            # transfer. Operates on the CPU numpy view of the dataloader's
            # X_batch, then re-wraps to a torch tensor for the device move.
            # Per-sample stochastic with prob window_warp_p — un-warped
            # samples flow through unchanged. Composes cleanly with the
            # downstream mixup branch (warped pair → mixup interpolation).
            if window_warp_active:
                X_np = X_batch.numpy()
                X_np = _window_warp_batch(
                    X_np,
                    p=float(window_warp_p),
                    window_frac=float(window_warp_window_frac),
                    scale_low=float(window_warp_scale_low),
                    scale_high=float(window_warp_scale_high),
                    rng=window_warp_rng)
                X_batch = torch.from_numpy(X_np)
            # ===== SPECAUGMENT-STYLE TIME MASKING (THIS ATTEMPT #335) =====
            # Apply per-sequence contiguous time-block zeroing BEFORE the
            # .to(device) transfer. Operates on the CPU numpy view of the
            # dataloader's X_batch (or the warped output if window warping
            # is also on — the two compose), then re-wraps to a torch
            # tensor for the device move. Per-sample stochastic with prob
            # time_mask_p — un-masked samples flow through unchanged.
            # Composes cleanly with the downstream mixup branch (masked
            # pair → mixup interpolation, since zero is the per-feature
            # median post-RobustScaler).
            if time_mask_active:
                X_np = X_batch.numpy()
                X_np = _time_mask_batch(
                    X_np,
                    p=float(time_mask_p),
                    mask_size_frac=float(time_mask_size_frac),
                    n_masks=int(time_mask_n_masks),
                    rng=time_mask_rng)
                X_batch = torch.from_numpy(X_np)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # PnL-magnitude curriculum mask. Applied BEFORE mixup so convex
            # combinations never mix in-phase with out-of-phase samples. In
            # the 'full' phase the thresholds are both 0 so (pnl > 0) |
            # (pnl < 0) still excludes only the exact-zero samples, which is
            # a vanishing share — effectively a no-op.
            if curriculum_active and pnl_batch_raw is not None and cur_phase != 'full':
                curriculum_samples_seen += int(X_batch.size(0))
                keep_mask = (pnl_batch_raw > cur_win_thr) | (pnl_batch_raw < cur_loss_thr)
                n_keep = int(keep_mask.sum().item())
                if n_keep < int(curriculum_min_batch):
                    curriculum_batches_skipped += 1
                    continue
                curriculum_batches_kept += 1
                curriculum_samples_kept += n_keep
                X_batch = X_batch[keep_mask]
                y_batch = y_batch[keep_mask]
                if shift_batch is not None:
                    shift_batch = shift_batch[keep_mask]
                if date_ids_batch is not None:
                    date_ids_batch = date_ids_batch[keep_mask]
                if env_ids_batch is not None:
                    env_ids_batch = env_ids_batch[keep_mask]
                if y_soft_pnl_batch is not None:
                    y_soft_pnl_batch = y_soft_pnl_batch[keep_mask]
                # pnl_batch_raw no longer used after this point in the loop
                # (BCE path doesn't consume it downstream), but subset it
                # for consistency in case future code reads it.
                pnl_batch_raw = pnl_batch_raw[keep_mask]
            elif curriculum_active and pnl_batch_raw is not None:
                # 'full' phase — count but don't filter.
                curriculum_samples_seen += int(X_batch.size(0))
                curriculum_batches_kept += 1
                curriculum_samples_kept += int(X_batch.size(0))

            # MANIFOLD MIXUP per-batch coin flip (THIS ATTEMPT #328).
            # When manifold_mixup_active AND the coin flip lands on the
            # manifold side, we DO NOT mix the input — instead we sample
            # lam and perm now, and mix at the pre-FC pooled hidden layer
            # in the forward pass below. y_mix uses the same (lam, perm)
            # so labels match the hidden-space interpolation.
            do_manifold_mixup_this_batch = (
                manifold_mixup_active
                and (manifold_mixup_rng.rand() < float(manifold_mixup_p)))

            if do_manifold_mixup_this_batch:
                # Hidden-space mixup branch: leave X_mix as the original
                # X_batch (no input interpolation). Sample lam and perm
                # so the forward pass can apply them to the pooled
                # features. y_mix is mixed in label space at the same
                # lam so the BCE target matches the interpolated head
                # prediction.
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(X_batch.size(0), device=device)
                X_mix = X_batch  # input UNTOUCHED — mix happens at hidden layer
                y_mix = lam * y_batch + (1 - lam) * y_batch[perm]
                if shift_batch is not None:
                    shift_mix = lam * shift_batch + (1 - lam) * shift_batch[perm]
                if y_soft_pnl_batch is not None:
                    y_soft_pnl_mix = (lam * y_soft_pnl_batch
                                      + (1 - lam) * y_soft_pnl_batch[perm])
            elif mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(X_batch.size(0), device=device)
                X_mix = lam * X_batch + (1 - lam) * X_batch[perm]
                y_mix = lam * y_batch + (1 - lam) * y_batch[perm]
                if use_xgb_distill:
                    y_soft_mix = lam * y_soft_batch + (1 - lam) * y_soft_batch[perm]
                    y_soft_mix = y_soft_mix.clamp(1e-6, 1 - 1e-6)
                if use_pnl_loss:
                    pnl_mix = lam * pnl_batch + (1 - lam) * pnl_batch[perm]
                if shift_batch is not None:
                    # Shift weights are positive scalars — linear mix is
                    # consistent with the sample pairing from Mixup.
                    shift_mix = lam * shift_batch + (1 - lam) * shift_batch[perm]
                if qr_active and pnl_batch_raw is not None:
                    # Linearly interpolate PnL targets in sync with Mixup.
                    # A convex combination of two PnL scalars is still a
                    # reasonable regression target — mirrors how y_mix
                    # linearly interpolates the binary labels.
                    pnl_mix_q = lam * pnl_batch_raw + (1 - lam) * pnl_batch_raw[perm]
                if sortino_active and pnl_batch_raw is not None:
                    # Sortino aux uses PnL-magnitude info; mix in lockstep
                    # with the predictions so the ratio statistic is
                    # internally consistent with the soft-return formula
                    # r_i = pred_mix_i * (pnl_mix_sortino_i - commission).
                    pnl_mix_sortino = (lam * pnl_batch_raw
                                       + (1 - lam) * pnl_batch_raw[perm])
                if y_soft_pnl_batch is not None:
                    # Linearly interpolate the soft labels in lockstep with
                    # the mixup'd inputs. A convex combination of two
                    # sigmoid-transformed PnL distances is itself a smooth
                    # target — both endpoints are in (0, 1) so the mixed
                    # value stays a valid BCE target without re-clipping.
                    y_soft_pnl_mix = (lam * y_soft_pnl_batch
                                      + (1 - lam) * y_soft_pnl_batch[perm])
            else:
                X_mix, y_mix = X_batch, y_batch
                if use_xgb_distill:
                    y_soft_mix = y_soft_batch
                if use_pnl_loss:
                    pnl_mix = pnl_batch
                if shift_batch is not None:
                    shift_mix = shift_batch
                if qr_active and pnl_batch_raw is not None:
                    pnl_mix_q = pnl_batch_raw
                if sortino_active and pnl_batch_raw is not None:
                    pnl_mix_sortino = pnl_batch_raw
                if y_soft_pnl_batch is not None:
                    y_soft_pnl_mix = y_soft_pnl_batch

            # Compute the mini-batch loss.
            optimizer.zero_grad()

            if dann_active:
                # Shared forward pass up to the pooled feature vector so
                # (a) the classifier head sees the same features that the
                # domain classifier is trying to make train/val-invariant,
                # and (b) the GRL gradient can flow all the way back.
                feat_train = _lstm_features(model, X_mix)
                pred_local = _head_forward(model, feat_train)

                # Adversarial batch: pair each training example with an
                # equal-count random val sample. Concatenating keeps BN/
                # LayerNorm statistics consistent if the model ever grows
                # such layers later.
                n_dann = min(X_mix.size(0), dann_val_batch_size)
                val_idx = torch.randint(0, X_val_dev.size(0),
                                        (n_dann,), device=device)
                X_val_sub = X_val_dev[val_idx]
                feat_val = _lstm_features(model, X_val_sub)

                # Ramp alpha from 0 to lambda_max over training. Ganin's
                # 2/(1+e^-γp)-1 schedule: slow start (no adversarial
                # pressure while classifier finds a sensible decision
                # boundary), faster growth near mid-training.
                p = min(1.0, epoch / max(1, max_epochs - 1))
                alpha = float(dann_lambda_max * (2.0 / (1.0 + np.exp(-dann_gamma * p)) - 1.0))

                # GRL output → domain classifier → BCE-with-logits.
                # Labels: 0 = train, 1 = val.
                feat_combined = torch.cat([feat_train, feat_val], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(feat_train.size(0), device=device),
                    torch.ones(feat_val.size(0), device=device),
                ])
                feat_reversed = _grad_reverse(feat_combined, alpha)
                domain_logit = domain_clf(feat_reversed)
                domain_loss = domain_criterion(domain_logit, domain_labels)
            elif do_manifold_mixup_this_batch:
                # ===== MANIFOLD MIXUP forward path (THIS ATTEMPT #328) =====
                # Mix at the LSTM's pre-FC pooled HIDDEN representation
                # rather than at the input. Single forward pass: extract
                # features for the original (un-mixed) batch, then form
                # the convex combination of features at row i and row
                # perm[i] using the same lam used to mix labels above.
                # Apply the FC head to the mixed features.
                #
                # Verma et al. 2019 (ICML): "Manifold Mixup forces models
                # to predict less confidently when interpolating in
                # hidden space, which leads to flatter decision boundaries
                # and better generalization than input mixup alone."
                #
                # Same compute cost as the standard input-mixup branch:
                # one forward pass, one backward pass. The only structural
                # difference is which layer's output gets convex-combined.
                feat_clean = _lstm_features(model, X_batch)
                feat_mix = lam * feat_clean + (1.0 - lam) * feat_clean[perm]
                pred_local = _head_forward(model, feat_mix)
                feat_shared = feat_mix  # so downstream aux heads see mixed feats
                domain_loss = None
            else:
                # When the quantile aux OR the SelectiveNet abstain head
                # is active, extract pre-FC features once and route them
                # through both the classifier and the auxiliary head(s).
                # Both heads see the SAME dropout/noise realization, so
                # gradients from each loss reach the backbone through a
                # single forward pass.
                if qr_active or sel_active or target_aux_active:
                    feat_shared = _lstm_features(model, X_mix)
                    pred_local = _head_forward(model, feat_shared)
                else:
                    feat_shared = None
                    pred_local = model(X_mix)
                domain_loss = None

            if use_xgb_distill:
                loss = _distill_loss(pred_local, y_mix, y_soft_mix)
            elif use_pnl_loss:
                loss = _pnl_loss(pred_local, pnl_mix)
            else:
                # Class-balanced supervised loss composed with covariate-shift
                # weights. class_weight addresses label imbalance; shift_mix
                # addresses temporal distribution shift between train and
                # val/test. Composing multiplicatively keeps each a per-sample
                # scalar.
                #
                # Per-sample loss is GCE (Zhang & Sabuncu 2018) when
                # use_gce=True (opt-in; rejected as #305 default), else
                # BCE (Entry 197 fallback). Both are returned with
                # reduction='none' shape matching y_mix, so the same
                # multiplication by total_weight and the same .mean()
                # aggregation apply unchanged.
                #
                # PnL-distance soft labels (THIS ATTEMPT #321): when active,
                # the BCE TARGET is replaced by y_soft_pnl_mix (a continuous
                # function of realized PnL), but pos_weight class
                # re-balancing still uses the BINARY y_mix because that's
                # the metric the gate's WR is computed against. The two
                # are conceptually independent — class weighting is about
                # the empirical positive RATE; the loss target is about
                # what each individual sample teaches the model.
                class_weight = y_mix * pos_weight + (1.0 - y_mix)
                total_weight = class_weight * shift_mix
                if pnl_smooth_active and y_soft_pnl_batch is not None:
                    bce_target = y_soft_pnl_mix
                else:
                    bce_target = y_mix
                if brier_active:
                    # Brier score (THIS ATTEMPT #331): MSE on probabilities.
                    # Strictly proper, bounded linear gradient. See the
                    # use_brier_loss docstring for theory and rationale.
                    per_sample = (pred_local - bce_target) ** 2
                elif gce_active:
                    per_sample = _gce_loss(pred_local, bce_target, gce_q)
                else:
                    per_sample = criterion(pred_local, bce_target)
                weighted_per_sample = per_sample * total_weight
                ce_mean = weighted_per_sample.mean()
                loss = ce_mean

                # ===== R-DROP REGULARIZED DROPOUT (THIS ATTEMPT #325) =====
                # Run a SECOND forward pass on the same X_mix; because the
                # model is in train() mode, this second pass uses an
                # independent dropout mask realization. The two predictions
                # p1 (=pred_local) and p2 differ in proportion to the
                # model's dropout sensitivity. Symmetric KL pushes them
                # toward each other → optimizer learns dropout-invariant
                # features. Linear ramp-up of the KL weight over the
                # first r_drop_rampup_epochs avoids early-epoch divergence
                # when both forward passes are essentially random.
                #
                # Reuses pred_local as p1 to preserve the existing BCE
                # gradient path. The second BCE on p2 is added with a
                # 0.5 weight so the total class-prediction gradient is
                # comparable in magnitude to the single-pass baseline
                # (each forward pass contributes 0.5 * weighted BCE).
                if r_drop_active:
                    pred_local2 = model(X_mix)
                    if brier_active:
                        per_sample2 = (pred_local2 - bce_target) ** 2
                    elif gce_active:
                        per_sample2 = _gce_loss(pred_local2, bce_target, gce_q)
                    else:
                        per_sample2 = criterion(pred_local2, bce_target)
                    weighted_per_sample2 = per_sample2 * total_weight
                    ce_mean2 = weighted_per_sample2.mean()
                    # Average the two BCE means so total class-prediction
                    # gradient stays at single-pass magnitude — preserves
                    # tuning of pos_weight and shift weights.
                    loss = 0.5 * (ce_mean + ce_mean2)

                    # Symmetric KL between the two binary distributions.
                    # KL(P||Q) = p*log(p/q) + (1-p)*log((1-p)/(1-q)).
                    # Stabilization: clamp probs into (1e-6, 1-1e-6)
                    # so log/division never produce inf/NaN.
                    p1 = pred_local.clamp(1e-6, 1.0 - 1e-6)
                    p2 = pred_local2.clamp(1e-6, 1.0 - 1e-6)
                    kl_12 = (p1 * (torch.log(p1) - torch.log(p2))
                             + (1.0 - p1) * (torch.log(1.0 - p1)
                                             - torch.log(1.0 - p2)))
                    kl_21 = (p2 * (torch.log(p2) - torch.log(p1))
                             + (1.0 - p2) * (torch.log(1.0 - p2)
                                             - torch.log(1.0 - p1)))
                    kl_sym = 0.5 * (kl_12.mean() + kl_21.mean())

                    # Linear ramp: KL weight grows from 0 over
                    # r_drop_rampup_epochs. Avoids two-random-forward
                    # passes producing huge KL early in training.
                    rd_ramp = min(1.0,
                                  float(epoch + 1)
                                  / max(1.0, float(r_drop_rampup_epochs)))
                    loss = loss + rd_ramp * float(r_drop_alpha) * kl_sym

                # AUXILIARY SORTINO-RATIO LOSS (THIS ATTEMPT #322).
                # Adds a small batch-level term that maximizes risk-adjusted
                # return of soft-selected trades, with downside-only variance
                # in the denominator. Penalizes inconsistent losing trades
                # — directly attacks the cross-split WR std failure mode.
                # See module docstring for full rationale. Linear warmup
                # over the first sortino_warmup_epochs lets BCE establish
                # a sensible decision boundary before the Sortino term
                # pulls on it (avoids early-epoch divergence on noisy
                # batch-level std estimates).
                if (sortino_active
                        and pnl_batch_raw is not None
                        and pred_local.numel() == pnl_mix_sortino.numel()):
                    s_ramp = min(1.0,
                                 float(epoch + 1)
                                 / max(1.0, float(sortino_warmup_epochs)))
                    s_loss = _sortino_aux_loss(
                        pred_local, pnl_mix_sortino,
                        commission=float(sortino_commission),
                        target_sel=float(sortino_target_selectivity),
                        budget_lambda=float(sortino_budget_lambda),
                        eps=float(sortino_eps))
                    loss = loss + s_ramp * float(sortino_aux_weight) * s_loss

                # SelectiveNet abstention loss (THIS ATTEMPT #312).
                # g(x) ∈ (0,1) selects the "trust region"; selective_risk
                # is the g-weighted mean of per-sample loss; coverage
                # penalty stops g from collapsing to 0. Combined loss:
                #   L = (1-α)·BCE + α·(selective_risk + λ·cov_penalty)
                # The α·selective_risk term routes EXTRA gradient through
                # f(x) on g-trusted samples (sharper there); the (1-α)·BCE
                # keeps f(x) honest on un-trusted samples (no collapse);
                # λ·cov_penalty stops g(x) from learning "always abstain".
                if sel_active and sel_head is not None and feat_shared is not None:
                    g = sel_head(feat_shared)
                    coverage = g.mean()
                    selective_risk = (g * weighted_per_sample).sum() / (g.sum() + 1e-6)
                    cov_penalty = float(selective_lambda) * torch.clamp(
                        float(selective_target_coverage) - coverage,
                        min=0.0).pow(2)
                    sr_term = selective_risk + cov_penalty
                    alpha_s = float(selective_alpha)
                    loss = (1.0 - alpha_s) * ce_mean + alpha_s * sr_term

                # Multi-quantile distributional aux loss (THIS ATTEMPT).
                # q_head predicts {p10, p25, p50, p75, p90} of realized
                # PnL from the same shared features; pinball loss pulls
                # them to the correct quantiles. The resulting gradient
                # into feat_shared — and thus into the LSTM backbone —
                # encodes magnitude and tail-asymmetry information that
                # the binary BCE gradient throws away.
                if qr_active and q_head is not None and feat_shared is not None:
                    q_pred = q_head(feat_shared)
                    pnl_target_q = pnl_mix_q.clamp(
                        float(quantile_pnl_clip_low),
                        float(quantile_pnl_clip_high))
                    q_loss = _pinball_loss(q_pred, pnl_target_q, q_levels_tensor)
                    loss = loss + float(quantile_aux_weight) * q_loss

                # ===== DUAL-TARGET MULTI-TASK AUX HEAD loss (THIS ATTEMPT #330)
                # Build the binary aux target on the fly from pnl_batch_raw,
                # apply mixup using the SAME lam (so the aux target is
                # consistent with the input mix and the main task's y_mix),
                # forward through target_aux_head on the shared features, and
                # add a per-sample pos-weighted BCE-with-logits to the total
                # loss. Logits-based BCE is numerically stabler than
                # sigmoid-then-BCE for an aux head whose logits are not
                # temperature-scaled.
                if (target_aux_active
                        and target_aux_head is not None
                        and feat_shared is not None
                        and pnl_batch_raw is not None):
                    y_target_batch = (pnl_batch_raw
                                      >= float(target_aux_threshold)).float()
                    if mixup_alpha > 0:
                        # Reuse the SAME lam and perm that mixed X_mix and
                        # y_mix above so the aux target is consistent with
                        # the interpolated input. Linear interpolation of
                        # binary labels yields a soft target in [0, 1] which
                        # BCE-with-logits handles natively.
                        y_target_mix = (lam * y_target_batch
                                        + (1.0 - lam) * y_target_batch[perm])
                    else:
                        y_target_mix = y_target_batch
                    target_logit = target_aux_head(feat_shared).squeeze(-1)
                    aux_per_sample = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            target_logit, y_target_mix, reduction='none'))
                    aux_class_weight = (y_target_mix * float(target_pos_weight)
                                        + (1.0 - y_target_mix))
                    aux_loss = (aux_per_sample * aux_class_weight).mean()
                    loss = loss + float(target_aux_weight) * aux_loss

                # Temporal-consistency sub-sequence regularization
                # (THIS ATTEMPT). Forward pass on X_mix[:, trunc:, :]
                # using the SAME model weights. Both passes stay in
                # train() mode so dropout/input-noise stochastics apply
                # — the regularizer pushes the model to be robust to
                # truncation EVEN UNDER the usual training noise.
                # Linear ramp-up over consist_rampup_epochs lets the
                # classifier find a sensible decision surface before
                # consistency pressure kicks in.
                if (consist_active and X_mix.size(1) > int(consist_trunc) + 1):
                    trunc = int(consist_trunc)
                    X_trunc = X_mix[:, trunc:, :]
                    pred_trunc = model(X_trunc)
                    ramp = min(1.0, (epoch + 1) / max(1.0, float(consist_rampup_epochs)))
                    consist_term = ((pred_local - pred_trunc) ** 2).mean()
                    loss = loss + ramp * float(consist_weight) * consist_term

                # ===== IRM penalty (THIS ATTEMPT #320) =====
                # For each environment present in the batch with >=
                # irm_min_per_env samples, compute the IRMv1 penalty
                # (Arjovsky et al. 2019, eq. IRMv1):
                #     pen_e = || d/dw  BCE(w * f(x), y) ||^2  at w=1
                # where f(x) is the model's logit (pre-sigmoid). The
                # gradient of the dummy-scaled loss w.r.t. w at w=1 is
                # the "regret" of the optimal classifier in that
                # environment; squaring it gives a positive penalty
                # whose minimum (=0) is achieved iff the same classifier
                # is optimal across environments — i.e., the
                # representation captures only invariant features.
                #
                # Composition with mixup: env_ids_batch holds the
                # environment IDs of the PRIMARY (unmixed) samples.
                # With lam >= 0.5 the primary sample dominates X_mix[i],
                # so this approximation is consistent with how
                # class_weight uses y_mix downstream.
                #
                # Composition with class weighting: per-env BCE uses the
                # SAME pos_weight class re-balancing as the main loss,
                # ensuring positive samples carry equivalent gradient
                # mass in every environment regardless of base-rate
                # drift across the chronological partition.
                if irm_active and env_ids_batch is not None:
                    # Separate forward pass for logits — get_logits has its
                    # own input-noise/dropout realization, so the IRM
                    # penalty operates on a slightly different forward
                    # pass than the main BCE. This is fine: IRM averages
                    # over many batches and the gradient signal is on the
                    # FUNCTION the model represents, not on a specific
                    # noise realization.
                    logits_irm = model.get_logits(X_mix) / model._output_temp.clamp(min=1e-6)
                    irm_penalty = torch.zeros((), device=device)
                    n_active_envs = 0
                    for env_id in range(int(irm_n_envs)):
                        env_mask = env_ids_batch == env_id
                        n_e = int(env_mask.sum().item())
                        if n_e < int(irm_min_per_env):
                            continue
                        env_logits = logits_irm[env_mask]
                        env_y = y_mix[env_mask]
                        env_class_w = env_y * pos_weight + (1.0 - env_y)
                        scale = torch.tensor(1.0, device=device, requires_grad=True)
                        env_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                            env_logits * scale, env_y,
                            weight=env_class_w, reduction='mean')
                        env_grad = torch.autograd.grad(
                            env_bce, [scale], create_graph=True)[0]
                        irm_penalty = irm_penalty + env_grad ** 2
                        n_active_envs += 1
                    if n_active_envs >= 2:
                        # Linear warmup: zero penalty at epoch 0 lets the
                        # classifier find a sensible decision boundary
                        # before invariance pressure pulls on it. After
                        # warmup, lam_irm is at its full irm_lambda_max.
                        lam_irm = float(irm_lambda_max) * min(
                            1.0,
                            float(epoch) / max(1.0, float(irm_warmup_epochs)))
                        loss = loss + lam_irm * irm_penalty / float(n_active_envs)

            # Mean Teacher consistency loss (BCE path only). Teacher sees
            # the same mixup'd batch but in eval() mode — no input dropout,
            # no INPUT_NOISE, no hidden noise — so it provides a cleaner,
            # trajectory-averaged reference. MSE on [0,1] sigmoid outputs
            # has a natural magnitude (<= 0.25 at worst), so mt_lambda_max
            # = 0.5 scales the consistency term to a fraction of typical
            # BCE (~0.3-0.7). Ramp-up avoids suppressing the student before
            # the teacher has diverged from its init.
            if mt_active and model_ema is not None:
                rampup_p = min(1.0, (epoch + 1) / max(1.0, float(mt_rampup_epochs)))
                mt_lambda = float(mt_lambda_max * np.exp(-5.0 * (1.0 - rampup_p) ** 2))
                with torch.no_grad():
                    teacher_pred = model_ema(X_mix)
                consistency = ((pred_local - teacher_pred) ** 2).mean()
                loss = loss + mt_lambda * consistency

            # Daily ranking loss. Listwise top-K (THIS ATTEMPT) when
            # use_listnet_rank=True AND pnl_batch_raw is available;
            # pairwise margin-hinge (#303, legacy) otherwise. Both variants
            # require a CLEAN (non-mixup) forward pass because mixup
            # scrambles per-date structure — X_mix[i] is a convex
            # combination of two samples that may be from different dates.
            # Model stays in train() mode, so dropout and INPUT_NOISE are
            # active; we want the ranking signal to be robust to the same
            # input noise the BCE path sees.
            if daily_rank_active and date_ids_batch is not None:
                pred_clean = model(X_batch)
                listnet_ok = (listnet_active
                              and (pnl_batch_raw is not None)
                              and (pnl_batch_raw.numel() == pred_clean.numel()))
                if listnet_ok:
                    rank_result = _listnet_topk_loss(
                        pred_clean, pnl_batch_raw, date_ids_batch,
                        min_per_day=daily_rank_min_per_day,
                        top_k_frac=float(listnet_top_k_frac),
                        temperature=float(listnet_temperature))
                    if rank_result[0] is not None:
                        rank_loss_term, n_days_batch = rank_result
                        loss = loss + float(daily_rank_lambda) * rank_loss_term
                        rank_pair_count_epoch += n_days_batch
                        rank_loss_sum_epoch += float(rank_loss_term.item())
                        rank_batches_with_pairs += 1
                else:
                    rank_result = _daily_rank_loss(
                        pred_clean, y_batch, date_ids_batch,
                        min_per_day=daily_rank_min_per_day,
                        margin=daily_rank_margin)
                    if rank_result[0] is not None:
                        rank_loss_term, n_pairs_batch = rank_result
                        loss = loss + float(daily_rank_lambda) * rank_loss_term
                        rank_pair_count_epoch += n_pairs_batch
                        rank_loss_sum_epoch += float(rank_loss_term.item())
                        rank_batches_with_pairs += 1

            # Domain-adversarial loss is added with plain sum — the alpha
            # scaling lives inside the gradient reversal so the domain
            # classifier itself trains at unit gradient (it needs normal
            # SGD to stay sharp), while the feature extractor sees the
            # alpha-scaled reverse gradient.
            if domain_loss is not None:
                loss = loss + domain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            # Annealed gradient noise injection (THIS ATTEMPT #323).
            # Inject AFTER clipping so each parameter's effective update is
            # clip(grad) + xi_t. This preserves the full noise contribution;
            # injecting BEFORE clipping would let large noise samples cap
            # ||grad+xi||=1 and silently drop noise mass on big-grad steps.
            #
            # Step counter increments per optimizer.step(), not per epoch,
            # so the polynomial decay 1/(1+t)^gamma anneals uniformly
            # regardless of dataset size or epoch boundaries. Warmup steps
            # inject zero noise, letting Adam first establish a sensible
            # gradient direction before exploration begins.
            if (grad_noise_active
                    and grad_noise_step >= int(grad_noise_warmup_steps)):
                t_post = grad_noise_step - int(grad_noise_warmup_steps) + 1
                sigma_t = float(grad_noise_eta) / (
                    (1.0 + float(t_post)) ** float(grad_noise_gamma))
                # Diagnostic: track per-step grad and noise norms so the
                # returned info dict can flag pathological regimes
                # (e.g., grad_norm == 0 — would mean optimizer is pure
                # noise; or noise_norm << grad_norm — would mean noise
                # is irrelevant). Sums are over the FIRST 200 injected
                # steps to keep the bookkeeping cheap.
                if grad_noise_diag['n_steps_injected'] < 200:
                    g_sq = 0.0
                    n_sq = 0.0
                with torch.no_grad():
                    for p in trainable_params:
                        if p.grad is None:
                            continue
                        noise = torch.randn(
                            p.grad.shape, generator=grad_noise_gen,
                            dtype=p.grad.dtype, device=p.grad.device) * sigma_t
                        if grad_noise_diag['n_steps_injected'] < 200:
                            g_sq += float(p.grad.pow(2).sum().item())
                            n_sq += float(noise.pow(2).sum().item())
                        p.grad.add_(noise)
                if grad_noise_diag['n_steps_injected'] < 200:
                    grad_noise_diag['grad_norm_sum'] += float(g_sq) ** 0.5
                    grad_noise_diag['noise_norm_sum'] += float(n_sq) ** 0.5
                    grad_noise_diag['n_steps_injected'] += 1

            optimizer.step()
            grad_noise_step += 1

            # EMA update. Done after the student step so the teacher
            # integrates the most recent iterate. Buffers (e.g.
            # _output_temp, _seq_normalize) are non-trainable constants so
            # copying them verbatim keeps the teacher semantically
            # identical to the student in inference behavior.
            if mt_active and model_ema is not None:
                with torch.no_grad():
                    for p_ema, p_s in zip(model_ema.parameters(),
                                          model.parameters()):
                        p_ema.data.mul_(mt_ema_decay).add_(
                            p_s.data, alpha=1.0 - mt_ema_decay)

            train_loss += loss.item()

        model.eval()
        # Multi-fold val selection: collect per-sample losses across the
        # whole val set ONCE, then aggregate by fold. Equivalent to
        # running the val pass K times but with K-fold less compute.
        val_per_sample_losses_chunks = []
        full_val_loss_running = 0.0
        full_val_count_running = 0
        with torch.no_grad():
            for batch in val_loader:
                if use_xgb_distill:
                    X_batch, y_batch, y_soft_batch = batch
                    y_soft_batch_d = y_soft_batch.to(device)
                elif use_pnl_loss:
                    X_batch, y_batch, pnl_batch = batch
                    pnl_batch_d = pnl_batch.to(device)
                else:
                    X_batch, y_batch = batch
                X_batch_d, y_batch_d = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch_d)
                if use_xgb_distill:
                    # Distill loss is reduce='mean' inside; track scalar only.
                    batch_loss = _distill_loss(pred, y_batch_d, y_soft_batch_d).item()
                    full_val_loss_running += batch_loss * X_batch_d.size(0)
                    full_val_count_running += X_batch_d.size(0)
                elif use_pnl_loss:
                    batch_loss = _pnl_loss(pred, pnl_batch_d).item()
                    full_val_loss_running += batch_loss * X_batch_d.size(0)
                    full_val_count_running += X_batch_d.size(0)
                else:
                    # Per-sample loss (BCE/GCE/Brier) so we can aggregate by fold.
                    # Val loss MUST use the same loss family training uses,
                    # otherwise SWA snapshot selection (gated on val_loss
                    # improvement) compares apples to oranges.
                    if brier_active:
                        per_sample = (pred - y_batch_d) ** 2
                    elif gce_active:
                        per_sample = _gce_loss(pred, y_batch_d, gce_q)
                    else:
                        per_sample = criterion(pred, y_batch_d)
                    val_per_sample_losses_chunks.append(per_sample.detach().cpu())
                    full_val_loss_running += per_sample.sum().item()
                    full_val_count_running += X_batch_d.size(0)

        # Full-val mean loss (still computed for logging and as fallback).
        full_val_loss = (full_val_loss_running
                         / max(full_val_count_running, 1))

        # Per-fold losses (BCE/GCE path only — non-BCE paths don't compose
        # cleanly with chronological-fold partitioning of the loss).
        per_fold_losses = []
        if multifold_active and val_per_sample_losses_chunks:
            all_per_sample = torch.cat(val_per_sample_losses_chunks).numpy()
            for fold_idx in fold_indices_list:
                # Guard against any out-of-range index from K-partition
                # vs actual val length mismatches (should never happen
                # but defensive).
                if len(fold_idx) == 0 or fold_idx[-1] >= len(all_per_sample):
                    continue
                per_fold_losses.append(float(all_per_sample[fold_idx].mean()))

        # Selection metric: aggregate per-fold losses by the configured
        # aggregator when we have >=2 folds; else fall back to full-val
        # mean (legacy behavior). Median is the default — suppresses
        # outlier folds without forcing convergence to epoch 0 the way
        # max does (max selects epoch 0 because per-fold losses
        # NECESSARILY diverge as the model starts learning, and the
        # earliest-chronological fold is the most distribution-shifted
        # vs train).
        if len(per_fold_losses) >= 2:
            agg = str(multifold_aggregator).lower()
            if agg == 'max':
                selection_val_loss = max(per_fold_losses)
            elif agg == 'mean':
                selection_val_loss = float(np.mean(per_fold_losses))
            else:  # 'median' default
                selection_val_loss = float(np.median(per_fold_losses))
        else:
            selection_val_loss = full_val_loss

        # Keep `val_loss` symbol for unchanged-downstream logging strings.
        val_loss = full_val_loss
        scheduler.step(selection_val_loss)

        if selection_val_loss < best_val_loss:
            best_val_loss = selection_val_loss
            best_loss_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
            if len(swa_states) > swa_window:
                swa_states.pop(0)
            patience_counter = 0
        else:
            patience_counter += 1

        if curriculum_active:
            curriculum_epoch_log.append({
                'epoch': int(epoch),
                'phase': cur_phase,
                'win_thr': float(cur_win_thr),
                'loss_thr': float(cur_loss_thr),
                'batches_kept': int(curriculum_batches_kept),
                'batches_skipped': int(curriculum_batches_skipped),
                'samples_kept': int(curriculum_samples_kept),
                'samples_seen': int(curriculum_samples_seen),
            })

        if verbose and epoch % 5 == 0:
            msg = (f'  Epoch {epoch}: train={train_loss/len(train_loader):.4f} '
                   f'val_loss={val_loss:.4f}')
            if multifold_active and len(per_fold_losses) >= 2:
                fold_str = '/'.join(f'{x:.3f}' for x in per_fold_losses)
                msg += f' folds=[{fold_str}] sel({str(multifold_aggregator).lower()})={selection_val_loss:.4f}'
            if daily_rank_active and rank_batches_with_pairs > 0:
                avg_rank = rank_loss_sum_epoch / max(rank_batches_with_pairs, 1)
                msg += (f' rank_loss={avg_rank:.4f} '
                        f'pairs={rank_pair_count_epoch} '
                        f'batches_with_pairs={rank_batches_with_pairs}')
            if curriculum_active:
                keep_frac = (curriculum_samples_kept
                             / max(curriculum_samples_seen, 1))
                msg += (f' cur[{cur_phase}] '
                        f'thr=(>{cur_win_thr:+.3f}|<{cur_loss_thr:+.3f}) '
                        f'keep={curriculum_samples_kept}/'
                        f'{curriculum_samples_seen} '
                        f'({100.0*keep_frac:.1f}%) '
                        f'skip_bat={curriculum_batches_skipped}')
            print(msg)

        if patience_counter >= patience:
            if verbose:
                print(f'  Early stopping at epoch {epoch}')
            break

    # Restoration: SWA avg if >=2 snapshots available, else best val_loss state.
    used = 'none'
    if len(swa_states) >= 2:
        avg_state = {}
        for key in swa_states[0]:
            t0 = swa_states[0][key]
            if t0.dtype.is_floating_point:
                stacked = torch.stack([s[key].float() for s in swa_states], dim=0)
                avg_state[key] = stacked.mean(dim=0).to(t0.dtype)
            else:
                avg_state[key] = best_loss_state[key].clone()
        model.cpu()
        model.load_state_dict(avg_state)
        sel_metric = (f'{str(multifold_aggregator).lower()}-fold val_loss'
                      if multifold_active else 'val_loss')
        used = f'SWA avg over {len(swa_states)} {sel_metric} snapshots'
    elif best_loss_state is not None:
        model.cpu()
        model.load_state_dict(best_loss_state)
        sel_metric = (f'{str(multifold_aggregator).lower()}-fold'
                      if multifold_active else 'val')
        used = f'best {sel_metric}_loss={best_val_loss:.4f}'
    else:
        model.cpu()

    if verbose:
        print(f'  Selected weights: {used}')

    # Stage A2 (THIS ATTEMPT #318): RECENT-WINDOW DECISION-HEAD ADAPTATION.
    # Runs AFTER SWA averaging so the encoder we freeze is the smoothed
    # weight-average (the "best ensemble of late-epoch snapshots") not any
    # single noisy iterate. The FC head is then re-fit on the most-recent
    # recent_ft_days of train, adapting the decision boundary to the
    # regime closest to test. Bias calibration (Stage B) runs after this,
    # so the precision-targeted threshold is computed against the
    # recent-FT-shifted predictions and composes cleanly.
    if use_recent_finetune and bce_path_active:
        recent_ft_info = _finetune_on_recent(
            model, X_train_scaled, y_train, dates_train,
            n_recent_days=recent_ft_days,
            epochs=recent_ft_epochs,
            lr=recent_ft_lr,
            batch_size=batch_size,
            verbose=verbose)
        # _finetune_on_recent moved the model back to DEVICE for training;
        # bring it back to CPU so the calibration helper (which expects
        # CPU tensors) and the saved model artifact see the standard form.
        model.cpu()
    else:
        recent_ft_info = {
            'applied': False,
            'reason': ('disabled by flag' if not use_recent_finetune
                       else 'non-BCE path'),
        }

    # Stage A3 (THIS ATTEMPT #327): POST-SWA TEMPERATURE SCALING
    # (Guo et al. 2017, ICML). Refits the model's _output_temp buffer
    # to minimize binary-cross-entropy NLL on val. Pure monotone
    # post-hoc 1D fit — score ORDERING is unchanged. Runs AFTER any
    # recent-window FT (so the temperature is fit to the final
    # decision boundary the bias calibration will see) and BEFORE
    # bias calibration (so b_shift = T * Δlogit uses the right T).
    # See _temperature_scale docstring for the full rationale.
    if use_temperature_scaling and bce_path_active:
        temp_info = _temperature_scale(
            model, X_val_scaled, y_val,
            batch_size=batch_size,
            T_min=float(temperature_min),
            T_max=float(temperature_max),
            min_nll_improvement=float(temperature_min_nll_improvement),
            verbose=verbose)
    else:
        temp_info = {
            'applied': False,
            'reason': ('disabled by flag' if not use_temperature_scaling
                       else 'non-BCE path'),
        }

    # Stage B: PRECISION-TARGETED bias calibration (THIS ATTEMPT).
    # Previously: shifted the FC bias so the (1 - TARGET_VAL_SELECTIVITY)
    # quantile of val predictions landed at LIVE_THRESHOLD=0.6 — i.e., top
    # 10% by predicted score clears the gate regardless of whether those
    # samples are actually higher-precision than the rest.
    # Now: y_val is used to find the highest threshold where empirical
    # val PRECISION >= precision_target (default 0.30, ≈3x base rate)
    # subject to n_flags/N in [2%, 25%]. That threshold maps to 0.6.
    # Each WF split computes its own precision-optimal threshold against
    # its own val period, so the decision surface self-adapts per split.
    # Falls back to the old selectivity calibration when no threshold
    # clears the target (e.g., degenerate scores or very weak signal).
    calib_info = _calibrate_fc_bias(
        model, X_val_scaled, y_val=y_val, pnl_val=pnl_val,
        batch_size=batch_size,
        use_ev_calibration=bool(use_ev_calibration),
        ev_commission=float(ev_commission),
        verbose=verbose)

    return model, scaler, {
        'best_val_loss': best_val_loss,
        'selector_used': used,
        'epochs': epoch + 1,
        'pos_weight': pos_weight,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'use_attention': use_attention,
        'survival_weight': survival_weight,
        'fp_weight': fp_weight,
        'sequence_normalize': sequence_normalize,
        'mixup_alpha': mixup_alpha,
        'swa_snapshots': len(swa_states),
        'calibration': calib_info,
        'optimizer': 'Adam',
        'covariate_shift': shift_info,
        'dann': dann_info,
        'ss_pretrain': ss_pretrain_info,
        'supcon_pretrain': supcon_info,
        'mean_teacher': mt_info,
        'daily_rank': daily_rank_info,
        'curriculum': curriculum_info,
        'curriculum_epoch_log': curriculum_epoch_log[:10],
        'gce': gce_info,
        'quantile_aux': qr_info,
        'temporal_consist': consist_info,
        'multifold_select': multifold_info,
        'selective_head': sel_info,
        'time_decay': td_info,
        'recent_finetune': recent_ft_info,
        'irm': irm_info,
        'pnl_smooth_labels': pnl_smooth_info,
        'sortino_aux': sortino_info,
        'r_drop': r_drop_info,
        'stationarity_mask': stationarity_info,
        'temperature_scaling': temp_info,
        'manifold_mixup': manifold_mixup_info,
        'target_aux': target_aux_info,
        'brier': brier_info,
        'label_cleaning': label_cleaning_info,
        'window_warp': window_warp_info,
        'time_mask': time_mask_info,
        'grad_noise': (
            {**grad_noise_info,
             'n_steps_total': int(grad_noise_step),
             'n_steps_diag_sampled': int(grad_noise_diag['n_steps_injected']),
             'avg_grad_norm_first_200': (
                 float(grad_noise_diag['grad_norm_sum']
                       / max(grad_noise_diag['n_steps_injected'], 1))
                 if grad_noise_diag['n_steps_injected'] > 0 else None),
             'avg_noise_norm_first_200': (
                 float(grad_noise_diag['noise_norm_sum']
                       / max(grad_noise_diag['n_steps_injected'], 1))
                 if grad_noise_diag['n_steps_injected'] > 0 else None),
             'noise_to_grad_ratio_first_200': (
                 float(grad_noise_diag['noise_norm_sum']
                       / max(grad_noise_diag['grad_norm_sum'], 1e-9))
                 if grad_noise_diag['grad_norm_sum'] > 0 else None),
             }),
        'loss_mode': ('xgb_distill' if use_xgb_distill
                      else ('pnl_profit' if use_pnl_loss
                            else (('brier' if brier_info.get('applied')
                                   else ('gce' if gce_info.get('applied') else 'bce'))
                                  + ('_target_aux' if target_aux_info.get('applied') else '')
                                  + ('_time_mask' if time_mask_info.get('applied') else '')
                                  + ('_window_warp' if window_warp_info.get('applied') else '')
                                  + ('_label_cleaning' if label_cleaning_info.get('applied') else '')
                                  + ('_manifold_mixup' if manifold_mixup_info.get('applied') else '')
                                  + ('_temp_scale' if temp_info.get('applied') else '')
                                  + ('_stationarity_mask' if stationarity_info.get('applied') else '')
                                  + ('_r_drop' if r_drop_info.get('applied') else '')
                                  + ('_ev_calib' if (calib_info.get('strategy') == 'ev_targeted') else '')
                                  + ('_grad_noise' if grad_noise_info.get('applied') else '')
                                  + ('_sortino_aux' if sortino_info.get('applied') else '')
                                  + ('_pnl_smooth' if pnl_smooth_info.get('applied') else '')
                                  + ('_irm' if irm_info.get('applied') else '')
                                  + ('_recent_ft' if recent_ft_info.get('applied') else '')
                                  + ('_time_decay' if td_info.get('applied') else '')
                                  + ('_selective_head' if sel_info.get('applied') else '')
                                  + ('_multifold_select' if multifold_info.get('applied') else '')
                                  + ('_dann' if dann_info.get('applied') else '')
                                  + ('_covariate_shift' if shift_info.get('applied') else '')
                                  + ('_curriculum' if curriculum_info.get('applied') else '')
                                  + ('_listnet_rank' if (daily_rank_info.get('applied')
                                                          and daily_rank_info.get('variant') == 'listnet_topk')
                                     else ('_daily_rank' if daily_rank_info.get('applied') else ''))
                                  + ('_quantile_aux' if qr_info.get('applied') else '')
                                  + ('_temporal_consist' if consist_info.get('applied') else '')
                                  + ('_mean_teacher' if mt_info.get('applied') else '')
                                  + ('_ss_pretrain' if ss_pretrain_info.get('applied') else '')
                                  + ('_supcon_pretrain' if supcon_info.get('applied') else '')))),
        'distill_alpha': distill_alpha if use_xgb_distill else None,
        'pnl_loss_scale': pnl_loss_scale if use_pnl_loss else None,
        'pnl_budget_lambda': pnl_budget_lambda if use_pnl_loss else None,
        'pnl_target_selectivity': pnl_target_selectivity if use_pnl_loss else None,
        'pnl_commission': pnl_commission if use_pnl_loss else None,
        'daily_rank_lambda': daily_rank_lambda if daily_rank_info.get('applied') else None,
        'daily_rank_margin': daily_rank_margin if daily_rank_info.get('applied') else None,
        'daily_rank_min_per_day': daily_rank_min_per_day if daily_rank_info.get('applied') else None,
    }


def _temperature_scale(model, X_val_scaled, y_val, batch_size=256,
                        T_min=0.3, T_max=5.0,
                        min_nll_improvement=0.01,
                        verbose=False):
    """Post-SWA temperature scaling (THIS ATTEMPT #327, Guo et al. 2017 ICML).

    Fits a single scalar T to minimize binary cross-entropy NLL on the val
    set, with logits transformed as sigmoid(logit / T). Updates the model's
    `_output_temp` buffer in place. Pure post-hoc monotone calibration:
    score ORDERING is preserved exactly, only the spread of confidence
    values changes.

    Strategy:
      1. Run val through model.get_logits(...) on CPU (model is post-SWA,
         already moved to CPU by the caller).
      2. Use scipy.optimize.minimize_scalar (method='bounded') to find
         T* in [T_min, T_max] minimizing -E[y*log(sigmoid(z/T))
         + (1-y)*log(1-sigmoid(z/T))].
      3. Apply T* only if val NLL improved by >= min_nll_improvement
         (relative). Otherwise leave the model's _output_temp unchanged.

    Why a relative-improvement gate?
      - At init, _output_temp = 0.5 (a sharpening prior). For very small
        val (just above MIN_VAL_FOR_CALIBRATION), the NLL surface around
        T=0.5 is nearly flat; tiny noise can push T to a slightly
        different value with negligible real benefit but visible bias-
        shift consequences downstream. The 1% relative-improvement gate
        keeps the post-hoc fit honest — we only intervene when there's
        a meaningful calibration win on val.

    Falls through to a no-op if scipy is unavailable (degrades to a
    coarse grid search in pure numpy).
    """
    if len(X_val_scaled) < MIN_VAL_FOR_CALIBRATION or y_val is None:
        return {'applied': False,
                'reason': f'val too small ({len(X_val_scaled)} < {MIN_VAL_FOR_CALIBRATION})'
                          if y_val is not None else 'no y_val'}

    model.eval()
    device = next(model.parameters()).device
    logits_chunks = []
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(X_val_t), batch_size):
            batch = X_val_t[start:start + batch_size]
            # get_logits returns z (raw logit). The model normally applies
            # sigmoid(z / _output_temp); we want to refit that T against
            # val NLL of sigmoid(z / T).
            logits_chunks.append(model.get_logits(batch).detach().cpu().numpy().ravel())
    logits = np.concatenate(logits_chunks).astype(np.float64)
    y_val_arr = np.asarray(y_val).ravel().astype(np.float64)
    if logits.shape[0] != y_val_arr.shape[0]:
        return {'applied': False,
                'reason': f'logits/y_val shape mismatch: {logits.shape[0]} vs {y_val_arr.shape[0]}'}

    T_init = float(model._output_temp.item()) if hasattr(model, '_output_temp') else 1.0

    def _nll(T):
        T_safe = max(float(T), 1e-3)
        # log(sigmoid(z/T)) and log(1 - sigmoid(z/T)) via the numerically
        # stable softplus identity, avoiding overflow in exp(-z/T).
        z = logits / T_safe
        # log_sigmoid(z) = -softplus(-z); log(1-sigmoid(z)) = -softplus(z).
        log_p = -np.logaddexp(0.0, -z)
        log_1mp = -np.logaddexp(0.0, z)
        return float(-(y_val_arr * log_p + (1.0 - y_val_arr) * log_1mp).mean())

    nll_init = _nll(T_init)

    T_opt = T_init
    nll_opt = nll_init
    method_used = 'no-op'

    try:
        from scipy.optimize import minimize_scalar  # noqa: WPS433
        res = minimize_scalar(
            _nll, bounds=(float(T_min), float(T_max)),
            method='bounded', options={'xatol': 1e-3})
        if res is not None and np.isfinite(res.fun):
            T_opt_candidate = float(res.x)
            nll_opt_candidate = float(res.fun)
            if nll_opt_candidate < nll_opt:
                T_opt = T_opt_candidate
                nll_opt = nll_opt_candidate
                method_used = 'scipy-bounded'
    except Exception as exc:
        # Fallback: 200-point log-spaced grid in [T_min, T_max].
        if verbose:
            print(f'  Temperature scaling: scipy unavailable ({exc}) — using grid fallback')
        grid = np.exp(np.linspace(np.log(float(T_min)), np.log(float(T_max)), 200))
        nll_grid = np.array([_nll(t) for t in grid])
        best_idx = int(nll_grid.argmin())
        T_opt_candidate = float(grid[best_idx])
        nll_opt_candidate = float(nll_grid[best_idx])
        if nll_opt_candidate < nll_opt:
            T_opt = T_opt_candidate
            nll_opt = nll_opt_candidate
            method_used = 'grid-200pt'

    nll_reduction_rel = (nll_init - nll_opt) / max(nll_init, 1e-9)
    applied = (nll_reduction_rel >= float(min_nll_improvement)
               and abs(T_opt - T_init) > 1e-3)
    if applied:
        with torch.no_grad():
            model._output_temp.fill_(float(T_opt))

    info = {
        'applied': bool(applied),
        'T_init': float(T_init),
        'T_opt': float(T_opt),
        'T_final': float(T_opt) if applied else float(T_init),
        'nll_init': float(nll_init),
        'nll_opt': float(nll_opt),
        'nll_reduction_rel': float(nll_reduction_rel),
        'min_nll_improvement': float(min_nll_improvement),
        'method': method_used,
        'T_min': float(T_min),
        'T_max': float(T_max),
        'val_size': int(len(y_val_arr)),
    }
    if verbose:
        marker = ('APPLIED' if applied
                  else f'skipped (Δrel={nll_reduction_rel*100:.2f}% < '
                       f'{float(min_nll_improvement)*100:.1f}% gate '
                       f'or |T-T0|<1e-3)')
        print(f'  Temperature scaling ({method_used}): T={T_init:.3f} -> {T_opt:.3f} '
              f'NLL {nll_init:.4f} -> {nll_opt:.4f} '
              f'(Δrel={nll_reduction_rel*100:+.2f}%) [{marker}]')
    return info


def _calibrate_fc_bias(model, X_val_scaled, y_val=None, pnl_val=None,
                       batch_size=256,
                       precision_target=0.30,
                       min_flags_frac=0.02, max_flags_frac=0.25,
                       use_ev_calibration=True,
                       ev_commission=0.011,
                       verbose=False):
    """EV-targeted FC bias calibration (THIS ATTEMPT #324).

    Replaces the precision-targeted threshold search (Entry 308 baseline)
    with one that maximizes empirical EV — mean realized PnL net of
    commission — on the val window. Same monotone bias-shift mechanism:
    once the EV-optimal threshold T* is found, the FC bias shifts so
    that T* maps to LIVE_THRESHOLD=0.6 in probability space.

    Why EV-targeting attacks the cross-split-WR-variance plateau
    -----------------------------------------------------------
      1. The gate's hard rules include EV >= -1% and "WR must beat
         market base rate". With the +15% target / -3% SL asymmetric
         payoff, EV > 0 implies alpha > 0 mechanically. So optimizing
         empirical val EV directly aligns with what the gate measures
         — more directly than precision, which ignores trade magnitude.
      2. Commission is 1.1% round-trip per trade. Precision-targeted
         search picks the threshold with highest val precision but is
         BLIND to commission drag. EV-targeted search subtracts
         commission per trade, so the EV-optimal threshold is typically
         HIGHER than the precision-optimal threshold — fewer, more
         confident trades. Targeting matches the trading-cost analysis
         in CLAUDE.md ("target 30-100 trades per split", not 1000s).
      3. Per-split self-adaptation: a weak split (low expected EV at
         every threshold) pushes T* high → very selective → few trades
         at higher precision; a strong split allows a lower T* → more
         trades at moderate precision. This NATURALLY reduces cross-
         split WR variance (the 12.2% std killer in Entry 197) without
         imposing a one-size-fits-all selectivity cap.

    Why this is NOT one of the already-rejected mechanisms
    -----------------------------------------------------
      * #199 PnL-magnitude weighted BCE (REJECTED): scaled per-sample
        loss by |pnl| during TRAINING. EV calibration leaves training
        entirely untouched; only the post-SWA FC bias shifts.
      * #290 differentiable PnL-optimization loss (REJECTED): replaced
        BCE with -E[pred·(pnl-commission)] as the training objective.
        EV calibration is POST-HOC threshold selection on the trained
        model. The model is still optimizing standard class-balanced
        BCE; only the decision boundary's location shifts.
      * Precision-targeted calibration (Entry 308 baseline, ACTIVE):
        same monotone bias-shift mechanism, different objective. EV
        targeting incorporates trade MAGNITUDES (continuous PnL) and
        COMMISSION (a constant per-trade drag) — both information
        sources that precision targeting throws away.
      * Selectivity calibration (Entry 289, post-hoc): forces a fixed
        10% selectivity regardless of signal strength. EV calibration
        ADAPTS selectivity per split based on the val EV curve.
      * Listed as UNTRIED option (6) in the condensed lessons:
        "Calibration-based selection (use predicted EV × probability
         rather than fixed 0.6 threshold)."

    Mechanism
    ---------
      1. Compute val predictions from the SWA-restored (post-Stage-A)
         model (or post-recent-FT model when that mechanism is on).
      2. Build a dense quantile grid (50th -> 99.5th percentile of raw
         predictions, 100 candidate thresholds).
      3. For each candidate T in the [min_flags_frac, max_flags_frac]
         flag-count band, compute empirical EV on val:
             ev(T) = mean( pnl_val[pred>=T] ) - ev_commission
         where pnl_val is the per-sample realized P&L (already net of
         commission via labels.py — but the calibration uses an explicit
         commission term so it can be tuned independently if needed).
      4. Pick T* = argmax_{T in band} ev(T). Tie-break by HIGHER T
         (stricter selection — more robust under regime shift).
      5. Verify T* gives ev(T*) > base_ev (mean PnL on whole val), so
         the calibration only fires when there's genuine edge. If not,
         fall through to the precision-targeted path.

    Fallback hierarchy
    ------------------
      - No pnl_val OR use_ev_calibration=False -> precision-targeted.
      - No threshold in the flag-count band has ev > base_ev
        -> precision-targeted.
      - Precision search also fails -> selectivity calibration.
      - All paths share the same monotone bias-shift formula.

    Why this is SAFE
    ----------------
      - Saved model has IDENTICAL signature to Entry 197. Gate loader,
        live trader, scaler are all unchanged. Only the FC bias differs
        from what precision-targeted calibration would have set it to.
      - The shift is at most O(few units) in logit space (we cap the
        absolute b_shift at +/-5.0 below to avoid degenerate score
        compression). The SWA-restored model's score ORDERING is
        completely preserved — only the location of the 0.6 boundary
        on that ordering moves.
      - Degrades gracefully: missing pnl_val falls through to the
        existing precision-targeted path, which itself falls through
        to selectivity. The whole calibration stack is wrapped in
        N>=MIN_VAL_FOR_CALIBRATION and degenerate-pred guards.

    --- LEGACY HEADER (precision-targeted) ---

    Previously: shifted FC bias so the top TARGET_VAL_SELECTIVITY of val
    predictions (a fixed 10%) landed at LIVE_THRESHOLD=0.6. Pure count-
    based — no use of y_val at calibration time, so "the top 10%" might
    equally well contain 15%-precision or 40%-precision trades depending
    on how well calibrated the raw model is.

    Now (when y_val is supplied):
      1. Compute val predictions from the SWA-restored model.
      2. Scan candidate thresholds (dense quantile grid between the 50th
         and 99th percentile of raw predictions).
      3. For each threshold T, compute empirical precision among val
         samples with pred >= T, subject to flag count in
         [min_flags_frac, max_flags_frac] * N_val. This band excludes
         over-noisy ultra-high thresholds and under-selective low ones.
      4. Pick the threshold with highest empirical precision (ties broken
         by the HIGHER threshold — preferring stricter selection).
      5. Shift FC bias so that threshold maps to LIVE_THRESHOLD=0.6,
         using the same monotone logit-shift formula:
             b_shift = T_temp * (logit(0.6) - logit(chosen_threshold))

    Why this attacks the 37.5%-WR plateau
    -------------------------------------
      - The WF gate's hard rule is "WR must beat market base rate
        across ALL 7 splits". Its signal is WR, not selectivity. The
        old calibration optimized selectivity — a proxy that diverges
        from WR whenever the raw score distribution is not uniformly
        informative across its tail. Precision-targeted calibration
        chooses the threshold DIRECTLY on the metric the gate judges.
      - Splits with weak signal naturally end up with higher threshold
        → fewer trades at higher precision, which is exactly the right
        response to regime uncertainty. Splits with strong signal push
        the threshold LOWER — trading more when the model is clearly
        right. Old calibration forced 10% selectivity regardless of
        underlying signal strength.
      - The [2%, 25%] flag-count band guarantees: at least enough val
        flags to have meaningful precision estimates AND enough trades
        on test to clear MIN_TRADES_PER_SPLIT=10 (2% of ~5K val trades
        ≈ 100 flagged, scaling proportionally to test-split samples).

    Why this is NOT one of the already-rejected calibrations
    --------------------------------------------------------
      * Post-training quantile calibration to 10% selectivity (current
        baseline — Entry 289) uses ONLY the score ordering. This
        change uses the actual LABELS at calibration time.
      * FP-penalty / survival-weighting (#47, #56, #198, #215, #218,
        REJECTED) tried to SHAPE TRAINING so high-confidence predictions
        were more selective. Repeatedly collapsed scores or yielded 0
        trades. This change leaves training untouched — the raw model's
        learned score ORDERING is preserved — and only shifts where
        the decision boundary falls on that ordering.
      * Precision@0.6 early stopping (#63, #64, #67, REJECTED) tried
        to pick a training epoch where the 0.6 threshold produced
        good precision. Unstable because (a) 0.6 is an arbitrary
        threshold for mid-training scores, and (b) it gated on a
        RARE-event metric computed on a small held-out slice, which
        is noisy. This change uses the WHOLE val set and explicitly
        searches over thresholds instead of epochs.

    Fallback hierarchy (robust against degenerate training):
      - No y_val → selectivity calibration (old path, unchanged).
      - Precision search finds no threshold in the flag-count band with
        precision > base rate → selectivity calibration.
      - Degenerate predictions (all equal) → skip calibration.

    Silent no-op when val is too small (< MIN_VAL_FOR_CALIBRATION).
    """
    if len(X_val_scaled) < MIN_VAL_FOR_CALIBRATION:
        if verbose:
            print(f'  Bias calibration skipped: val size {len(X_val_scaled)} < {MIN_VAL_FOR_CALIBRATION}')
        return {'applied': False, 'reason': 'val too small'}

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        for start in range(0, len(X_val_tensor), batch_size):
            batch = X_val_tensor[start:start + batch_size]
            val_preds_list.append(model(batch).cpu().numpy().ravel())
    val_preds = np.concatenate(val_preds_list)

    eps = 1e-6
    val_preds_c = np.clip(val_preds, eps, 1 - eps)
    N = len(val_preds_c)

    # ===== EV-TARGETED SEARCH (THIS ATTEMPT #324) =====
    # When pnl_val and use_ev_calibration are both supplied, find the
    # threshold T* that maximizes empirical val EV among flag-count band
    # candidates. The WF gate passes pnl_val unconditionally, so this
    # path activates by default. Falls through to precision-targeted
    # below if no positive-EV threshold is found.
    chosen_threshold = None
    search_info = {'strategy': 'selectivity_default'}
    y_val_arr = None
    pnl_val_arr = None

    if (use_ev_calibration and pnl_val is not None
            and y_val is not None
            and len(pnl_val) == N
            and len(y_val) == N):
        y_val_arr = np.asarray(y_val).ravel().astype(np.float32)
        pnl_val_arr = np.asarray(pnl_val).ravel().astype(np.float32)
        base_rate = float(y_val_arr.mean())
        base_ev = float(pnl_val_arr.mean()) - float(ev_commission)

        # Dense quantile grid 50th -> 99.5th percentile. Going to 99.5
        # (vs 99 in precision path) lets the EV search reach more
        # selective thresholds — useful when only the very top decile
        # has positive EV.
        q_levels = np.linspace(0.50, 0.995, 100)
        thr_candidates = np.unique(np.quantile(val_preds_c, q_levels))

        min_flags = max(int(min_flags_frac * N), 20)
        max_flags = max(int(max_flags_frac * N), min_flags + 1)

        best_ev = -float('inf')
        best_thr = None
        best_flags = 0
        best_prec = 0.0
        best_mean_pnl = 0.0
        n_scanned = 0
        for thr in thr_candidates:
            flags = val_preds_c >= thr
            n_flags = int(flags.sum())
            if n_flags < min_flags or n_flags > max_flags:
                continue
            n_scanned += 1
            mean_pnl = float(pnl_val_arr[flags].mean())
            ev = mean_pnl - float(ev_commission)
            prec = float(y_val_arr[flags].mean())
            # Maximize EV. Tie-break by HIGHER threshold (stricter,
            # more robust under regime shift), then by higher precision.
            if (ev > best_ev + 1e-6) or (
                    abs(ev - best_ev) <= 1e-6
                    and (best_thr is None or thr > best_thr)):
                best_ev = ev
                best_thr = float(thr)
                best_flags = n_flags
                best_prec = prec
                best_mean_pnl = mean_pnl

        search_info = {
            'strategy': 'ev_targeted',
            'base_rate_val': base_rate,
            'base_ev_val': base_ev,
            'commission': float(ev_commission),
            'best_val_ev': best_ev,
            'best_val_mean_pnl': best_mean_pnl,
            'best_val_precision': best_prec,
            'best_threshold_raw': best_thr,
            'best_n_flags': best_flags,
            'min_flags': min_flags,
            'max_flags': max_flags,
            'n_thresholds_scanned': n_scanned,
            'N_val': N,
        }

        # Require positive lift over the baseline EV (mean PnL on the
        # whole val window minus commission). If best_ev <= base_ev,
        # selecting a sub-population with this threshold would produce
        # WORSE expected returns than trading every val sample at the
        # same commission rate — no edge, fall through.
        if best_thr is not None and best_ev > base_ev:
            chosen_threshold = best_thr
            if verbose:
                print(f'  Bias calibration: EV-TARGETED (this attempt #324)')
                print(f'    val base_rate={base_rate:.3f}, base_ev={base_ev:+.4f}')
                print(f'    Scanned {n_scanned} thresholds in '
                      f'[{min_flags}, {max_flags}] flag band')
                print(f'    Best val EV={best_ev:+.4f} '
                      f'(lift={best_ev - base_ev:+.4f}) '
                      f'mean_pnl={best_mean_pnl:+.4f} '
                      f'precision={best_prec:.3f}')
                print(f'    @ thr={best_thr:.4f}, n_flags={best_flags}/{N} '
                      f'({100*best_flags/N:.1f}%)')
        elif verbose:
            print(f'  Bias calibration: EV search found no threshold > base_ev '
                  f'({base_ev:+.4f}) in flag band — falling back to precision-targeted')

    # Precision-targeted search (active when y_val supplied).
    if chosen_threshold is None and y_val is not None and len(y_val) == N:
        if y_val_arr is None:
            y_val_arr = np.asarray(y_val).ravel().astype(np.float32)
        base_rate = float(y_val_arr.mean())

        # Dense quantile grid from 50th→99th percentile of val preds
        # (100 points). Converting quantiles rather than linspacing
        # on raw scores automatically adapts to whatever score
        # distribution the model produces.
        q_levels = np.linspace(0.50, 0.99, 100)
        thr_candidates = np.unique(np.quantile(val_preds_c, q_levels))

        min_flags = max(int(min_flags_frac * N), 20)
        max_flags = max(int(max_flags_frac * N), min_flags + 1)

        best_prec = -1.0
        best_thr = None
        best_flags = 0
        n_scanned = 0
        for thr in thr_candidates:
            flags = val_preds_c >= thr
            n_flags = int(flags.sum())
            if n_flags < min_flags or n_flags > max_flags:
                continue
            n_scanned += 1
            prec = float(y_val_arr[flags].mean())
            # Higher precision wins. Ties broken by higher threshold
            # (stricter → more robust under regime shift).
            if (prec > best_prec + 1e-6) or (abs(prec - best_prec) <= 1e-6 and
                                              (best_thr is None or thr > best_thr)):
                best_prec = prec
                best_thr = float(thr)
                best_flags = n_flags

        search_info = {
            'strategy': 'precision_targeted',
            'base_rate_val': base_rate,
            'precision_target': precision_target,
            'best_val_precision': best_prec,
            'best_threshold_raw': best_thr,
            'best_n_flags': best_flags,
            'min_flags': min_flags,
            'max_flags': max_flags,
            'n_thresholds_scanned': n_scanned,
            'N_val': N,
        }

        # Require that the winning precision beats base rate. If it
        # doesn't, the model has no edge at any flag-rate in the band
        # — fall back to selectivity so at least the count behavior
        # matches the previous baseline.
        if best_thr is not None and best_prec > base_rate:
            chosen_threshold = best_thr
            if verbose:
                alpha_val = best_prec - base_rate
                print(f'  Bias calibration: PRECISION-TARGETED')
                print(f'    val base_rate={base_rate:.3f}, target_prec={precision_target:.2f}')
                print(f'    Scanned {n_scanned} thresholds in '
                      f'[{min_flags}, {max_flags}] flag band')
                print(f'    Best val precision={best_prec:.3f} (alpha={alpha_val:+.3f}) '
                      f'@ thr={best_thr:.4f}, n_flags={best_flags}/{N} '
                      f'({100*best_flags/N:.1f}%)')
                if best_prec < precision_target:
                    print(f'    Note: best precision below target {precision_target:.2f} '
                          '— using best available (still > base rate)')
        else:
            if verbose:
                print(f'  Bias calibration: precision search found no threshold > base_rate '
                      f'({base_rate:.3f}) in flag band — falling back to selectivity')

    # Selectivity fallback (or default when y_val is None).
    if chosen_threshold is None:
        chosen_threshold = float(np.quantile(val_preds_c, 1.0 - TARGET_VAL_SELECTIVITY))
        search_info['strategy'] = 'selectivity_fallback' if y_val_arr is not None else 'selectivity_default'
        search_info['selectivity_target'] = TARGET_VAL_SELECTIVITY

    # Degeneracy guard.
    if chosen_threshold <= eps * 2 or chosen_threshold >= 1 - eps * 2:
        if verbose:
            print(f'  Bias calibration skipped: degenerate threshold ({chosen_threshold:.4f})')
        return {'applied': False, 'reason': 'degenerate preds',
                'chosen_threshold': chosen_threshold, **search_info}

    # Locate the FC layer (may be Sequential if head_dims was used).
    if isinstance(model.fc, nn.Linear):
        final_layer = model.fc
    else:
        final_layer = model.fc[-1]

    T = float(model._output_temp.item()) if hasattr(model, '_output_temp') else 1.0
    logit_target = float(np.log(LIVE_THRESHOLD / (1.0 - LIVE_THRESHOLD)))
    logit_qraw = float(np.log(chosen_threshold / (1.0 - chosen_threshold)))
    b_shift_raw = T * (logit_target - logit_qraw)
    # Cap b_shift to +/-5 logit units. With T<=1, this caps the practical
    # bias movement to +/-5 in logit space — a 99.3%/0.7% probability
    # range. Larger shifts indicate the calibration is fighting a model
    # whose score distribution is fundamentally pathological; cap and log.
    b_shift = float(np.clip(b_shift_raw, -5.0, 5.0))
    b_shift_clipped = (abs(b_shift_raw) > 5.0)

    with torch.no_grad():
        final_layer.bias.data += b_shift

    # Verify post-calibration metrics for logging.
    val_preds_post_list = []
    with torch.no_grad():
        for start in range(0, len(X_val_tensor), batch_size):
            batch = X_val_tensor[start:start + batch_size]
            val_preds_post_list.append(model(batch).cpu().numpy().ravel())
    val_preds_post = np.concatenate(val_preds_post_list)
    post_flags = val_preds_post >= LIVE_THRESHOLD
    post_sel = float(post_flags.mean())
    if y_val_arr is not None and post_flags.sum() > 0:
        post_prec = float(y_val_arr[post_flags].mean())
    else:
        post_prec = float('nan')
    # Post-cal EV at the LIVE_THRESHOLD — the metric that mirrors the
    # WF gate's per-trade EV check.
    if pnl_val_arr is not None and post_flags.sum() > 0:
        post_mean_pnl = float(pnl_val_arr[post_flags].mean())
        post_ev = post_mean_pnl - float(ev_commission)
    else:
        post_mean_pnl = float('nan')
        post_ev = float('nan')

    if verbose:
        print(f'  Bias calibration: chosen_thr={chosen_threshold:.4f}, '
              f'T={T:.2f}, b_shift={b_shift:+.4f}'
              + (' (CLIPPED from ' + f'{b_shift_raw:+.4f})' if b_shift_clipped else ''))
        print(f'    Pre-cal  @ 0.6 selectivity: {(val_preds >= LIVE_THRESHOLD).mean():.3f}')
        print(f'    Post-cal @ 0.6 selectivity: {post_sel:.3f}')
        if not np.isnan(post_prec):
            print(f'    Post-cal @ 0.6 precision:  {post_prec:.3f}')
        if not np.isnan(post_ev):
            print(f'    Post-cal @ 0.6 EV:         {post_ev:+.4f} '
                  f'(mean_pnl={post_mean_pnl:+.4f}, n={int(post_flags.sum())})')

    return {
        'applied': True,
        'chosen_threshold_raw': chosen_threshold,
        'b_shift': b_shift,
        'b_shift_raw': b_shift_raw,
        'b_shift_clipped': b_shift_clipped,
        'post_selectivity': post_sel,
        'post_precision': post_prec,
        'post_mean_pnl': post_mean_pnl,
        'post_ev': post_ev,
        'temperature': T,
        **search_info,
    }


def evaluate_model(model, scaler, X_test, y_test, n_features):
    X_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    pred_labels = (preds >= 0.5).astype(int)
    tp = ((pred_labels == 1) & (y_test == 1)).sum()
    fp = ((pred_labels == 1) & (y_test == 0)).sum()
    fn = ((pred_labels == 0) & (y_test == 1)).sum()
    tn = ((pred_labels == 0) & (y_test == 0)).sum()

    pred_labels_live = (preds >= LIVE_THRESHOLD).astype(int)
    tp6 = ((pred_labels_live == 1) & (y_test == 1)).sum()
    fp6 = ((pred_labels_live == 1) & (y_test == 0)).sum()
    precision_at_live = float(tp6 / (tp6 + fp6)) if (tp6 + fp6) > 0 else 0.0

    return {
        'accuracy': float((pred_labels == y_test).mean()),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'precision_at_live': precision_at_live,
        'n_preds_live': int(tp6 + fp6),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'positive_rate': float(y_test.mean()),
    }


def save_model(model, scaler, features, metrics, train_metrics, seq_len,
               model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    tmp_model = model_path + '.tmp'
    with h5py.File(tmp_model, 'w') as f:
        f.attrs['model_type'] = 'pytorch_lstm'
        f.attrs['input_size'] = len(features)
        f.attrs['hidden_size'] = train_metrics.get('hidden_size', 128)
        f.attrs['num_layers'] = train_metrics.get('num_layers', 2)
        f.attrs['dropout'] = train_metrics.get('dropout', 0.3)
        f.attrs['seq_len'] = seq_len
        f.attrs['features'] = ','.join(features)
        f.attrs['test_accuracy'] = metrics['accuracy']
        f.attrs['test_precision'] = metrics['precision']
        f.attrs['test_recall'] = metrics['recall']
        f.attrs['use_attention'] = int(train_metrics.get('use_attention', False))
        f.attrs['survival_weight'] = train_metrics.get('survival_weight', 1.0)
        f.attrs['fp_weight'] = train_metrics.get('fp_weight', 1.0)
        f.attrs['sequence_normalize'] = int(train_metrics.get('sequence_normalize', False))
        f.attrs['trained_at'] = datetime.now().isoformat()
        weights_grp = f.create_group('weights')
        for key, tensor in model.state_dict().items():
            weights_grp.create_dataset(key, data=tensor.cpu().numpy())
    os.replace(tmp_model, model_path)

    tmp_scaler = scaler_path + '.tmp'
    with open(tmp_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    os.replace(tmp_scaler, scaler_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=os.path.join(BASE_PATH, 'models', 'lstm'))
    parser.add_argument('--hidden-size', type=int, default=48)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--use-attention', action='store_true')
    parser.add_argument('--survival-weight', type=float, default=1.0)
    parser.add_argument('--fp-weight', type=float, default=1.0)
    args = parser.parse_args()

    print(f'Device: {DEVICE}')

    X, y, dates, early_sl, features, pnl = load_sequences(seq_len=args.seq_len)
    n_features = len(features)
    print(f'Features ({n_features}): {features}')
    print(f'Total samples: {len(X)} | Positive: {y.sum():.0f} ({y.mean():.1%})')

    splits = time_split(X, y, dates, early_sl=early_sl, pnl=pnl)
    print(f'  Train: {len(splits["X_train"])} | Val: {len(splits["X_val"])} | Test: {len(splits["X_test"])}')

    model, scaler, train_metrics = train_model(
        splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'],
        n_features, features,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
        early_sl_train=splits.get('early_sl_train'),
        early_sl_val=splits.get('early_sl_val'),
        survival_weight=args.survival_weight,
        fp_weight=args.fp_weight,
        use_attention=args.use_attention,
        sequence_normalize=False,
        mixup_alpha=0.3,
        swa_window=5,
        pnl_train=splits.get('pnl_train'),
        pnl_val=splits.get('pnl_val'),
        # Pass training dates so the daily-ranking auxiliary loss
        # activates when run standalone (the WF gate already passes
        # these kwargs; this keeps parity between the two entry points).
        dates_train=splits.get('train_dates'),
        daily_rank_enabled=True,
        daily_rank_lambda=0.5,
        daily_rank_min_per_day=6)

    if model is None:
        raise RuntimeError(f'Training failed: {train_metrics}')

    print(f'Positive weight: {train_metrics["pos_weight"]:.1f}x')
    print(f'Selector: {train_metrics.get("selector_used")}')

    metrics = evaluate_model(model, scaler, splits['X_test'], splits['y_test'], n_features)
    print(f'\n=== Test Results ===')
    print(f'  Accuracy:  {metrics["accuracy"]:.2%}')
    print(f'  Precision@0.5: {metrics["precision"]:.2%}')
    print(f'  Precision@0.6: {metrics["precision_at_live"]:.2%} (n={metrics["n_preds_live"]})')
    print(f'  Recall:    {metrics["recall"]:.2%}')

    # When saving into the candidates directory, also write the
    # candidate_model.h5 / candidate_scaler.pkl filenames the ml-improve
    # pipeline expects first, plus trading_model.h5 / scaler.pkl as a
    # fallback for older entry points.
    if os.path.basename(os.path.normpath(args.output_dir)) == 'candidates':
        cand_model_path = os.path.join(args.output_dir, 'candidate_model.h5')
        cand_scaler_path = os.path.join(args.output_dir, 'candidate_scaler.pkl')
        save_model(model, scaler, features, metrics, train_metrics, args.seq_len,
                   cand_model_path, cand_scaler_path)
        print(f'\nCandidate model saved to: {cand_model_path}')
        print(f'Candidate scaler saved to: {cand_scaler_path}')

    model_path = os.path.join(args.output_dir, 'trading_model.h5')
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    save_model(model, scaler, features, metrics, train_metrics, args.seq_len,
               model_path, scaler_path)
    print(f'Model saved to: {model_path}')
    print(f'Scaler saved to: {scaler_path}')


if __name__ == '__main__':
    main()
