"""Shared labeling function for LSTM pipeline.

Single source of truth — mirrors backtest_lstm.py simulate_trade() exit logic.
Used by: lstm_trainer.py, model_gate.py
"""
import numpy as np


# Default parameters — must match backtest_lstm.py constants
STOP_PCT = 0.03
TARGET_PCT = 0.15
TRAILING_TRIGGER = 0.07
TRAILING_FLOOR = 0.50
MAX_HOLD = 10


def check_early_stop(lows, entry_price, stop_pct=STOP_PCT, days=3):
    """Check if trade hits stop loss within the first N days.

    Returns True if min(lows[0:days]) breaches stop loss level.
    Used to generate sample weights for survival-weighted training.
    """
    n = min(len(lows), days)
    if n == 0:
        return False
    stop_loss = entry_price * (1 - stop_pct)
    return bool(np.min(lows[:n]) <= stop_loss)


COMMISSION_PCT = 0.0055  # 0.55% round trip — actual BLS Cash Balance rate.
# Source: bualuang.co.th — Cash Balance via i-Trading commission is 0.157% per side
# (entry tier, +7% VAT, no minimum for e-confirmation). SET fees 0.005% trading +
# 0.001% clearing + 0.001% regulatory = 0.007% per side. Slippage 0.05-0.10%
# per side on SET50/100. Round trip:
#   commission+VAT: 0.157% × 2 × 1.07 = 0.336%
#   fees:           0.007% × 2         = 0.014%
#   slippage:       0.05-0.10% × 2     = 0.10-0.20%
#   total:                              = 0.45-0.55%
# Use mid-estimate 0.55%. Was 0.011 (1.1%, too conservative); briefly 0.004 (0.4%,
# too optimistic). Updated 2026-05-04.

# Minimum net profit for positive label (after commission).
# Reverted 0.06 → 0.04: the 0.06 cutoff (attempt #271) collapsed the pipeline.
# Every subsequent attempt (#272–#276) produced 0 valid splits across 7 splits,
# because positives drop from ~22% to ~10% and the BCE+pos_weight trainer can
# no longer push enough samples past the 0.6 inference threshold. Entry #197
# hit 37.5% WR on 7/7 splits at 0.04 — that is the highest-confidence baseline
# we have, and it is the label regime the rest of the loss/arch stack was
# tuned against. Restoring 0.04 unblocks the whole loop so structural changes
# on top of a working baseline can actually be measured.
MIN_PROFIT_PCT = 0.04

# "Clean win" constraint: positive label requires BOTH (pnl > MIN_PROFIT_PCT)
# AND (max adverse excursion <= CLEAN_WIN_MAX_DD). Rationale: every prior
# attempt learned a degraded "random eventual winners" signal that includes
# a large class of trades that spent days underwater before recovering.
# Those recoveries are regime-dependent — they require bull-market dip-buying
# that is absent in chop/bear regimes, and they mostly share the WR-degrading
# pathology: during a ~-2% excursion the signal-filter threshold logic still
# holds, but these samples need regime tailwind to come back. By contrast,
# trades that rise cleanly from entry (MAE < 2%) reflect immediate institutional
# demand — a pattern that projects far more consistently across regimes
# because the features describing early strength (volume_ratio, intraday_range,
# stock_vs_sector_ret, etc.) are direct observations, not downstream effects
# of bullish sentiment. This is distinct from raising MIN_PROFIT_PCT (#271's
# failed approach): we keep the same profit bar, but require the path to it
# to be structurally healthy. Expected positive rate: ~12% (vs ~22%), with
# pos_weight at 7.3x still comfortably under the 10x cap the trainer enforces.
CLEAN_WIN_MAX_DD = 0.02


def label_trade_with_pnl(highs, lows, closes, entry_price,
                         stop_pct=STOP_PCT, target_pct=TARGET_PCT,
                         trailing_trigger=TRAILING_TRIGGER, trailing_floor=TRAILING_FLOOR,
                         max_hold=MAX_HOLD, commission_pct=COMMISSION_PCT):
    """Return (label, pnl_pct) — label is 0/1, pnl_pct is net P&L after commission.

    Exit priority: SL(-3%) → Target(+15%) → Trailing(+7% trigger, 50% floor) → Day 10 hard exit.

    "Clean-win" constraint is enforced via the returned pnl_pct, not the label
    channel. Both lstm_trainer.py and lstm_trainer_candidate.py bypass the
    label return and recompute y = (pnl > MIN_PROFIT_PCT); therefore the only
    way to shift the classification target without modifying the trainers is
    to map *rocky wins* (pnl above the threshold but with a deep intraday
    drawdown) onto a pnl just below MIN_PROFIT_PCT. The pnl stays strictly
    positive so the walk-forward gate's win_rate = (pnls > 0) still counts
    the trade as a win — only EV/avg_win are biased slightly downward
    (conservative), never upward. Real stop-loss exits and marginal wins
    (already below the threshold) are left untouched.
    """
    stop_loss = entry_price * (1 - stop_pct)
    target = entry_price * (1 + target_pct)
    peak_gain = 0.0
    # Track worst intraday excursion (most negative gain vs entry). Used to
    # distinguish "clean" wins (never went meaningfully underwater) from
    # "rocky" wins (recovered from a > CLEAN_WIN_MAX_DD drawdown).
    worst_gain = 0.0

    # Cap value for rocky wins: just below MIN_PROFIT_PCT so y=0 after
    # load_sequences' inline threshold, but comfortably > 0 so the gate's
    # WR still credits the trade. Using MIN_PROFIT_PCT / 2 (0.02) keeps
    # the cap well-separated from the 0 boundary and from MIN_PROFIT_PCT.
    ROCKY_PNL_CAP = MIN_PROFIT_PCT / 2.0

    def _apply_clean_constraint(pnl):
        """If pnl would pass the win threshold but worst_gain breached the
        clean bar, cap the pnl just below the threshold so the inline label
        becomes 0 while keeping the trade accounted as a win in the gate."""
        if pnl > MIN_PROFIT_PCT and worst_gain <= -CLEAN_WIN_MAX_DD:
            return ROCKY_PNL_CAP
        return pnl

    n = min(len(highs), max_hold)
    for day in range(n):
        # Update worst adverse excursion from today's low before any exit check.
        day_low_gain = (lows[day] - entry_price) / entry_price
        if day_low_gain < worst_gain:
            worst_gain = day_low_gain

        # 1. Stop-loss (intraday low) — always label 0, pnl unchanged.
        if lows[day] <= stop_loss:
            gross = -stop_pct
            return 0, gross - commission_pct

        # 2. Target (intraday high) — apply clean-win constraint.
        if highs[day] >= target:
            gross = target_pct
            pnl = gross - commission_pct
            pnl = _apply_clean_constraint(pnl)
            label = 1 if pnl > MIN_PROFIT_PCT else 0
            return label, pnl

        # 3. Trailing stop logic
        current_gain = (closes[day] - entry_price) / entry_price
        if current_gain >= trailing_trigger:
            peak_gain = max(peak_gain, current_gain)

        if peak_gain >= trailing_trigger:
            trail_floor_level = peak_gain * trailing_floor
            if current_gain < trail_floor_level:
                pnl = current_gain - commission_pct
                pnl = _apply_clean_constraint(pnl)
                label = 1 if pnl > MIN_PROFIT_PCT else 0
                return label, pnl

    # 4. Day 10 hard exit
    if n > 0:
        final_gain = (closes[n - 1] - entry_price) / entry_price
        pnl = final_gain - commission_pct
        pnl = _apply_clean_constraint(pnl)
        label = 1 if pnl > MIN_PROFIT_PCT else 0
        return label, pnl

    return 0, -commission_pct


def label_trade(highs, lows, closes, entry_price,
                stop_pct=STOP_PCT, target_pct=TARGET_PCT,
                trailing_trigger=TRAILING_TRIGGER, trailing_floor=TRAILING_FLOOR,
                max_hold=MAX_HOLD):
    """Return 1 if trade would be profitable AFTER commission, 0 otherwise.

    Commission-aware: a trade that gains 0.5% gross but loses 0.6% after
    1.1% round-trip commission is labeled 0 (loss), not 1. This eliminates
    marginal false positives near the decision boundary.

    Args:
        highs: array of high prices for the lookahead window
        lows: array of low prices for the lookahead window
        closes: array of close prices for the lookahead window
        entry_price: entry price (close of last sequence bar)
        stop_pct: stop loss percentage (default 3%)
        target_pct: target percentage (default 15%)
        trailing_trigger: gain % to activate trailing stop (default 7%)
        trailing_floor: fraction of peak gain to keep (default 50%)
        max_hold: maximum holding days (default 10)

    Returns:
        1 if trade exits profitably after commission, 0 otherwise
    """
    label, pnl = label_trade_with_pnl(highs, lows, closes, entry_price,
                                       stop_pct, target_pct,
                                       trailing_trigger, trailing_floor,
                                       max_hold, COMMISSION_PCT)
    return 1 if pnl > MIN_PROFIT_PCT else 0
