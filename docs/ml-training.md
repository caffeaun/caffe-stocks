# Caffe-Stocks: ML Training Whitepaper

Status: **draft v1** — locks the design before we write `return_gate.py` and the next iteration of the LSTM trainer.

## 1. Objective

A self-funding trading account that **returns 50% gross per year on a 10k THB base**, withdraws 5k to SCB at year-end, and resets to 10k for the next loop.

Sub-goals:
- Net annual return ≥ 50% after friction (broker, slippage, any future tax)
- Per-trade win rate ≥ 30% (design buffer — math works at 25%)
- Max drawdown ≤ 25% (10k → 7.5k floor; survivable, recoverable in months)
- ≥ 30 trades per 12-month walk-forward window (statistical significance)
- Robust across regimes (positive return in ≥ 3 of 4 quarters per window)
- All-THB universe (no FX exposure, no foreign-income remittance complications)

**Not goals:**
- Maximizing Sharpe (we can tolerate volatility for the return target)
- Minimizing trade count (more trades, properly sized, compound faster)
- Beating per-trade WR of any benchmark (a 30% WR strategy with 5:1 RR beats a 60% WR 1:1 strategy)
- Trading TFEX / leveraged derivatives in v1

## 2. Universe

Anything **purchasable and sellable in THB on a Thai broker (BLS)**. By instrument type:

| Tier | Examples | Slippage R/T (typical) |
|---|---|---|
| SET50 | PTT, SCB, KBANK, AOT | 0.06–0.10% |
| SET51–full | broader SET names (~650) | 0.10–0.40% |
| mai | small/mid caps (~200) | 0.30–0.80% |
| ETFs | TDEX, GLD, CHINA, JAPAN, BANK, ENGY | 0.05–0.20% |
| Depositary Receipts | TCNT01 (Tencent), AAPL01, etc. | 0.10–0.30% |
| REITs / IFFs | CPNREIT, BTSGIF, ... | 0.10–0.40% |

The model treats each symbol as a row in a unified feature matrix; the *type* of instrument is one input feature (one-hot encoded), not a structural separation.

### Liquidity gate (hard filter)

Per-symbol filter applied before any signal is considered:

- **Stocks**: 20-day average daily turnover ≥ **5,000,000 THB**
- **ETFs / DRs / REITs**: 20-day average daily turnover ≥ **1,000,000 THB**
  (looser — APs and market-makers tighten spreads even on low-volume names)

This excludes ~half of mai stocks and the bottom decile of full-SET names.

## 3. Strategy hypothesis

We exploit **asymmetric expected value** with a high reward-to-risk ratio. The mechanic:

| Parameter | Default |
|---|---|
| Stop-loss | -3% from entry |
| Target | +15% from entry |
| Trailing stop | activates at +8%, trails 5% behind peak |
| RR ratio | 5:1 |

Math at 30% WR, 0.5% all-in friction:

```
EV per trade = 0.30 × (+15%) + 0.70 × (-3%) − 0.5%
             = +4.5% − 2.1% − 0.5%
             = +1.9% net
```

50 trades/year compounded → **+155%**. Cushion is large enough to absorb worse WR, fewer trades, or higher friction.

The role of the model is **filtering** — to lift the rules-only baseline WR (whatever that is — first task of the new gate is to measure it) by enough percentage points to clear the friction cost. **The model is not the strategy; the rules + sizing + exits are the strategy. The model is one filter inside it.**

## 4. Data

### Current

- `~/projects/caffe-stocks/data/candles.db` — Thai SET daily OHLCV, ~70 stocks (current universe — to be expanded)
- `~/projects/caffe-stocks/data/us_candles.db` — US daily OHLCV (currently for cross-market features only)
- `~/projects/caffe-stocks/data/training_data.csv` — engineered features for ML

### Required additions

- **Universe expansion**: ingest full SET (~700) + mai (~200) + ETFs (~25) + DRs (~varies) into `candles.db`. Source: SET API, settrade scrape, or commercial data vendor. Estimated additional storage: ~500MB.
- **Liquidity history**: bid/ask spread + intraday depth for slippage modeling. Daily ADTV is computable from candles, but **intraday spread requires tick or 1-min data** — defer to v1.5 if not available; use ATR-based slippage proxy in v1.
- **Corporate actions**: dividends, splits, mergers. Adjust prices accordingly. SET publishes; ingest via scheduled job.
- **Instrument-type metadata**: stock / ETF / DR / REIT / IFF flag per symbol. Static table.

### Data quality requirements

- Coverage ≥ 80% per symbol over the backtest window (handle delistings, halts)
- No look-ahead bias: only data available at close of day T can be used for signal at day T+1 open
- Survivorship bias acknowledged: backtest uses point-in-time membership of indices

## 5. Features

### Carry forward from existing pipeline (`feature_eng.py`, 21 features)

- Returns: 1d, 5d, 20d
- Volatility: ATR(14), realized vol(20)
- Momentum: RSI(14), MACD signal/histogram
- Volume: 5d/20d volume ratio
- Price relative: SMA(50)/SMA(200) ratio, distance from 52w high/low
- (full list in `feature_eng.py`)

### Add for v1

- **Cross-sectional rank** of each feature within the universe on each date. Captures relative strength.
- **Sector/instrument-type encoding**: one-hot for stock vs ETF vs DR + sector for stocks
- **Liquidity feature**: log(ADTV) — model can learn that more liquid names behave differently
- **Market regime**: SET50 trend (TDEX 50-day return) — the model sees overall market context
- **Spread-derived features**: rolling ratio of (high - low) / close as a proxy for intraday volatility

### Drop or rethink

- Single-stock features that don't generalize across instrument types (anything specific to financials, e.g., P/B ratio, would not work for ETFs and DRs)
- Features with low coverage (`MIN_COVERAGE = 0.65` filter is currently in place — keep)

Final feature count target: 25–35.

## 6. Labels

### Current (to be replaced)

- "Did the next 3-day return cross +X%?" — binary
- This is what the LSTM has been training on; it produces poor WR

### Proposed v1 label

**Realistic-trade label**: simulate the actual entry/exit logic and label "1" if the trade would have been a winner (hit target before stop), "0" if loser, **excluded** if neither hit within max_hold_days.

```
For each (symbol, date) candidate:
  entry_price = open[T+1]
  for d in 1..max_hold_days:
    if low[T+d] <= entry_price * (1 - stop_pct):
      label = 0; break  (lost)
    if high[T+d] >= entry_price * (1 + target_pct):
      label = 1; break  (won)
  else:
    label = -1   (excluded — no decision in window)
```

Defaults: `stop_pct = 0.03`, `target_pct = 0.15`, `max_hold_days = 30`.

### Why this label

- **Matches the strategy**. The model learns to predict trades that hit target before stop — exactly what we want.
- **Naturally excludes ambiguous cases** (the trade hit neither in window — model isn't penalized for these).
- **Robust to lookahead**: only uses data within the simulated trade window.
- **Adjusts cleanly for parameter sweeps**: if we change stop/target, regenerate labels.

### Class imbalance

Expected label distribution:
- ~25–30% wins (those become label=1)
- ~50–60% losses (label=0)
- ~10–20% excluded (label=-1)

Class imbalance is mild; standard cross-entropy with optional class weighting is fine. **Do not use focal loss** — past experiments (run #~80 in `ml-feedback.json`) showed it didn't help and over-biased predictions.

## 7. Model

### v1: gradient boosting over engineered features (XGBoost or LightGBM)

**Why switch from LSTM:**

- Daily bar sequences carry mostly noise; LSTMs over-parameterize what is essentially a tabular problem.
- 335 LSTM training runs hit a ceiling at ~39% WR. The bottleneck is the data + label, not the model.
- Tree-based models train in minutes vs. hours, are interpretable (feature importance), require less data per parameter, and handle missing values natively.
- Industry benchmark for tabular financial ML: XGBoost / LightGBM beats neural nets unless N >> 1M.

**Model spec:**

```python
LightGBMClassifier(
  num_leaves=31,
  max_depth=6,
  learning_rate=0.05,
  n_estimators=300,
  min_child_samples=50,
  subsample=0.8,
  colsample_bytree=0.8,
  reg_alpha=0.1,
  reg_lambda=0.1,
  class_weight={0: 1.0, 1: 1.5},  # mild over-weight on positive class
  early_stopping_rounds=20,
)
```

### v2 (later): keep LSTM only if it beats GBT in head-to-head walk-forward

If post-v1 we want to experiment with sequence models, the test is: train both on identical splits, gate both at the same return threshold, ship whichever is better. Don't keep LSTM as a sunk-cost commitment.

### Output

Probability `p` that the next trade hits target before stop. Bot enters if `p ≥ threshold`. Threshold tuned on validation set per walk-forward window.

## 8. Training procedure

### Walk-forward layout

```
Total history: ~5 years of daily data per symbol
Window: rolling, 12-month out-of-sample test on each step
Step: 3 months (so each year contributes 4 OOS windows)
Training data per step: all data up to start of OOS window
                        (expanding window, not sliding)
Validation: last 3 months before OOS, used for early stopping
            and threshold tuning
```

This produces ~16 OOS windows over 4 years — enough for stable gate evaluation.

### Reproducibility

- Random seed fixed per run, logged to `ml-feedback.json`
- All hyperparameters recorded
- Training data hash recorded (so we know if the upstream data changed mid-experiment)

## 9. Acceptance gate (`return_gate.py`)

A model **passes** the gate if, over the walk-forward windows:

```
For each window:
  Run live-style simulation:
    - Each day, score every symbol in universe
    - Take top-K predictions where p ≥ threshold AND liquidity gate passes
    - Enter at next-day open, exit per stop/target/trailing/max-hold rules
    - Apply per-symbol friction (commission + VAT + slippage)

  Window passes if:
    - Annualized return ≥ 50%
    - Max drawdown ≤ 25%
    - ≥ 5 trades in window
    - Per-quarter return positive in ≥ 3 of 4 sub-quarters
    - Per-trade WR ≥ 30%

Model passes if ≥ 80% of windows pass (e.g., 13 of 16 OOS windows).
```

The 80%-of-windows threshold is intentional — it rewards consistency across regimes, not one big win that masks losing periods.

## 10. Friction model

Per-trade total cost (round-trip) = commission_cost + slippage_cost.

### Commission cost (BLS, online cash balance)

```
commission_pct = max(broker_rate, min_commission / trade_value)
commission_cost = commission_pct × trade_value × (1 + 0.07)  # +VAT
                + 0.005% (trading fee)
                + 0.001% (clearing)
                + 0.001% (regulatory)
```

`broker_rate` to be confirmed from BLS account statement (placeholder: 0.10%–0.157%).

### Slippage cost (per side)

Tier-based default if no spread data:

| Tier | Slippage / side |
|---|---|
| SET50 + ETFs (high-volume) | 0.05% |
| SET51–100 | 0.10% |
| SET101+, DRs (high-AUM) | 0.20% |
| mai, low-volume DRs/ETFs | 0.40% |

When intraday spread data is available (post-v1.5), slippage = `recent_spread × 0.5` (cross half the spread per side).

### Future-proofing for FTT

`config/friction.yaml` includes a `ftt_pct` field, currently 0. When FTT enacts, flip to 0.055% (year 1) or 0.11% (thereafter) on the sell side. Gate re-runs reflect new friction without code changes.

## 11. Position sizing

### v1: single-position, full-capital

- One open position at a time
- Each entry uses 100% of available cash
- Stop-loss is portfolio-level: -3% of capital per trade
- Target is portfolio-level: +15% of capital

This maximizes compounding and matches the math in §3. Trade-off: every entry is a concentration bet.

### v2 (post-paper-trade validation): split sizing

If concentration causes painful DDs, split into 50/50 across two simultaneous positions. Halves the per-trade impact, halves the compounding rate, halves the DD. Only consider once v1 has run for a quarter.

### Hard rule: never margin

The strategy works without leverage. Margin would amplify both wins and losses but also drawdowns and emotional pressure. Defer until the system has proven itself across a full year.

## 12. Exit rules

```
Entry: t_0 at open, trade_id assigned

Loop daily:
  current_price = close

  if current_price <= stop_loss:
    EXIT at next-day open  (slippage applies)
    label = LOSS

  elif trailing_active:
    if current_price < peak_price * (1 - trailing_pct):
      EXIT at next-day open
      label = TRAIL_STOP_HIT

  if current_price >= entry * (1 + activate_trail_pct):
    trailing_active = True
    peak_price = max(peak_price, current_price)

  if current_price >= target:
    EXIT at next-day open
    label = TARGET_HIT

  if days_held >= max_hold_days:
    EXIT at next-day open
    label = TIME_EXIT
```

Defaults:

| Parameter | Value | Rationale |
|---|---|---|
| stop_pct | 3% | Tight enough to limit DD, loose enough to avoid noise stops |
| target_pct | 15% | 5:1 RR |
| activate_trail_pct | 8% | Locks in some gain before market reverses |
| trailing_pct | 5% | Tight enough to capture move, loose enough to ride trend |
| max_hold_days | 30 | If neither stop nor target hit in 6 weeks, the trade thesis is dead |

These defaults are first-pass; backtest may suggest tighter/looser values.

## 13. Risk management

### Daily portfolio kill-switch

If portfolio_value drops below `0.85 × peak_value` (15% drawdown), pause new entries for 5 trading days. Forces a reflection pause before doubling down on a losing run.

### Annual kill-switch

If by month 6 the YTD return is **negative**, pause all new entries until end of year. Don't try to "make it back" — that's how -25% becomes -40%.

### Per-symbol cooldown

After exiting a trade, do not re-enter the same symbol for 5 trading days. Prevents whipsaw on choppy names.

### Manual override always allowed

The Telegram bot's `t skip`, `t notfilled`, etc. commands let the operator bypass the model entirely. Discretion is a feature, not a bug.

## 14. Phased rollout

| Phase | Duration | Exit criteria |
|---|---|---|
| **A. Backtest** | 1–2 weeks | Gate passes ≥ 80% of walk-forward windows |
| **B. Paper trade** | 4 weeks (~20 trading days) | YTD paper return on track for 50% annualized; no surprises in execution |
| **C. Real money — half size** | 4 weeks at 5k THB | YTD return positive; no execution issues vs. paper |
| **D. Real money — full size** | 12 months at 10k THB | End of year: assess, withdraw 5k to SCB if ≥ 15k, reset for next cycle |

Each phase is a checkpoint; failing any returns to the previous phase, not back to phase A unless the failure is fundamental.

## 15. Open questions

These are flagged for resolution before / during implementation:

1. **Exact BLS commission rate?** Pending account statement.
2. **Universe data source.** Settrade API has rate limits; commercial vendors cost money. Decide before ingest job is scheduled.
3. **Cross-sectional ranking — at what cadence?** Daily? Weekly? More frequent rebalance feels right but costs compute.
4. **Model retraining cadence.** Weekly retrain? Monthly? Re-evaluate after first month of paper trading.
5. **Should the model also output expected hold time?** Could improve sizing if we know "this is a 3-day flip" vs "this is a 20-day move".
6. **What to do with `data/ml-feedback.json` (335 historical runs)?** Could mine it for patterns of what didn't work, to avoid re-deriving lessons. Worth a one-time analysis pass.

## 16. Appendix — math derivations

### A1. Expected value of one trade

```
EV = WR × (target_pct − slippage − commission)
   + (1 − WR) × (−stop_pct − slippage − commission)

At WR=0.30, target=0.15, stop=0.03, friction=0.005:
EV = 0.30 × (0.15 − 0.005) − 0.70 × (0.03 + 0.005)
   = 0.30 × 0.145 − 0.70 × 0.035
   = 0.0435 − 0.0245
   = +0.019 → +1.9% per trade
```

### A2. Compounded annual return

```
CAR = (1 + EV)^N − 1     where N = number of trades per year

At EV=0.019, N=50:
CAR = (1.019)^50 − 1 = 2.55 − 1 = +155%

At N=30:
CAR = (1.019)^30 − 1 = 1.76 − 1 = +76%

At N=20:
CAR = (1.019)^20 − 1 = 1.46 − 1 = +46%  ← below goal
```

So **trade frequency floor is ~25/year** to clear 50% comfortably at these defaults.

### A3. Drawdown bound from consecutive losses

```
DD after k losing trades in a row = 1 − (1 − stop_pct)^k

stop_pct=0.03:
  k=5  → 14.1%   (recoverable)
  k=8  → 21.7%   (approaching kill-switch at 25%)
  k=10 → 26.3%   (kill-switch trips)

P(10 consecutive losses | WR=0.30) = (1 − 0.30)^10 ≈ 2.8%
```

So a 10-loss streak is improbable but not negligible. The 15% pause kill-switch (after ~5 consecutive) is the real backstop.

### A4. Why WR cushion matters

If the live WR drifts to 25% (down from 30% target) due to regime change:
```
EV = 0.25 × 0.145 − 0.75 × 0.035 = 0.0363 − 0.0263 = +0.010 → +1.0%
At N=50: (1.01)^50 = +64%
```

Still hits goal. Below 22% WR, math breaks down. The 30% target gives us ~7 percentage points of WR cushion — enough to absorb regime drift without dropping below goal.

---

**End of v1 draft.** Open for review and iteration before implementing `return_gate.py`.
