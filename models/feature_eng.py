"""Centralized feature engineering for LSTM pipeline.

Single source of truth for deriving features from candles data.
Used by: lstm_trainer.py, backtest_lstm.py, model_gate.py, lstm_trader.py
"""
import os

import numpy as np
import pandas as pd


# Base features from compute_indicators.py (always present in candles table)
BASE_FEATURES = ['rsi', 'macd', 'atr', 'volume_ratio']

# Derived features computed per-symbol from price/indicator data
DERIVED_FEATURES = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
    'bb_position', 'atr_pct', 'vol_momentum', 'rsi_momentum',
    # Short-term risk features (predict early stop-loss)
    'close_vs_low_3d', 'close_position_in_range', 'atr_sl_ratio',
    'down_gap_count', 'intraday_range_3d',
    # Momentum-persistence + volatility-regime (attempt 260)
    'up_days_5d', 'atr_ratio_20_60',
]

# SET Index regime features (from compute_indicators.py, 99% coverage)
SET_FEATURES = ['set_ret_5d', 'set_ret_10d', 'set_above_sma20', 'set_rsi', 'set_volatility']

# Sector-relative features (from our own stock data, 100% coverage)
SECTOR_FEATURES = [
    'sector_avg_ret_1d', 'stock_vs_sector_ret',
    'sector_breadth', 'sector_avg_volume_ratio',
]

# Market breadth features (daily across all stocks, 100% coverage)
BREADTH_FEATURES = [
    'market_breadth_adv', 'market_breadth_above_sma20', 'market_new_highs',
]

# Supplemental features (sparse — only use when coverage > 80%)
SUPPLEMENTAL_FEATURES = [
    'first_hour_range', 'first_hour_vol_ratio',
    'sector_change_pct',
    'foreign_net', 'foreign_buy_pct', 'foreign_sell_pct',
    'pe', 'pbv', 'dividend_yield', 'market_percent_change',
    'foreign_net_monthly', 'foreign_buy_monthly', 'foreign_sell_monthly',
]

# Minimum coverage to include a feature in training
# 65% allows intraday features (68.8% coverage — 99.6% in recent val/test windows)
MIN_COVERAGE = 0.65

# Percentile rank parameters
PCTRANK_WINDOW = 60
PCTRANK_MIN_PERIODS = 30

# Cross-sectional rank features: rank each stock vs all peers on the same day
# These are regime-invariant by construction (always [0,1] regardless of market level)
XRANK_FEATURES = [
    'rsi', 'volume_ratio', 'atr_pct', 'ret_1d', 'ret_5d', 'ret_20d',
    'macd', 'bb_position', 'rsi_momentum', 'vol_momentum',
    'stock_vs_sector_ret',
    'foreign_net_monthly',
]


def compute_derived_features(df):
    """Compute per-symbol derived features. Returns DataFrame with new columns.
    Skips features already present in the DataFrame (pre-computed in candles table).
    Input must have: close, bb_upper, bb_lower, atr, volume_ratio, rsi, symbol, timestamp/date.
    """
    # Check if derived features are already in the data (pre-computed by compute_indicators.py)
    needed = [f for f in DERIVED_FEATURES if f not in df.columns]
    need_20d_high = '20_day_high' not in df.columns

    if not needed and not need_20d_high:
        return df

    parts = []
    sort_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    for symbol in df['symbol'].unique():
        sdf = df[df['symbol'] == symbol].copy().sort_values(sort_col)

        if 'ret_1d' not in df.columns:
            sdf['ret_1d'] = sdf['close'].pct_change(1)
        if 'ret_5d' not in df.columns:
            sdf['ret_5d'] = sdf['close'].pct_change(5)
        if 'ret_10d' not in df.columns:
            sdf['ret_10d'] = sdf['close'].pct_change(10)
        if 'ret_20d' not in df.columns:
            sdf['ret_20d'] = sdf['close'].pct_change(20)

        if 'bb_position' not in df.columns:
            bb_width = sdf['bb_upper'] - sdf['bb_lower']
            sdf['bb_position'] = np.where(bb_width > 0,
                                           (sdf['close'] - sdf['bb_lower']) / bb_width, 0.5)

        if 'atr_pct' not in df.columns:
            sdf['atr_pct'] = sdf['atr'] / sdf['close']
        if 'vol_momentum' not in df.columns:
            sdf['vol_momentum'] = sdf['volume_ratio'] / sdf['volume_ratio'].rolling(5).mean()
        if 'rsi_momentum' not in df.columns:
            sdf['rsi_momentum'] = sdf['rsi'] - sdf['rsi'].shift(3)

        # Short-term risk features
        if 'close_vs_low_3d' not in df.columns:
            low_3d = sdf['low'].rolling(3).min()
            sdf['close_vs_low_3d'] = (sdf['close'] - low_3d) / sdf['close']

        if 'close_position_in_range' not in df.columns:
            daily_range = sdf['high'] - sdf['low']
            daily_pos = np.where(daily_range > 0,
                                 (sdf['close'] - sdf['low']) / daily_range, 0.5)
            sdf['close_position_in_range'] = pd.Series(daily_pos, index=sdf.index).rolling(3).mean()

        if 'atr_sl_ratio' not in df.columns:
            sdf['atr_sl_ratio'] = sdf['atr_pct'] / 0.03 if 'atr_pct' in sdf.columns else (sdf['atr'] / sdf['close']) / 0.03

        if 'down_gap_count' not in df.columns:
            down_gap = (sdf['open'] < sdf['close'].shift(1) * 0.99).astype(int)
            sdf['down_gap_count'] = down_gap.rolling(10, min_periods=1).sum()

        if 'intraday_range_3d' not in df.columns:
            sdf['intraday_range_3d'] = ((sdf['high'] - sdf['low']) / sdf['close']).rolling(3).mean()

        # Momentum persistence: fraction of up-days in the last 5 sessions.
        # Bounded [0,1] by construction → regime-invariant across splits.
        # Discriminates sustained trends (0.8+) from choppy action (~0.4-0.6).
        if 'up_days_5d' not in df.columns:
            up = (sdf['close'].pct_change(1) > 0).astype(float)
            sdf['up_days_5d'] = up.rolling(5, min_periods=5).mean()

        # Volatility regime: 20d avg ATR / 60d avg ATR. >1 = expanding vol
        # (often post-breakout or regime shift), <1 = contracting vol
        # (mean-reversion territory). Ratio is scale-free → regime-invariant.
        if 'atr_ratio_20_60' not in df.columns:
            atr_20 = sdf['atr'].rolling(20, min_periods=20).mean()
            atr_60 = sdf['atr'].rolling(60, min_periods=60).mean()
            sdf['atr_ratio_20_60'] = (atr_20 / atr_60).replace([np.inf, -np.inf], np.nan)

        # For backtester: 20-day high needed by rule filter
        if need_20d_high:
            sdf['20_day_high'] = sdf['high'].rolling(20).max()

        parts.append(sdf)

    return pd.concat(parts, ignore_index=True)


def compute_percentile_ranks(df, window=PCTRANK_WINDOW, min_periods=PCTRANK_MIN_PERIODS):
    """Add rolling percentile rank columns for all numeric features.

    Per-symbol rolling percentile rank maps values to [0,1] relative to recent
    history, making features time-invariant across market regimes.
    Called from prepare_data() so columns are available for any model that needs them.
    """
    sort_col = 'timestamp' if 'timestamp' in df.columns else 'date'

    # All features that should get pctrank versions (skip binary)
    skip = {'set_above_sma20'}
    candidates = list(BASE_FEATURES) + list(DERIVED_FEATURES) + list(SET_FEATURES) + list(SECTOR_FEATURES) + list(BREADTH_FEATURES) + list(SUPPLEMENTAL_FEATURES)
    source_features = [f for f in candidates if f in df.columns and f not in skip]

    # Skip if already computed
    if all(f'{f}_pctrank' in df.columns for f in source_features):
        return df

    parts = []
    for symbol in df['symbol'].unique():
        sdf = df[df['symbol'] == symbol].copy().sort_values(sort_col)
        for feat in source_features:
            col_name = f'{feat}_pctrank'
            if col_name not in sdf.columns:
                sdf[col_name] = sdf[feat].rolling(
                    window, min_periods=min_periods
                ).apply(lambda x: np.mean(x <= x[-1]), raw=True)
        parts.append(sdf)

    return pd.concat(parts, ignore_index=True)


def compute_cross_sectional_ranks(df):
    """Add cross-sectional percentile rank features.

    For each trading date, rank each feature across all symbols → [0, 1].
    Unlike rolling pctrank (compares to own history), xrank compares to ALL
    peers on the same day. This is regime-invariant: a stock at the 90th
    percentile of RSI stays at the 90th percentile regardless of market level.
    """
    sort_col = 'timestamp' if 'timestamp' in df.columns else 'date'

    # Skip if already computed
    if all(f'{f}_xrank' in df.columns for f in XRANK_FEATURES if f in df.columns):
        return df

    # Create date key for grouping (date only, no time)
    df['_xrank_date'] = pd.to_datetime(df[sort_col]).dt.date

    for feat in XRANK_FEATURES:
        col_name = f'{feat}_xrank'
        if col_name not in df.columns and feat in df.columns:
            df[col_name] = df.groupby('_xrank_date')[feat].rank(pct=True)

    df = df.drop(columns=['_xrank_date'])
    return df


def compute_cross_features(df):
    """Add interaction (cross) features built from already-bounded regime-invariant inputs.

    Product of two [0,1] features is also [0,1] — so these are regime-invariant
    by construction AND easier to interpret than LSTM-discovered interactions.
    The LSTM has 20-bar windows and limited ability to find multiplicative
    interactions across features, so explicit pre-computation helps.
    """
    # Momentum × Volume concordance: high when BOTH relative momentum (rsi_xrank)
    # AND relative volume (volume_ratio_xrank) are elevated simultaneously.
    # This is the classic "confirmed breakout" signal — neither alone is enough.
    if 'momentum_volume_cross' not in df.columns:
        if 'rsi_xrank' in df.columns and 'volume_ratio_xrank' in df.columns:
            df['momentum_volume_cross'] = df['rsi_xrank'] * df['volume_ratio_xrank']

    # Market × Sector alignment: high when both the overall market and the stock's
    # sector are in positive breadth. Captures regime concordance — stocks rarely
    # run when both their sector and the market are weak.
    if 'market_sector_aligned' not in df.columns:
        if 'market_breadth_adv' in df.columns and 'sector_breadth' in df.columns:
            df['market_sector_aligned'] = df['market_breadth_adv'] * df['sector_breadth']

    return df


def compute_set_regime_zscore(df, window=60, min_periods=30):
    """Add ``set_ret_5d_zscore_60d`` — a market-wide regime feature.

    Iter #714 lesson: the day-gate XGB needs a sharper bear-regime signal to
    avoid over-filtering bull regimes (W7) while properly demoting bear ones
    (W5). ``set_ret_5d`` is constant across all symbols on a given date, so its
    rolling 60-day z-score is computed on the unique daily series and broadcast
    back to every (date, symbol) row.

    Z-score is the right shape because it's regime-invariant by construction:
    a -3% 5-day SET drop has different meaning in low-vol vs high-vol periods,
    but its rank vs the trailing 60d distribution is comparable across splits.
    The day-gate (last-timestep mean across symbols) sees this value exactly;
    the GRU sees its full 20-day trajectory.
    """
    if 'set_ret_5d_zscore_60d' in df.columns:
        return df
    if 'set_ret_5d' not in df.columns:
        return df

    sort_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    # Unique date series for set_ret_5d (market-wide → same value all symbols).
    daily = (df[[sort_col, 'set_ret_5d']]
             .dropna()
             .groupby(sort_col, as_index=False)['set_ret_5d'].first()
             .sort_values(sort_col)
             .reset_index(drop=True))
    rolling_mean = daily['set_ret_5d'].rolling(window, min_periods=min_periods).mean()
    rolling_std = daily['set_ret_5d'].rolling(window, min_periods=min_periods).std()
    safe_std = rolling_std.where(rolling_std > 1e-9, 1.0)
    daily['set_ret_5d_zscore_60d'] = ((daily['set_ret_5d'] - rolling_mean)
                                       / safe_std).clip(-5.0, 5.0)

    df = df.merge(daily[[sort_col, 'set_ret_5d_zscore_60d']],
                  on=sort_col, how='left')
    return df


# Curated feature set: hybrid absolute + xrank + regime context (attempt 136).
#
# Key insight from attempts 131-135:
# - Pure xranks (attempt 135) → 0 trades (no magnitude signal for +15% target)
# - 4 absolute features (attempts 132-134) → 48-50% WR but ~24% std (scaler drift)
# - Regime context features (attempt 131) → lowest std (19.9%)
#
# Strategy: keep 2 absolute features (atr_pct, volume_ratio) for magnitude signal,
# replace ret_1d/ret_5d (highest drift) with regime-context features
# (foreign_net_monthly, market_breadth_above_sma20) that tell the model WHEN
# market conditions favor +15% moves. ret_1d_xrank/ret_5d_xrank still capture
# relative return signal without absolute drift.
# DO NOT MODIFY — Claude ML improvement must NOT change this list.
# Feature changes require manual verification of coverage (>65% in candles.db)
# and retraining of the production scaler. Changing features here without
# updating the scaler causes silent shape mismatch → garbage predictions.
CURATED_FEATURES = [
    # Absolute magnitude features (2)
    'atr_pct',                # volatility magnitude (ATR/close)
    'volume_ratio',           # volume vs 20d avg
    # Bounded ratios (3)
    'atr_sl_ratio',           # ATR vs 3% SL distance
    'intraday_range_3d',      # avg((high-low)/close)
    'bb_position',            # Bollinger Band position [0,1]
    # Cross-sectional ranks (5) — all [0,1] bounded, regime-invariant
    'rsi_xrank',              # relative momentum vs peers
    'volume_ratio_xrank',     # relative volume vs peers
    'ret_1d_xrank',           # relative 1d move vs peers
    'ret_5d_xrank',           # relative 5d trend vs peers
    'macd_xrank',             # relative MACD rank vs peers — independent momentum signal
                              # (MACD captures medium-term trend acceleration, complements
                              # RSI's mean-reversion tendency and ret_5d's raw trend)
    # Relative context (2)
    'stock_vs_sector_ret',    # stock return minus sector avg
    'close_vs_low_3d',        # (close - 3d low) / close
    # Sector + market context (5)
    'sector_breadth',         # fraction of sector stocks positive [0,1]
    # NOTE: foreign_net_monthly is a MARKET-WIDE aggregate (same value for all 70
    # stocks on a given day), so its cross-sectional xrank collapses to a constant
    # ~0.5 and contributes zero signal. Use rolling time-based pctrank instead:
    # "how unusual is today's foreign flow vs the last 60 days?" — a real regime
    # signal, bounded [0,1], uniform across all walk-forward splits.
    'foreign_net_monthly_pctrank',  # rolling 60d rank of market-wide foreign flow [0,1]
    'sector_avg_ret_1d',      # sector average daily return
    'market_breadth_adv',     # fraction of stocks advancing [0,1]
    'market_breadth_above_sma20', # fraction of stocks above SMA20 [0,1]
    # Discrimination + regime features (3) — attempt 263
    # All three are regime-invariant by construction (bounded [0,1] or
    # scale-free ratios) so they carry the same meaning in bull/bear/sideways
    # regimes, which is what walk-forward generalization needs.
    'up_days_5d',             # fraction of up days in last 5 sessions [0,1]
    'atr_ratio_20_60',        # 20d/60d ATR ratio (vol regime, scale-free)
    'market_new_highs',       # fraction of stocks at new 20d highs [0,1]
    # Sector-wide volume regime (1) — complements sector_breadth
    # (fraction positive) and individual volume_ratio by capturing
    # sector-wide unusual volume, a leading signal for regime shifts.
    # Sector-normalized ratio → regime-invariant across bull/bear periods.
    'sector_avg_volume_ratio',
    # Interaction (cross) features (2) — attempt 268: Priority 2 combo
    # Explicitly encoded feature interactions that a 2-layer LSTM with
    # 20-bar windows struggles to auto-discover. Both are products of
    # already regime-invariant [0,1] features so they stay [0,1] bounded
    # across all market regimes — exactly what walk-forward needs.
    'momentum_volume_cross',   # rsi_xrank * volume_ratio_xrank (confirmed-breakout signal)
    'market_sector_aligned',   # market_breadth_adv * sector_breadth (regime concordance)
    # SET-index regime z-score (1) — iter #715
    # Same value across all symbols on a given date, so the day-gate XGB
    # (per-date last-timestep mean) sees it exactly; lets the gate sharpen
    # bear-regime detection (W5 -13.2% set_ret/31% breadth) without
    # over-filtering bull regimes (W7 +16.3% set_ret/53% breadth) the way
    # raw set_ret_5d would under per-window RobustScaler drift. Z-score
    # over a trailing 60-day SET-index window is regime-invariant by
    # construction → comparable meaning across all 7 walk-forward splits.
    'set_ret_5d_zscore_60d',
    # Per-stock raw daily return (1) — iter #1759
    # Added so foundation-model trainers (torch_sundial / torch_timesfm)
    # have a proper continuous per-stock time-series channel to consume.
    # Iter #1757/#1758 swept channel_index over {0=atr_pct, 1=volume_ratio,
    # 7=ret_1d_xrank (cross-sectional rank), 8=ret_5d_xrank, 23=set zscore}
    # and all collapsed: ranks are uniform-distributed [0,1] (not a TS),
    # and SET zscore is identical across all symbols on a given date so
    # Sundial's sampled future-path stats become a per-DAY constant rather
    # than a per-ROW signal. ret_1d (pct_change(1)) is the natural input
    # for Sundial's revin → flow-matching backbone, near-Gaussian and
    # zero-centered after the trainer's manual instance normalization.
    'ret_1d',
]


# Global macro features (Phase 2, 2026-05-27). Populated by
# scripts/fetch_macro.py into data/ml-feedback.db macro_daily. NOT added
# to CURATED_FEATURES (intentionally) — that would force a re-promotion
# of the production panel since `models/panel/rank{1,2,3}/scaler.pkl`
# was fit on the 25-feature CURATED. Macro is opt-in per iter via
# `prepare_data(df, features=CURATED_FEATURES + MACRO_FEATURES)`;
# AMP-driven HP ablations can add macro one feature at a time without
# breaking the panel.
#
# Per symbol we expose 3 derived features:
#   {col}_z60         60-day z-score (regime-invariant)
#   {col}_ret_5d      5-day return
#   {col}_above_sma20 binary (close vs 20-day SMA)
#
# Symbols cover: VIX (risk-off), USD/THB (currency coupling), gold, oil,
# US 10Y yield, Singapore STI (ASEAN regional regime).
MACRO_SYMBOLS = ['vix', 'usd_thb', 'gold', 'oil', 'us_10y', 'sti']
MACRO_FEATURES = [
    f'{sym}_{suffix}'
    for sym in MACRO_SYMBOLS
    for suffix in ('z60', 'ret_5d', 'above_sma20')
]


def attach_macro_features(df, db_path: str | None = None):
    """Join macro_daily (date-indexed) onto df by `timestamp`, computing
    z60 / ret_5d / above_sma20 derivatives per symbol.

    Uses `.shift(1)` on each macro series so date D's macro value is the
    macro CLOSE on D-1 (NYC EOD, before next-morning BKK open) — no
    look-ahead. Rows where macro_daily has no entry for D-1 get NaN.

    Mutates df in place AND returns it for chaining. Caller must add
    MACRO_FEATURES to its feature list explicitly — this helper does
    not modify CURATED_FEATURES.
    """
    import sqlite3
    import pandas as pd

    if db_path is None:
        db_path = os.path.expanduser('~/projects/caffe-stocks/data/ml-feedback.db')
    try:
        with sqlite3.connect(db_path) as conn:
            macro = pd.read_sql_query(
                'SELECT * FROM macro_daily ORDER BY date', conn,
                parse_dates=['date'],
            )
    except Exception:
        # Table missing or DB unreadable — skip macro entirely; caller's
        # MIN_COVERAGE filter in get_available_features will drop the
        # macro feature names so existing trainers see no change.
        return df

    if macro.empty:
        return df

    macro['date'] = macro['date'].dt.strftime('%Y-%m-%d')
    macro = macro.set_index('date').sort_index()

    for sym in MACRO_SYMBOLS:
        if sym not in macro.columns:
            continue
        s = macro[sym].astype(float)
        # Forward-fill NaNs first (US holidays don't align with Thai
        # market days — Memorial Day, Thanksgiving, etc. leave gaps).
        # Then shift by 1 to enforce no-look-ahead: date D's macro is
        # the LAST AVAILABLE close strictly before D.
        s = s.ffill().shift(1)
        sma20 = s.rolling(20, min_periods=10).mean()
        std60 = s.rolling(60, min_periods=20).std()
        mean60 = s.rolling(60, min_periods=20).mean()
        z60 = (s - mean60) / std60.replace(0.0, np.nan)
        ret5 = s.pct_change(5)
        above = (s > sma20).astype(float).where(s.notna() & sma20.notna())

        # df.timestamp is the stock date; map via dict for speed
        z60_map = z60.to_dict()
        ret5_map = ret5.to_dict()
        above_map = above.to_dict()
        ts = df['timestamp'].astype(str)
        df[f'{sym}_z60'] = ts.map(z60_map).astype(float)
        df[f'{sym}_ret_5d'] = ts.map(ret5_map).astype(float)
        df[f'{sym}_above_sma20'] = ts.map(above_map).astype(float)

    return df


def get_available_features(df):
    """Return curated features with sufficient coverage.

    Uses CURATED_FEATURES (25 non-redundant features) instead of auto-detecting
    all 65+ features. This reduces noise from correlated features (e.g., rsi +
    rsi_pctrank + rsi_xrank) while covering all signal types needed for
    walk-forward generalization.
    """
    import sys

    features = [f for f in CURATED_FEATURES
                if f in df.columns and df[f].notna().mean() >= MIN_COVERAGE]

    # Warn about low-quality features
    for f in features:
        if f in df.columns:
            zero_rate = (df[f] == 0).mean()
            if zero_rate > 0.5:
                print(f"WARNING: Feature '{f}' has {zero_rate:.0%} zeros — likely noise", file=sys.stderr)

    return features


def prepare_data(df, features=None):
    """Full pipeline: compute derived features, determine available features, drop NaN.
    If features is provided (e.g., from a saved model), use those.
    Otherwise auto-detect available features.
    Returns (df, features_list).
    """
    df = compute_derived_features(df)
    df = compute_percentile_ranks(df)
    df = compute_cross_sectional_ranks(df)
    df = compute_cross_features(df)
    df = compute_set_regime_zscore(df)

    if features is None:
        features = get_available_features(df)

    # Only drop NaN on the features we'll use
    df = df.dropna(subset=[f for f in features if f in df.columns])

    return df, features
