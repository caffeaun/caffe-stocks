#!/bin/bash
# US Market Daily Pipeline
# Runs at 04:15 ICT (after US market closes)

set -euo pipefail
set -a && source ~/projects/caffe-stocks/config/.env && set +a

if [ "$MARKET_US_ENABLED" != "true" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [US] Market disabled, skipping."
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [US] Starting daily pipeline..."

# 1. Fetch US market data via Alpha Vantage (all symbols, writes to us_candles_raw)
/home/kanoonth-ai/shared-venv/bin/python ~/projects/caffe-stocks/scripts/fetch_alpha_vantage.py --all

# 2. Compute technical indicators (reads us_candles_raw, writes us_candles)
/home/kanoonth-ai/shared-venv/bin/python ~/projects/caffe-stocks/scripts/compute_us_indicators.py

# 3. Score and generate signals (reads us_candles, writes us_latest_signal.json)
/home/kanoonth-ai/shared-venv/bin/python ~/projects/caffe-stocks/scripts/score_us_signals.py

echo "$(date '+%Y-%m-%d %H:%M:%S') [US] Pipeline complete."
