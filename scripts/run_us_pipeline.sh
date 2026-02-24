#!/bin/bash
# US Market Daily Pipeline
# Runs at 04:15 ICT (after US market closes)

set -euo pipefail
set -a && source ~/trading-system/config/.env && set +a

if [ "$MARKET_US_ENABLED" != "true" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [US] Market disabled, skipping."
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [US] Starting daily pipeline..."

# 1. Fetch US market data via yfinance
python ~/trading-system/models/features/pipeline.py --market us

# 2. Generate signals
python ~/trading-system/openclaw/skills/set-trader/signal.py --market us

echo "$(date '+%Y-%m-%d %H:%M:%S') [US] Pipeline complete."
