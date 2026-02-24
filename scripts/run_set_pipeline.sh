#!/bin/bash
# SET Market Daily Pipeline
# Runs at 17:00 ICT (30 min after SET closes)

set -euo pipefail
set -a && source ~/trading-system/config/.env && set +a

echo "$(date '+%Y-%m-%d %H:%M:%S') [SET] Starting daily pipeline..."

# 1. Scrape closing data from Streaming app
python ~/openclaw/skills/set-trader/scraper.py

# 2. Run feature pipeline
python ~/trading-system/models/features/pipeline.py

# 3. Generate signals
python ~/openclaw/skills/set-trader/signal.py

echo "$(date '+%Y-%m-%d %H:%M:%S') [SET] Pipeline complete."
