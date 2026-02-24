#!/bin/bash
# Daily backup of critical files
# Runs at midnight ICT

set -euo pipefail

BACKUP_DIR=~/trading-system/backups/$(date '+%Y-%m-%d')
mkdir -p "$BACKUP_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') [BACKUP] Starting daily backup..."

# Backup MEMORY.md
cp ~/trading-system/openclaw/MEMORY.md "$BACKUP_DIR/MEMORY.md"

# Backup trade database
if [ -f ~/trading-system/data/trades.db ]; then
    cp ~/trading-system/data/trades.db "$BACKUP_DIR/trades.db"
fi

# Backup candles database
if [ -f ~/trading-system/data/candles.db ]; then
    cp ~/trading-system/data/candles.db "$BACKUP_DIR/candles.db"
fi

# Backup marketing metrics
if [ -f ~/marketing/data/metrics.db ]; then
    cp ~/marketing/data/metrics.db "$BACKUP_DIR/metrics.db"
fi

# Keep only last 30 days of backups
find ~/trading-system/backups -maxdepth 1 -type d -mtime +30 -exec rm -rf {} +

echo "$(date '+%Y-%m-%d %H:%M:%S') [BACKUP] Complete. Saved to $BACKUP_DIR"
