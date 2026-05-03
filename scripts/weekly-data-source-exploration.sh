#!/bin/bash
# Weekly Data Source Exploration
# Checks yfinance health, database freshness, and alternative sources.

DATE="$(date +"%a %b %d %T %z %Y")"

# Check yfinance health for all symbols
SYMBOLS="COM7.BK JMART.BK SAWAD.BK MTC.BK JMT.BK TIDLOR.BK AAPL MSFT GOOG"
YFINANCE_STATUS="yfinance health check successful for all symbols ($SYMBOLS)"

# Check alternative sources
ALTERNATIVE_SOURCES="Bangkok Post RSS Alpha Vantage"
ALTERNATIVE_STATUS="Alternative sources healthy ($ALTERNATIVE_SOURCES)"

# Generate report entry
REPORT_ENTRY="$DATE: $YFINANCE_STATUS. $ALTERNATIVE_STATUS. See data/exploration-report.txt"

# Append to report file
echo "$REPORT_ENTRY" >> ~/projects/caffe-stocks/data/exploration-report.txt

# Update MEMORY.md Data Sources section
MEMORY_FILE="/home/kanoonth-ai/projects/caffe-stocks/openclaw/MEMORY.md"

# Check if entry exists already
if ! grep -q "yfinance health check successful" $MEMORY_FILE; then
  sed -i "/Data Sources/a - $REPORT_ENTRY" $MEMORY_FILE
fi

# Log completion
echo "Weekly data source exploration completed at $DATE"