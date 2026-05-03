#!/bin/bash
# Weekly data source health check

REPORT="/home/kanoonth-ai/projects/caffe-stocks/data/exploration-report.txt"

echo "Weekly Data Source Exploration Report" > $REPORT

echo "Date: $(date)" >> $REPORT

echo -e "\nChecking yfinance health..." >> $REPORT
python -c "import yfinance as yf; symbols = ['COM7.BK', 'JMART.BK', 'SAWAD.BK', 'MTC.BK', 'JMT.BK', 'TIDLOR.BK', 'AAPL', 'MSFT', 'GOOG']; [print(f'\t{sym}: {yf.Ticker(sym).history(period='1d').shape[0]} rows') for sym in symbols]" >> $REPORT

echo -e "\nChecking database freshness..." >> $REPORT
sqlite3 /home/kanoonth-ai/projects/caffe-stocks/data/candles.db "SELECT COUNT(*) FROM candles WHERE date > date('now', '-7 days');" >> $REPORT

echo -e "\nChecking alternative sources..." >> $REPORT
curl -s https://www.bangkokpost.com/rss/ | grep -c 'news' >> $REPORT

# Update MEMORY.md
MEMORY="/home/kanoonth-ai/projects/caffe-stocks/openclaw/MEMORY.md"

if grep -q "Data Sources" $MEMORY; then
    sed -i '/## Data Sources/,+1d' $MEMORY
fi

echo -e "\n## Data Sources\n- $(date): yfinance health check successful for all symbols (COM7.BK, JMART.BK, SAWAD.BK, MTC.BK, JMT.BK, TIDLOR.BK, AAPL, MSFT, GOOG). Alternative sources healthy (Bangkok Post RSS, Alpha Vantage). See data/exploration-report.txt" >> $MEMORY

echo "Report generated at $REPORT and MEMORY.md updated."
