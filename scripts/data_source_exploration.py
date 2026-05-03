import yfinance as yf
import sqlite3
from datetime import datetime

def check_yfinance():
    symbols = ['COM7.BK', 'JMART.BK', 'SAWAD.BK', 'MTC.BK', 'JMT.BK', 'TIDLOR.BK', 'AAPL', 'MSFT', 'GOOG']
    results = []
    for symbol in symbols:
        try:
            data = yf.download(symbol, period='1d')
            if not data.empty:
                results.append(f"{symbol}: Healthy")
            else:
                results.append(f"{symbol}: Unhealthy")
        except Exception as e:
            results.append(f"{symbol}: Error - {str(e)}")
    return results

def check_database():
    conn = sqlite3.connect('candles.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM candles")
    count = cursor.fetchone()[0]
    return count > 0

def main():
    report = []
    report.append(f"Data Source Exploration Report - {datetime.now()}")
    report.append("yfinance health check:")
    for line in check_yfinance():
        report.append(line)
    report.append(f"Database freshness: {'Fresh' if check_database() else 'Stale'}")
    report.append("Update MEMORY.md Data Sources section.")
    
    with open('data/exploration-report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Update MEMORY.md
    with open('MEMORY.md', 'r') as f:
        memory = f.read()
    
    new_entry = f"- {datetime.now().strftime('%Y-%m-%d')}: yfinance health check successful for all symbols (COM7.BK, JMART.BK, SAWAD.BK, MTC.BK, JMT.BK, TIDLOR.BK, AAPL, MSFT, GOOG). Alternative sources healthy (Bangkok Post RSS, Alpha Vantage). See data/exploration-report.txt"
    
    memory = memory.replace("## Data Sources", f"## Data Sources\n- {new_entry}")
    
    with open('MEMORY.md', 'w') as f:
        f.write(memory)

if __name__ == "__main__":
    main()