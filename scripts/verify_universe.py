"""Quick DB verification after universe expansion."""
import sqlite3

NEW_MAI = ['MOONG.BK', 'BIZ.BK', 'RBF.BK', 'WICE.BK', 'THMUI.BK',
           'PROEN.BK', 'JR.BK', 'CHAYO.BK', 'NCL.BK', 'TAKUNI.BK']
NEW_ETF = ['TDEX.BK', '1DIV.BK', 'GLD.BK', 'HEALTH.BK', 'CHINA.BK', 'ENGY.BK']

conn = sqlite3.connect('/home/kanoonth-ai/projects/caffe-stocks/data/candles.db')
cur = conn.cursor()
cur.execute('SELECT COUNT(DISTINCT symbol), COUNT(*) FROM candles_raw')
distinct, total = cur.fetchone()
print(f'distinct_symbols={distinct}  total_rows={total}')

print('\nNew Mai symbols in DB:')
for s in NEW_MAI:
    cur.execute('SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM candles_raw WHERE symbol=?', (s,))
    n, lo, hi = cur.fetchone()
    print(f'  {s:<12} rows={n:<5} {lo} -> {hi}')

print('\nNew ETF symbols in DB:')
for s in NEW_ETF:
    cur.execute('SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM candles_raw WHERE symbol=?', (s,))
    n, lo, hi = cur.fetchone()
    print(f'  {s:<12} rows={n:<5} {lo} -> {hi}')

new_total = sum(1 for _ in NEW_MAI + NEW_ETF)
cur.execute(
    'SELECT COUNT(*) FROM candles_raw WHERE symbol IN (' +
    ','.join('?' * new_total) + ')',
    NEW_MAI + NEW_ETF
)
new_rows = cur.fetchone()[0]
print(f'\nTotal rows from new symbols: {new_rows}')
conn.close()
