"""Phase 1 probe: test yfinance coverage for Mai/TDEX/SET symbols.

Writes a results table to stdout. Used once to decide whether to expand
the universe via yfinance or to seek an alternative data source.
"""
import sys
import time
import sqlite3
import yfinance as yf

# Sample groups
MAI_SAMPLE = [
    'MOONG.BK', 'BIZ.BK', 'RBF.BK', 'WICE.BK',
    'THMUI.BK', 'PROEN.BK', 'HEMP.BK',
]
TDEX_SAMPLE = [
    'TDEX.BK', 'BANK.BK', 'ENGY.BK', 'ENY.BK',
    '1DIV.BK', 'EBANK.BK', 'EFINANCE.BK', 'GLD.BK',
]
SET_SANITY = ['KBANK.BK']


def probe(symbol):
    try:
        df = yf.download(symbol, period='1y', interval='1d',
                         progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return ('EMPTY', 0, None, None)
        rows = len(df)
        last = df.index[-1].strftime('%Y-%m-%d')
        # Try to grab info to check exchange (best-effort, may fail)
        exch = None
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}
            exch = info.get('exchange') or info.get('fullExchangeName')
        except Exception:
            exch = '?'
        return ('OK', rows, last, exch)
    except Exception as e:
        return ('ERROR', 0, None, str(e)[:120])


def main():
    print('=' * 80)
    print('Phase 1 probe: yfinance Thai market coverage')
    print('=' * 80)

    # Baseline DB stats
    try:
        conn = sqlite3.connect('/home/kanoonth-ai/projects/caffe-stocks/data/candles.db')
        cur = conn.cursor()
        cur.execute('SELECT COUNT(DISTINCT symbol), COUNT(*) FROM candles_raw')
        d, t = cur.fetchone()
        print(f'\nDB baseline: distinct_symbols={d}  total_rows={t}\n')
        conn.close()
    except Exception as e:
        print(f'DB baseline error: {e}')

    groups = [('MAI', MAI_SAMPLE), ('TDEX/ETF', TDEX_SAMPLE), ('SET-sanity', SET_SANITY)]

    print(f'{"GROUP":<10} {"SYMBOL":<14} {"STATUS":<8} {"ROWS":>6}  {"LAST":<12} EXCH/ERR')
    print('-' * 80)
    summary = {'MAI': [0, 0], 'TDEX/ETF': [0, 0], 'SET-sanity': [0, 0]}
    for grp, syms in groups:
        for s in syms:
            status, rows, last, exch = probe(s)
            print(f'{grp:<10} {s:<14} {status:<8} {rows:>6}  {str(last):<12} {exch}')
            summary[grp][1] += 1
            if status == 'OK' and rows > 0:
                summary[grp][0] += 1
            time.sleep(0.4)  # gentle throttle

    print()
    print('Summary (covered/total):')
    for k, v in summary.items():
        print(f'  {k}: {v[0]}/{v[1]}')

    # Decision rule
    mai_cov = summary['MAI'][0] / max(1, summary['MAI'][1])
    etf_cov = summary['TDEX/ETF'][0] / max(1, summary['TDEX/ETF'][1])
    print()
    print(f'MAI coverage: {mai_cov:.0%} | TDEX/ETF coverage: {etf_cov:.0%}')
    if mai_cov >= 0.5 and etf_cov >= 0.5:
        print('DECISION: yfinance covers Mai + TDEX adequately -> proceed with Phase 2.')
    else:
        print('DECISION: Coverage gap detected -> recommend alternative data source.')


if __name__ == '__main__':
    main()
