"""Phase 2 expansion probe: test a broader set of candidate Mai stocks
and Thai ETFs to confirm yfinance coverage before adding to symbols.py.

Outputs two python lists ready to paste: VERIFIED_MAI and VERIFIED_ETF.
"""
import time
import yfinance as yf

# Candidate Mai-listed companies (well-known names, will verify each)
MAI_CANDIDATES = [
    'MOONG.BK', 'BIZ.BK', 'RBF.BK', 'WICE.BK', 'THMUI.BK', 'PROEN.BK',
    'JR.BK', 'CHAYO.BK', 'NCL.BK', 'TAKUNI.BK', 'SAMART.BK',
    'PROUD.BK', 'NEX.BK', 'SUN.BK', 'ARROW.BK', 'EE.BK',
    'PSP.BK', 'INSET.BK', 'D.BK', 'TPCH.BK', 'CHO.BK',
    'NCAP.BK', 'KK.BK', 'BLAND.BK', 'ZIGA.BK', 'TSE.BK',
    'TPLAS.BK', 'ALL.BK', 'MORE.BK',
]

# Candidate Thai ETFs (TDEX/ThaiDEX family + others)
ETF_CANDIDATES = [
    'TDEX.BK',     # ThaiDEX SET50
    '1DIV.BK',     # MFC ThaiDEX SET High Dividend
    'GLD.BK',      # SPDR Gold (ThaiDEX)
    'ENGY.BK',     # Energy sector ETF
    'BANK.BK',     # Bank sector
    'COMM.BK',     # Commerce sector
    'FOOD.BK',     # Food sector
    'ICT.BK',      # Tech / ICT
    'PROP.BK',     # Property sector
    'HEALTH.BK',   # Health sector
    'KSET50.BK',   # KTAM SET50
    'BCAP.BK',     # BCAP Mid/Small
    'TH100.BK',    # SET100 ETF
    'MSET50.BK',   # MFC SET50
    'EBANK.BK',    # MFC Bank Sector
    'ECOMM.BK',    # MFC Commerce
    'EICT.BK',     # MFC ICT
    'EHEALTH.BK',  # MFC Health
    'CHINA.BK',    # ThaiDEX China
    'CG.BK',       # CG ETF
    'GLOBAL.BK',   # already SET, skip overlap
    'GOLD99.BK',
    'GOLDFUT.BK',
    'TGOLDETF.BK',
    'GOLD.BK',
]


def probe(symbol):
    try:
        df = yf.download(symbol, period='1y', interval='1d',
                         progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return ('EMPTY', 0, None)
        last = df.index[-1].strftime('%Y-%m-%d')
        return ('OK', len(df), last)
    except Exception as e:
        return ('ERROR', 0, str(e)[:100])


def main():
    print('=== Mai candidates ===')
    verified_mai = []
    for s in MAI_CANDIDATES:
        st, n, last = probe(s)
        flag = 'OK' if st == 'OK' and n > 100 else 'SKIP'
        print(f'  {s:<14} {st:<6} rows={n:<4} last={last}  -> {flag}')
        if flag == 'OK':
            verified_mai.append(s)
        time.sleep(0.3)

    print('\n=== ETF candidates ===')
    verified_etf = []
    for s in ETF_CANDIDATES:
        st, n, last = probe(s)
        flag = 'OK' if st == 'OK' and n > 50 else 'SKIP'
        print(f'  {s:<14} {st:<6} rows={n:<4} last={last}  -> {flag}')
        if flag == 'OK':
            verified_etf.append(s)
        time.sleep(0.3)

    print('\n--- VERIFIED LISTS ---')
    print(f'VERIFIED_MAI ({len(verified_mai)}):')
    print(verified_mai)
    print(f'\nVERIFIED_ETF ({len(verified_etf)}):')
    print(verified_etf)


if __name__ == '__main__':
    main()
