"""Single source of truth for trading system symbols and sector mappings."""

# All tracked symbols (70 stocks)
SYMBOLS = [
    # Original 10
    'COM7.BK', 'PTT.BK', 'ADVANC.BK', 'CPALL.BK', 'SCC.BK',
    'BDMS.BK', 'MINT.BK', 'AWC.BK', 'CPN.BK', 'GULF.BK',
    # SET50 additions
    'KBANK.BK', 'BBL.BK', 'KTB.BK', 'SCB.BK', 'TTB.BK',
    'PTTEP.BK', 'PTTGC.BK', 'TOP.BK', 'OR.BK',
    'AOT.BK', 'BEM.BK', 'BTS.BK',
    'CPF.BK', 'HMPRO.BK', 'CRC.BK',
    'IVL.BK', 'DELTA.BK', 'KCE.BK',
    'TRUE.BK', 'BH.BK', 'EGCO.BK', 'BANPU.BK', 'GPSC.BK',
    # SET100 additions
    'SAWAD.BK', 'TISCO.BK', 'MTC.BK',
    'RATCH.BK', 'BGRIM.BK', 'IRPC.BK',
    'LH.BK', 'AP.BK', 'SPALI.BK', 'ORI.BK',
    'CBG.BK', 'OSP.BK', 'STGT.BK',
    'SCGP.BK', 'KTC.BK',
    'GLOBAL.BK', 'TU.BK', 'MAJOR.BK',
    # Additional liquid SET stocks
    'WHA.BK', 'CENTEL.BK', 'BEAUTY.BK', 'JMT.BK',
    'SINGER.BK', 'RS.BK', 'JMART.BK',
    'BJC.BK', 'PLANB.BK', 'VGI.BK',
    'SUPER.BK', 'BCPG.BK', 'GUNKUL.BK',
    'STA.BK', 'THANI.BK', 'NER.BK',
    'MEGA.BK', 'FORTH.BK', 'SYNEX.BK',
]

# SET Index for regime features
SET_INDEX = '^SET.BK'

# All symbols including index (for OHLCV fetch)
ALL_SYMBOLS = SYMBOLS + [SET_INDEX]

# Stock to SET sector mapping
STOCK_SECTOR = {
    # Original 10
    'COM7.BK': 'COMM', 'PTT.BK': 'ENERG', 'ADVANC.BK': 'ICT',
    'CPALL.BK': 'COMM', 'SCC.BK': 'CONMAT', 'BDMS.BK': 'HELTH',
    'MINT.BK': 'TOURISM', 'AWC.BK': 'PROP', 'CPN.BK': 'PROP',
    'GULF.BK': 'ENERG',
    # Banking
    'KBANK.BK': 'BANK', 'BBL.BK': 'BANK', 'KTB.BK': 'BANK',
    'SCB.BK': 'BANK', 'TTB.BK': 'BANK', 'TISCO.BK': 'BANK',
    # Energy
    'PTTEP.BK': 'ENERG', 'PTTGC.BK': 'PETRO', 'TOP.BK': 'ENERG',
    'OR.BK': 'ENERG', 'BANPU.BK': 'ENERG', 'GPSC.BK': 'ENERG',
    'IRPC.BK': 'PETRO', 'BCPG.BK': 'ENERG', 'GUNKUL.BK': 'ENERG',
    'SUPER.BK': 'ENERG', 'EGCO.BK': 'ENERG', 'RATCH.BK': 'ENERG',
    'BGRIM.BK': 'ENERG',
    # Transport / Infrastructure
    'AOT.BK': 'TRANS', 'BEM.BK': 'TRANS', 'BTS.BK': 'TRANS',
    # Consumer / Commerce
    'CPF.BK': 'FOOD', 'HMPRO.BK': 'COMM', 'CRC.BK': 'COMM',
    'BJC.BK': 'COMM', 'MAJOR.BK': 'MEDIA', 'TU.BK': 'FOOD',
    'GLOBAL.BK': 'FOOD', 'OSP.BK': 'FOOD', 'CBG.BK': 'FOOD',
    # Industrial / Tech
    'IVL.BK': 'PETRO', 'DELTA.BK': 'ETRON', 'KCE.BK': 'ETRON',
    'SCGP.BK': 'PKG', 'STGT.BK': 'HELTH', 'STA.BK': 'AGRI',
    'NER.BK': 'AGRI', 'SYNEX.BK': 'ICT', 'FORTH.BK': 'ICT',
    # Telecom
    'TRUE.BK': 'ICT',
    # Healthcare
    'BH.BK': 'HELTH', 'MEGA.BK': 'HELTH',
    # Property / Real Estate
    'LH.BK': 'PROP', 'AP.BK': 'PROP', 'SPALI.BK': 'PROP',
    'ORI.BK': 'PROP', 'WHA.BK': 'PROP',
    # Finance / Consumer Finance
    'SAWAD.BK': 'FIN', 'MTC.BK': 'FIN', 'KTC.BK': 'FIN',
    'THANI.BK': 'FIN', 'JMT.BK': 'FIN',
    # Tourism / Hotels
    'CENTEL.BK': 'TOURISM',
    # Media / Services
    'PLANB.BK': 'MEDIA', 'VGI.BK': 'MEDIA',
    'RS.BK': 'MEDIA', 'SINGER.BK': 'COMM',
    'BEAUTY.BK': 'COMM', 'JMART.BK': 'COMM',
}
