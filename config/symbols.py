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
    'AAI.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'AAV.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'AMATA.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'BAM.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'BCH.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'BCP.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'BTG.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'CCET.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'CK.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'CKP.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'CPAXT.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'EA.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'ERW.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'FSMART.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'HANA.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'ICHI.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'ITC.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'KKP.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'M.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'MBK.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'PR9.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'PRM.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'PSL.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'RCL.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'SMT.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'SPRC.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'SRICHA.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'STECON.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'STPI.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'TAE.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'TCAP.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'TFG.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'THCOM.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'TIDLOR.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'TLI.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'TPIPP.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'TSE.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'WHAUP.BK',  # extra SET liquid name (TradingView active list, batch 2)
    'ADVICE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AEONTS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AFC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AIMCG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AJ.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'ALLA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'ALUCON.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AMANAH.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AMARC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AMARIN.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AMR.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AQUA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'ASAP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'ASIA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AUCT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'AXTRART.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'B52.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BAY.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BBIK.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BCT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BE8.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BEC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BKA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BKIH.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BLA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BPP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BTNC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'BTW.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CEYE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CHARAN.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CHEWA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CHOTI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'COMAN.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CPANEL.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CPT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CPW.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CRD.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CREDIT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'CSP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'DCON.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'DOHOME.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'DPAINT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'DTCI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'EASON.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'EASTW.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'EMC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'EVER.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'FANCY.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'FNS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'FPT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'FSX.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'GEL.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'GENCO.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'GFPT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'GLAND.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'GPI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'GRAMMY.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'GRAND.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'I2.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'ICC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'IDG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'IFS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'IIG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'INET.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'INSET.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'IT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'JAK.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'JTS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'KC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'KCAR.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'KCG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'KGI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'KISS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'KK.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'KTIS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LALIN.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LANNA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LEE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LHFG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LIT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LOXLEY.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LPN.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'LTS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'M-PAT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MALEE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MASTEC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MCA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MCOT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MDX.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MFC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MFEC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MIDA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MJD.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MK.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'MRDIYT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'NCH.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'NCP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'NEP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'NOBLE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'NYT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'OCC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'OGC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PANEL.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PB.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PEER.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PERM.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PLE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PPP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PPPM.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PQS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PRAKIT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PRECHA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PRIME.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PROS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PSH.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PTC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'PTG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'QDC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'QH.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'QLT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'QTCG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'RAM.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'RICHY.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'ROCK.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'ROJNA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'RP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'RWI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SABINA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SAMART.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SAMCO.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SAWANG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SCCC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SCG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SEAFCO.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SENA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SIRI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SIS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SISB.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SJWD.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SK.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SKE.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SNC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SPCG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SPI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SPREME.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SR.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'STX.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SUSCO.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SVI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SVOA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SVR.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'SYNTEC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TASCO.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TCJ.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TCMC.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TEAM.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TERA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TFMAMA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TGPRO.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'THAI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TIGER.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TKS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TKT.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TM.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TOA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TPA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TPCS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TPLAS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TPOLY.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TRUBB.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TSI.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TSR.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TTA.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TTCL.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TTW.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TU-PF.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TVDH.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TVO.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TWP.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'TWZ.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'UNIQ.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'UPF.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    # 'VAYU1.BK' removed 2026-05-06: yfinance returns only 1 bar (too new, insufficient for indicators)
    'VS.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'WIN.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'WPH.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'WSOL.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
    'YGG.BK',  # batch 3 — TradingView/Investing wider screen 2026-05-06
]

# Mai-listed validation batch (~10) - probe-verified via yfinance 2026-05-05.
# Full Mai universe (~200) is a follow-up after end-to-end ingest succeeds.
MAI_SYMBOLS = [
    'MOONG.BK', 'BIZ.BK', 'RBF.BK', 'WICE.BK', 'THMUI.BK',
    'PROEN.BK', 'JR.BK', 'CHAYO.BK', 'NCL.BK', 'TAKUNI.BK',
    'AGE.BK',  # added by full-Mai expansion 2026-05-06
    'AIRA.BK',  # added by full-Mai expansion 2026-05-06
    'AKP.BK',  # added by full-Mai expansion 2026-05-06
    'ARIP.BK',  # added by full-Mai expansion 2026-05-06
    'ARROW.BK',  # added by full-Mai expansion 2026-05-06
    'ASN.BK',  # added by full-Mai expansion 2026-05-06
    'ATP30.BK',  # added by full-Mai expansion 2026-05-06
    'BGT.BK',  # added by full-Mai expansion 2026-05-06
    'BLESS.BK',  # added by full-Mai expansion 2026-05-06
    'BOL.BK',  # added by full-Mai expansion 2026-05-06
    'BPS.BK',  # added by full-Mai expansion 2026-05-06
    'BSM.BK',  # added by full-Mai expansion 2026-05-06
    'CAZ.BK',  # added by full-Mai expansion 2026-05-06
    'CHIC.BK',  # added by full-Mai expansion 2026-05-06
    'CHOW.BK',  # added by full-Mai expansion 2026-05-06
    'CI.BK',  # added by full-Mai expansion 2026-05-06
    'CIG.BK',  # added by full-Mai expansion 2026-05-06
    'CMO.BK',  # added by full-Mai expansion 2026-05-06
    'CPR.BK',  # added by full-Mai expansion 2026-05-06
    'CRANE.BK',  # added by full-Mai expansion 2026-05-06
    'DEMCO.BK',  # added by full-Mai expansion 2026-05-06
    'DIMET.BK',  # added by full-Mai expansion 2026-05-06
    'DOD.BK',  # added by full-Mai expansion 2026-05-06
    'DV8.BK',  # added by full-Mai expansion 2026-05-06
    'ECF.BK',  # added by full-Mai expansion 2026-05-06
    'EURO.BK',  # added by full-Mai expansion 2026-05-06
    'FLOYD.BK',  # added by full-Mai expansion 2026-05-06
    'GCAP.BK',  # added by full-Mai expansion 2026-05-06
    'GLORY.BK',  # added by full-Mai expansion 2026-05-06
    'HARN.BK',  # added by full-Mai expansion 2026-05-06
    'ILINK.BK',  # added by full-Mai expansion 2026-05-06
    'IRCP.BK',  # added by full-Mai expansion 2026-05-06
    'JSP.BK',  # added by full-Mai expansion 2026-05-06
    'KASET.BK',  # added by full-Mai expansion 2026-05-06
    'KCM.BK',  # added by full-Mai expansion 2026-05-06
    'KGEN.BK',  # added by full-Mai expansion 2026-05-06
    'KIAT.BK',  # added by full-Mai expansion 2026-05-06
    'LDC.BK',  # added by full-Mai expansion 2026-05-06
    'LEO.BK',  # added by full-Mai expansion 2026-05-06
    'LTMH.BK',  # added by full-Mai expansion 2026-05-06
    'MBAX.BK',  # added by full-Mai expansion 2026-05-06
    'META.BK',  # added by full-Mai expansion 2026-05-06
    'MITSIB.BK',  # added by full-Mai expansion 2026-05-06
    'MMM.BK',  # added by full-Mai expansion 2026-05-06
    'MORE.BK',  # added by full-Mai expansion 2026-05-06
    'MOTHER.BK',  # added by full-Mai expansion 2026-05-06
    'MPJ.BK',  # added by full-Mai expansion 2026-05-06
    'NAT.BK',  # added by full-Mai expansion 2026-05-06
    'NDR.BK',  # added by full-Mai expansion 2026-05-06
    'NETBAY.BK',  # added by full-Mai expansion 2026-05-06
    'NPK.BK',  # added by full-Mai expansion 2026-05-06
    'NTF.BK',  # added by full-Mai expansion 2026-05-06
    'PHOL.BK',  # added by full-Mai expansion 2026-05-06
    'PICO.BK',  # added by full-Mai expansion 2026-05-06
    'PLANET.BK',  # added by full-Mai expansion 2026-05-06
    'PMC.BK',  # added by full-Mai expansion 2026-05-06
    'PPM.BK',  # added by full-Mai expansion 2026-05-06
    'PPS.BK',  # added by full-Mai expansion 2026-05-06
    'PYLON.BK',  # added by full-Mai expansion 2026-05-06
    'SAAM.BK',  # added by full-Mai expansion 2026-05-06
    'SAF.BK',  # added by full-Mai expansion 2026-05-06
    'SALEE.BK',  # added by full-Mai expansion 2026-05-06
    'SANKO.BK',  # added by full-Mai expansion 2026-05-06
    'SCL.BK',  # added by full-Mai expansion 2026-05-06
    'SEI.BK',  # added by full-Mai expansion 2026-05-06
    'SGF.BK',  # added by full-Mai expansion 2026-05-06
    'SIMAT.BK',  # added by full-Mai expansion 2026-05-06
    'SKIN.BK',  # added by full-Mai expansion 2026-05-06
    'SMART.BK',  # added by full-Mai expansion 2026-05-06
    'SONIC.BK',  # added by full-Mai expansion 2026-05-06
    'SPTX.BK',  # added by full-Mai expansion 2026-05-06
    'SPVI.BK',  # added by full-Mai expansion 2026-05-06
    'STC.BK',  # added by full-Mai expansion 2026-05-06
    'SWC.BK',  # added by full-Mai expansion 2026-05-06
    'TAPAC.BK',  # added by full-Mai expansion 2026-05-06
    'TMC.BK',  # added by full-Mai expansion 2026-05-06
    'TMW.BK',  # added by full-Mai expansion 2026-05-06
    'TNDT.BK',  # added by full-Mai expansion 2026-05-06
    'TNH.BK',  # added by full-Mai expansion 2026-05-06
    'TPAC.BK',  # added by full-Mai expansion 2026-05-06
    'TQR.BK',  # added by full-Mai expansion 2026-05-06
    'TRC.BK',  # added by full-Mai expansion 2026-05-06
    'TRT.BK',  # added by full-Mai expansion 2026-05-06
    'TRV.BK',  # added by full-Mai expansion 2026-05-06
    'TVT.BK',  # added by full-Mai expansion 2026-05-06
    'UBA.BK',  # added by full-Mai expansion 2026-05-06
    'UBIS.BK',  # added by full-Mai expansion 2026-05-06
    'UEC.BK',  # added by full-Mai expansion 2026-05-06
    'UKEM.BK',  # added by full-Mai expansion 2026-05-06
    'UMS.BK',  # added by full-Mai expansion 2026-05-06
    'WARRIX.BK',  # added by full-Mai expansion 2026-05-06
    'WINNER.BK',  # added by full-Mai expansion 2026-05-06
    'YONG.BK',  # added by full-Mai expansion 2026-05-06
    'YUASA.BK',  # added by full-Mai expansion 2026-05-06
    'ZIGA.BK',  # added by full-Mai expansion 2026-05-06
]

# Thai ETFs (TDEX/ThaiDEX family) - probe-verified via yfinance 2026-05-05.
# ENGY.BK had last bar 2025-10-06 (likely thinly traded / suspended);
# included for completeness, ingest will pick up whatever yfinance returns.
ETF_SYMBOLS = [
    'TDEX.BK',    # ThaiDEX SET50
    '1DIV.BK',    # MFC ThaiDEX SET High Dividend
    'GLD.BK',     # SPDR Gold (ThaiDEX)
    'HEALTH.BK',  # Health sector ETF
    'CHINA.BK',   # ThaiDEX China
    'ENGY.BK',    # Energy sector ETF (stale on yfinance, kept for parity)
]

# SET Index for regime features
SET_INDEX = '^SET.BK'

# All symbols including index (for OHLCV fetch)

# Thai REITs and property funds (yfinance-validated)
REIT_SYMBOLS = [
    'AIMIRT.BK',
    'ALLY.BK',
    'AMATAR.BK',
    'AMATAV.BK',
    'BJCHI.BK',
    'BLAND.BK',
    'BTSGIF.BK',
    'BWG.BK',
    'CPNCG.BK',
    'CPNREIT.BK',
    'CPTGF.BK',
    'DIF.BK',
    'EGATIF.BK',
    'FTREIT.BK',
    'FUTUREPF.BK',
    'GROREIT.BK',
    'GVREIT.BK',
    'IMPACT.BK',
    'INETREIT.BK',
    'JASIF.BK',
    'KTBSTMR.BK',
    'LHHOTEL.BK',
    'LHPF.BK',
    'LHSC.BK',
    'MJLF.BK',
    'MNIT.BK',
    'MNRF.BK',
    'POPF.BK',
    'PROUD.BK',
    'QHHR.BK',
    'QHOP.BK',
    'QHPF.BK',
    'SRIPANWA.BK',
    'TFFIF.BK',
    'WHABT.BK',
    'WHAIR.BK',
    'WHART.BK',
]


# Thai Depositary Receipts (yfinance-validated)
DR_SYMBOLS = [
    'AAPL01.BK',
    'AAPL19.BK',
    'AMZN01.BK',
    'BABA01.BK',
    'GOOGL01.BK',
    'MSFT01.BK',
    'MSFT19.BK',
    'NVDA01.BK',
    'NVDA19.BK',
    'TSLA01.BK',
]

ALL_SYMBOLS = SYMBOLS + MAI_SYMBOLS + ETF_SYMBOLS + REIT_SYMBOLS + DR_SYMBOLS + [SET_INDEX]

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
    # Mai (auto-mapped from yfinance industry/sector)
    'AGE.BK': 'ENERG',  # Thermal Coal
    'AIRA.BK': 'FIN',  # Capital Markets
    'AKP.BK': 'INDUS',  # Waste Management
    'ARIP.BK': 'INDUS',  # Specialty Business Services
    'ARROW.BK': 'CONMAT',  # Building Products & Equipment
    'ASN.BK': 'FIN',  # Insurance Brokers
    'ATP30.BK': 'INDUS',  # Railroads
    'BGT.BK': 'COMM',  # Apparel Retail
    'BIZ.BK': 'HELTH',  # Medical Devices
    'BLESS.BK': 'PROP',  # Real Estate - Development
    'BOL.BK': 'INDUS',  # Consulting Services
    'BPS.BK': 'ICT',  # Electronics & Computer Distribution
    'BSM.BK': 'CONMAT',  # Building Products & Equipment
    'CAZ.BK': 'CONMAT',  # Engineering & Construction
    'CHAYO.BK': 'FIN',  # Credit Services
    'CHIC.BK': 'COMM',  # Furnishings, Fixtures & Appliances
    'CHOW.BK': 'STEEL',  # Steel
    'CI.BK': 'PROP',  # Real Estate - Development
    'CIG.BK': 'INDUS',  # Electrical Equipment & Parts
    'CMO.BK': 'MEDIA',  # Advertising Agencies
    'CPR.BK': 'AUTO',  # Auto Parts
    'CRANE.BK': 'INDUS',  # Rental & Leasing Services
    'DEMCO.BK': 'CONMAT',  # Engineering & Construction
    'DIMET.BK': 'PETRO',  # Specialty Chemicals
    'DOD.BK': 'FOOD',  # Packaged Foods
    'DV8.BK': 'MEDIA',  # Broadcasting
    'ECF.BK': 'COMM',  # Furnishings, Fixtures & Appliances
    'EURO.BK': 'COMM',  # Furnishings, Fixtures & Appliances
    'FLOYD.BK': 'CONMAT',  # Engineering & Construction
    'GCAP.BK': 'FIN',  # Credit Services
    'GLORY.BK': 'MEDIA',  # Publishing
    'HARN.BK': 'INDUS',  # Security & Protection Services
    'ILINK.BK': 'ICT',  # Communication Equipment
    'IRCP.BK': 'ICT',  # Information Technology Services
    'JR.BK': 'CONMAT',  # Engineering & Construction
    'JSP.BK': 'FOOD',  # Packaged Foods
    'KASET.BK': 'FOOD',  # Packaged Foods
    'KCM.BK': 'STEEL',  # Metal Fabrication
    'KGEN.BK': 'INDUS',  # Railroads
    'KIAT.BK': 'TRANS',  # Trucking
    'LDC.BK': 'HELTH',  # Medical Care Facilities
    'LEO.BK': 'TRANS',  # Integrated Freight & Logistics
    'LTMH.BK': 'MEDIA',  # Advertising Agencies
    'MBAX.BK': 'PKG',  # Packaging & Containers
    'META.BK': 'ENERG',  # Utilities - Renewable
    'MITSIB.BK': 'AUTO',  # Auto & Truck Dealerships
    'MMM.BK': 'INDUS',  # unknown
    'MOONG.BK': 'COMM',  # Household & Personal Products
    'MORE.BK': 'ENERG',  # Utilities - Regulated Water
    'MOTHER.BK': 'COMM',  # Specialty Retail
    'MPJ.BK': 'TRANS',  # Integrated Freight & Logistics
    'NAT.BK': 'ICT',  # Information Technology Services
    'NCL.BK': 'TRANS',  # Integrated Freight & Logistics
    'NDR.BK': 'AUTO',  # Auto Parts
    'NETBAY.BK': 'ICT',  # Software - Application
    'NPK.BK': 'COMM',  # Apparel Manufacturing
    'NTF.BK': 'FOOD',  # Food Distribution
    'PHOL.BK': 'INDUS',  # Security & Protection Services
    'PICO.BK': 'MEDIA',  # Advertising Agencies
    'PLANET.BK': 'ICT',  # Communication Equipment
    'PMC.BK': 'PKG',  # Packaging & Containers
    'PPM.BK': 'INDUS',  # Industrial Distribution
    'PPS.BK': 'CONMAT',  # Engineering & Construction
    'PROEN.BK': 'ICT',  # Telecom Services
    'PYLON.BK': 'CONMAT',  # Engineering & Construction
    'RBF.BK': 'FOOD',  # Packaged Foods
    'SAAM.BK': 'ENERG',  # Utilities - Renewable
    'SAF.BK': 'STEEL',  # Steel
    'SALEE.BK': 'PETRO',  # Specialty Chemicals
    'SANKO.BK': 'AUTO',  # Auto Parts
    'SCL.BK': 'AUTO',  # Auto Parts
    'SEI.BK': 'HELTH',  # Medical Distribution
    'SGF.BK': 'FIN',  # Credit Services
    'SIMAT.BK': 'ICT',  # Electronics & Computer Distribution
    'SKIN.BK': 'COMM',  # Household & Personal Products
    'SMART.BK': 'INDUS',  # unknown
    'SONIC.BK': 'TRANS',  # Integrated Freight & Logistics
    'SPTX.BK': 'ICT',  # Software - Application
    'SPVI.BK': 'ICT',  # Electronics & Computer Distribution
    'STC.BK': 'CONMAT',  # Building Materials
    'SWC.BK': 'PETRO',  # Specialty Chemicals
    'TAKUNI.BK': 'CONMAT',  # Engineering & Construction
    'TAPAC.BK': 'INDUS',  # unknown
    'THMUI.BK': 'INDUS',  # Industrial Distribution
    'TMC.BK': 'INDUS',  # Farm & Heavy Construction Machinery
    'TMW.BK': 'PETRO',  # Specialty Chemicals
    'TNDT.BK': 'CONMAT',  # Engineering & Construction
    'TNH.BK': 'HELTH',  # Medical Care Facilities
    'TPAC.BK': 'PKG',  # Packaging & Containers
    'TQR.BK': 'FIN',  # Insurance - Reinsurance
    'TRC.BK': 'CONMAT',  # Engineering & Construction
    'TRT.BK': 'INDUS',  # Electrical Equipment & Parts
    'TRV.BK': 'AUTO',  # Auto Parts
    'TVT.BK': 'ICT',  # Entertainment
    'UBA.BK': 'INDUS',  # Waste Management
    'UBIS.BK': 'PETRO',  # Specialty Chemicals
    'UEC.BK': 'STEEL',  # Steel
    'UKEM.BK': 'PETRO',  # Chemicals
    'UMS.BK': 'ENERG',  # Thermal Coal
    'WARRIX.BK': 'TOURISM',  # Leisure
    'WICE.BK': 'TRANS',  # Integrated Freight & Logistics
    'WINNER.BK': 'FOOD',  # Packaged Foods
    'YONG.BK': 'CONMAT',  # Building Materials
    'YUASA.BK': 'AUTO',  # Auto Parts
    'ZIGA.BK': 'STEEL',  # Steel,
    # REITs (all PROP)
    'AIMIRT.BK': 'PROP',  # REIT
    'ALLY.BK': 'PROP',  # REIT
    'AMATAR.BK': 'PROP',  # REIT
    'AMATAV.BK': 'PROP',  # REIT
    'BJCHI.BK': 'PROP',  # REIT
    'BLAND.BK': 'PROP',  # REIT
    'BTSGIF.BK': 'PROP',  # REIT
    'BWG.BK': 'PROP',  # REIT
    'CPNCG.BK': 'PROP',  # REIT
    'CPNREIT.BK': 'PROP',  # REIT
    'CPTGF.BK': 'PROP',  # REIT
    'DIF.BK': 'PROP',  # REIT
    'EGATIF.BK': 'PROP',  # REIT
    'FTREIT.BK': 'PROP',  # REIT
    'FUTUREPF.BK': 'PROP',  # REIT
    'GROREIT.BK': 'PROP',  # REIT
    'GVREIT.BK': 'PROP',  # REIT
    'IMPACT.BK': 'PROP',  # REIT
    'INETREIT.BK': 'PROP',  # REIT
    'JASIF.BK': 'PROP',  # REIT
    'KTBSTMR.BK': 'PROP',  # REIT
    'LHHOTEL.BK': 'PROP',  # REIT
    'LHPF.BK': 'PROP',  # REIT
    'LHSC.BK': 'PROP',  # REIT
    'MJLF.BK': 'PROP',  # REIT
    'MNIT.BK': 'PROP',  # REIT
    'MNRF.BK': 'PROP',  # REIT
    'POPF.BK': 'PROP',  # REIT
    'PROUD.BK': 'PROP',  # REIT
    'QHHR.BK': 'PROP',  # REIT
    'QHOP.BK': 'PROP',  # REIT
    'QHPF.BK': 'PROP',  # REIT
    'SRIPANWA.BK': 'PROP',  # REIT
    'TFFIF.BK': 'PROP',  # REIT
    'WHABT.BK': 'PROP',  # REIT
    'WHAIR.BK': 'PROP',  # REIT
    'WHART.BK': 'PROP',  # REIT
    # Depositary Receipts (foreign equity exposure — own sector code)
    'AAPL01.BK': 'DR',  # Depositary Receipt (foreign)
    'AAPL19.BK': 'DR',  # Depositary Receipt (foreign)
    'AMZN01.BK': 'DR',  # Depositary Receipt (foreign)
    'BABA01.BK': 'DR',  # Depositary Receipt (foreign)
    'GOOGL01.BK': 'DR',  # Depositary Receipt (foreign)
    'MSFT01.BK': 'DR',  # Depositary Receipt (foreign)
    'MSFT19.BK': 'DR',  # Depositary Receipt (foreign)
    'NVDA01.BK': 'DR',  # Depositary Receipt (foreign)
    'NVDA19.BK': 'DR',  # Depositary Receipt (foreign)
    'TSLA01.BK': 'DR',  # Depositary Receipt (foreign)
    # SET extras (auto-mapped from yfinance)
    'AAI.BK': 'FOOD',  # Packaged Foods
    'AAV.BK': 'TRANS',  # Airlines
    'AMATA.BK': 'PROP',  # Real Estate - Development
    'BAM.BK': 'FIN',  # Asset Management
    'BCH.BK': 'HELTH',  # Medical Care Facilities
    'BCP.BK': 'PETRO',  # Oil & Gas Refining & Marketing
    'BTG.BK': 'AGRI',  # Farm Products
    'CCET.BK': 'ETRON',  # Computer Hardware
    'CK.BK': 'CONMAT',  # Engineering & Construction
    'CKP.BK': 'ENERG',  # Utilities - Independent Power Producers
    'CPAXT.BK': 'FOOD',  # Food Distribution
    'EA.BK': 'ENERG',  # Utilities - Renewable
    'ERW.BK': 'TOURISM',  # Lodging
    'FSMART.BK': 'ICT',  # Telecom Services
    'HANA.BK': 'ETRON',  # Electronic Components
    'ICHI.BK': 'COMM',  # Beverages - Non-Alcoholic
    'ITC.BK': 'FOOD',  # Packaged Foods
    'KKP.BK': 'FIN',  # Banks - Regional
    'MBK.BK': 'INDUS',  # Conglomerates
    'PR9.BK': 'HELTH',  # Medical Care Facilities
    'PRM.BK': 'ENERG',  # Oil & Gas Midstream
    'PSL.BK': 'TRANS',  # Marine Shipping
    'RCL.BK': 'TRANS',  # Marine Shipping
    'SMT.BK': 'ETRON',  # Semiconductors
    'SPRC.BK': 'PETRO',  # Oil & Gas Refining & Marketing
    'SRICHA.BK': 'CONMAT',  # Engineering & Construction
    'STECON.BK': 'CONMAT',  # Engineering & Construction
    'STPI.BK': 'STEEL',  # Metal Fabrication
    'TAE.BK': 'PETRO',  # Chemicals
    'TCAP.BK': 'FIN',  # Credit Services
    'TFG.BK': 'AGRI',  # Farm Products
    'THCOM.BK': 'ICT',  # Telecom Services
    'TIDLOR.BK': 'FIN',  # Credit Services
    'TLI.BK': 'FIN',  # Insurance - Life
    'TPIPP.BK': 'ENERG',  # Utilities - Renewable
    'TSE.BK': 'ENERG',  # Utilities - Renewable
    'WHAUP.BK': 'ENERG',  # Utilities - Regulated Water
    # Batch 3 (yfinance-validated wider screen)
    'ADVICE.BK': 'ICT',  # Electronics & Computer Distribution
    'AEONTS.BK': 'FIN',  # Credit Services
    'AFC.BK': 'COMM',  # Textile Manufacturing
    'AIMCG.BK': 'PROP',  # REIT - Retail
    'AJ.BK': 'PKG',  # Packaging & Containers
    'ALLA.BK': 'INDUS',  # Industrial Distribution
    'ALUCON.BK': 'STEEL',  # Aluminum
    'AMANAH.BK': 'FIN',  # Credit Services
    'AMARC.BK': 'ICT',  # Scientific & Technical Instruments
    'AMARIN.BK': 'MEDIA',  # Publishing
    'AMR.BK': 'CONMAT',  # Engineering & Construction
    'AQUA.BK': 'MEDIA',  # Advertising Agencies
    'ASAP.BK': 'INDUS',  # Rental & Leasing Services
    'ASIA.BK': 'TOURISM',  # Lodging
    'AUCT.BK': 'AUTO',  # Auto & Truck Dealerships
    'AXTRART.BK': 'PROP',  # REIT - Retail
    'B52.BK': 'PROP',  # Real Estate Services
    'BA.BK': 'TRANS',  # Airlines
    'BAY.BK': 'FIN',  # Banks - Regional
    'BBIK.BK': 'INDUS',  # Consulting Services
    'BCT.BK': 'PETRO',  # Specialty Chemicals
    'BE8.BK': 'ICT',  # Information Technology Services
    'BEC.BK': 'MEDIA',  # Broadcasting
    'BKA.BK': 'INDUS',  # unknown
    'BKIH.BK': 'FIN',  # Insurance - Diversified
    'BLA.BK': 'FIN',  # Insurance - Life
    'BPP.BK': 'ENERG',  # Utilities - Independent Power Producers
    'BTNC.BK': 'COMM',  # Apparel Retail
    'BTW.BK': 'STEEL',  # Metal Fabrication
    'CEYE.BK': 'TOURISM',  # Leisure
    'CHARAN.BK': 'FIN',  # Insurance - Property & Casualty
    'CHEWA.BK': 'PROP',  # Real Estate - Development
    'CHOTI.BK': 'FOOD',  # Packaged Foods
    'COMAN.BK': 'ICT',  # Electronics & Computer Distribution
    'CPANEL.BK': 'CONMAT',  # Building Materials
    'CPT.BK': 'INDUS',  # Electrical Equipment & Parts
    'CPW.BK': 'ICT',  # Electronics & Computer Distribution
    'CRD.BK': 'CONMAT',  # Engineering & Construction
    'CREDIT.BK': 'FIN',  # Banks - Regional
    'CSP.BK': 'STEEL',  # Steel
    'DCON.BK': 'CONMAT',  # Building Materials
    'DOHOME.BK': 'COMM',  # Home Improvement Retail
    'DPAINT.BK': 'PETRO',  # Specialty Chemicals
    'DTCI.BK': 'INDUS',  # Business Equipment & Supplies
    'EASON.BK': 'PETRO',  # Specialty Chemicals
    'EASTW.BK': 'ENERG',  # Utilities - Regulated Water
    'EMC.BK': 'CONMAT',  # Engineering & Construction
    'EVER.BK': 'PROP',  # Real Estate - Development
    'FANCY.BK': 'COMM',  # Furnishings, Fixtures & Appliances
    'FNS.BK': 'FIN',  # Asset Management
    'FPT.BK': 'PROP',  # Real Estate - Development
    'FSX.BK': 'FIN',  # Capital Markets
    'GEL.BK': 'CONMAT',  # Building Materials
    'GENCO.BK': 'INDUS',  # Waste Management
    'GFPT.BK': 'FOOD',  # Packaged Foods
    'GLAND.BK': 'PROP',  # Real Estate Services
    'GPI.BK': 'MEDIA',  # Advertising Agencies
    'GRAMMY.BK': 'ICT',  # Entertainment
    'GRAND.BK': 'TOURISM',  # Lodging
    'I2.BK': 'ICT',  # Communication Equipment
    'ICC.BK': 'COMM',  # Apparel Retail
    'IDG.BK': 'ICT',  # Information Technology Services
    'IFS.BK': 'FIN',  # Credit Services
    'IIG.BK': 'ICT',  # Information Technology Services
    'INET.BK': 'ICT',  # Information Technology Services
    'INSET.BK': 'ICT',  # Information Technology Services
    'IT.BK': 'ICT',  # Electronics & Computer Distribution
    'JAK.BK': 'PROP',  # Real Estate - Development
    'JTS.BK': 'ICT',  # Telecom Services
    'KC.BK': 'PROP',  # Real Estate - Development
    'KCAR.BK': 'INDUS',  # Rental & Leasing Services
    'KCG.BK': 'FOOD',  # Packaged Foods
    'KGI.BK': 'FIN',  # Capital Markets
    'KISS.BK': 'COMM',  # Household & Personal Products
    'KK.BK': 'FOOD',  # Grocery Stores
    'KTIS.BK': 'FOOD',  # Confectioners
    'LALIN.BK': 'PROP',  # Real Estate - Development
    'LANNA.BK': 'ENERG',  # Thermal Coal
    'LEE.BK': 'FOOD',  # Packaged Foods
    'LHFG.BK': 'FIN',  # Banks - Regional
    'LIT.BK': 'FIN',  # Credit Services
    'LOXLEY.BK': 'INDUS',  # Conglomerates
    'LPN.BK': 'PROP',  # Real Estate - Development
    'LTS.BK': 'COMM',  # Home Improvement Retail
    'M-PAT.BK': 'INDUS',  # unknown
    'MALEE.BK': 'FOOD',  # Packaged Foods
    'MASTEC.BK': 'INDUS',  # Industrial Distribution
    'MC.BK': 'COMM',  # Apparel Manufacturing
    'MCA.BK': 'MEDIA',  # Advertising Agencies
    'MCOT.BK': 'MEDIA',  # Broadcasting
    'MDX.BK': 'PROP',  # Real Estate - Development
    'MFC.BK': 'FIN',  # Asset Management
    'MFEC.BK': 'ICT',  # Information Technology Services
    'MIDA.BK': 'FIN',  # Credit Services
    'MJD.BK': 'PROP',  # Real Estate - Development
    'MK.BK': 'PROP',  # Real Estate - Development
    'MRDIYT.BK': 'COMM',  # Specialty Retail
    'NCH.BK': 'PROP',  # Real Estate - Development
    'NCP.BK': 'INDUS',  # Specialty Business Services
    'NEP.BK': 'PKG',  # Packaging & Containers
    'NOBLE.BK': 'PROP',  # Real Estate - Development
    'NYT.BK': 'TRANS',  # Marine Shipping
    'OCC.BK': 'COMM',  # Household & Personal Products
    'OGC.BK': 'COMM',  # Furnishings, Fixtures & Appliances
    'PANEL.BK': 'CONMAT',  # Engineering & Construction
    'PB.BK': 'FOOD',  # Packaged Foods
    'PEER.BK': 'INDUS',  # Specialty Business Services
    'PERM.BK': 'STEEL',  # Steel
    'PG.BK': 'COMM',  # Apparel Manufacturing
    'PLE.BK': 'CONMAT',  # Engineering & Construction
    'PPP.BK': 'INDUS',  # Pollution & Treatment Controls
    'PPPM.BK': 'FOOD',  # Packaged Foods
    'PQS.BK': 'AGRI',  # Farm Products
    'PRAKIT.BK': 'MEDIA',  # Advertising Agencies
    'PRECHA.BK': 'PROP',  # Real Estate - Development
    'PRIME.BK': 'ENERG',  # Utilities - Renewable
    'PROS.BK': 'CONMAT',  # Engineering & Construction
    'PSH.BK': 'PROP',  # Real Estate - Development
    'PTC.BK': 'PETRO',  # Oil & Gas Refining & Marketing
    'PTG.BK': 'COMM',  # Specialty Retail
    'QDC.BK': 'FOOD',  # Restaurants
    'QH.BK': 'PROP',  # Real Estate - Development
    'QLT.BK': 'INDUS',  # Specialty Business Services
    'QTCG.BK': 'CONMAT',  # Engineering & Construction
    'RAM.BK': 'HELTH',  # Medical Care Facilities
    'RICHY.BK': 'PROP',  # Real Estate - Development
    'ROCK.BK': 'COMM',  # Furnishings, Fixtures & Appliances
    'ROJNA.BK': 'ENERG',  # Utilities - Independent Power Producers
    'RP.BK': 'TRANS',  # Marine Shipping
    'RWI.BK': 'STEEL',  # Steel
    'SABINA.BK': 'COMM',  # Apparel Manufacturing
    'SAMART.BK': 'ICT',  # Information Technology Services
    'SAMCO.BK': 'PROP',  # Real Estate Services
    'SAWANG.BK': 'COMM',  # Luxury Goods
    'SCCC.BK': 'CONMAT',  # Building Materials
    'SCG.BK': 'ENERG',  # Utilities - Regulated Electric
    'SE.BK': 'INDUS',  # Industrial Distribution
    'SEAFCO.BK': 'CONMAT',  # Engineering & Construction
    'SENA.BK': 'PROP',  # Real Estate - Development
    'SIRI.BK': 'PROP',  # Real Estate - Development
    'SIS.BK': 'ICT',  # Electronics & Computer Distribution
    'SISB.BK': 'COMM',  # Education & Training Services
    'SJWD.BK': 'TRANS',  # Integrated Freight & Logistics
    'SK.BK': 'CONMAT',  # Building Materials
    'SKE.BK': 'ENERG',  # Utilities - Diversified
    'SNC.BK': 'ETRON',  # Electronic Components
    'SPCG.BK': 'ENERG',  # Utilities - Renewable
    'SPI.BK': 'PROP',  # Real Estate Services
    'SPREME.BK': 'ICT',  # Electronics & Computer Distribution
    'SR.BK': 'INDUS',  # Conglomerates
    'STX.BK': 'CONMAT',  # Building Materials
    'SUSCO.BK': 'PETRO',  # Oil & Gas Refining & Marketing
    'SVI.BK': 'ETRON',  # Electronic Components
    'SVOA.BK': 'ICT',  # Electronics & Computer Distribution
    'SVR.BK': 'PROP',  # Real Estate - Development
    'SYNTEC.BK': 'CONMAT',  # Engineering & Construction
    'TASCO.BK': 'CONMAT',  # Building Materials
    'TC.BK': 'FOOD',  # Packaged Foods
    'TCJ.BK': 'INDUS',  # Farm & Heavy Construction Machinery
    'TCMC.BK': 'COMM',  # Textile Manufacturing
    'TEAM.BK': 'ETRON',  # Electronic Components
    'TERA.BK': 'ICT',  # Electronics & Computer Distribution
    'TFMAMA.BK': 'FOOD',  # Packaged Foods
    'TGPRO.BK': 'STEEL',  # Steel
    'THAI.BK': 'TRANS',  # Airlines
    'TIGER.BK': 'CONMAT',  # Engineering & Construction
    'TKS.BK': 'INDUS',  # Specialty Business Services
    'TKT.BK': 'AUTO',  # Auto Parts
    'TM.BK': 'HELTH',  # Medical Distribution
    'TOA.BK': 'PETRO',  # Specialty Chemicals
    'TPA.BK': 'PETRO',  # Specialty Chemicals
    'TPCS.BK': 'COMM',  # Textile Manufacturing
    'TPLAS.BK': 'PKG',  # Packaging & Containers
    'TPOLY.BK': 'CONMAT',  # Engineering & Construction
    'TRUBB.BK': 'PETRO',  # Specialty Chemicals
    'TSI.BK': 'FIN',  # Insurance - Property & Casualty
    'TSR.BK': 'INDUS',  # unknown
    'TTA.BK': 'INDUS',  # Conglomerates
    'TTCL.BK': 'CONMAT',  # Engineering & Construction
    'TTW.BK': 'ENERG',  # Utilities - Regulated Water
    'TU-PF.BK': 'INDUS',  # unknown
    'TVDH.BK': 'COMM',  # Internet Retail
    'TVO.BK': 'FOOD',  # Packaged Foods
    'TWP.BK': 'STEEL',  # Steel
    'TWZ.BK': 'COMM',  # Specialty Retail
    'UNIQ.BK': 'CONMAT',  # Engineering & Construction
    'UPF.BK': 'COMM',  # Textile Manufacturing
    # 'VAYU1.BK' removed (1-bar history)
    'VS.BK': 'INDUS',  # Specialty Business Services
    'WIN.BK': 'PROP',  # Real Estate Services
    'WPH.BK': 'HELTH',  # Medical Care Facilities
    'WSOL.BK': 'INDUS',  # Business Equipment & Supplies
    'YGG.BK': 'ICT',  # Entertainment
}
