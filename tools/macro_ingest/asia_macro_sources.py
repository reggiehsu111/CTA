"""
Source registry for the Taiwan / Japan macro ingest (QNT-10).

Every entry records WHERE the series comes from, WHAT the units are, and — the
part that actually matters for backtesting — HOW LATE it is publicly known
relative to its reference month (`pub_lag_days`).

`pub_lag_days` is a CONSERVATIVE, documented calendar-day lag from the FIRST DAY
of the reference month to the day the value is first public. It is deliberately
rounded LATE. It is not a scraped release date: none of these publishers expose a
machine-readable release calendar, so a per-observation `release_date` column
would be fabricated precision. Use the lag, and say so in any write-up.

Publication conventions (verified against each agency's release page, 2026-08-31):
  TW CPI          主計總處   ~5th of the following month          -> 36d
  TW unemployment 主計總處   ~22nd-23rd of the following month    -> 54d
  TW M1B/M2       中央銀行   ~25th of the following month         -> 56d
  TW PMI          中經院     1st business day of following month  -> 33d
  TW cycle/燈號   國發會     ~27th of the following month         -> 58d
  JP CPI          総務省統計局 ~19th-26th of following month       -> 57d
  JP unemployment 総務省統計局 ~end of following month             -> 61d
  JP BoJ assets   日本銀行   ~10th of following month             -> 41d
  JP 10y JGB      OECD/MoF   monthly avg, ~mid following month    -> 46d
Daily market series (Nikkei 225, USD/JPY) are same-day and carry lag 0.
"""

# ── Taiwan ──────────────────────────────────────────────────────────────────
TW_SOURCES = {
    "cpi": dict(
        dataset="data.gov.tw/6019 — 消費者物價基本分類指數 (行政院主計總處)",
        url="https://ws.dgbas.gov.tw/001/Upload/461/relfile/11525/230555/pr0101a1m.xml",
        fmt="xml", freq="M", pub_lag_days=36,
        units="index, 民國110年(2021)=100; *_yoy in percent",
    ),
    "unemployment": dict(
        dataset="data.gov.tw/6637 — 人力資源調查失業率 (行政院主計總處)",
        url="https://ws.dgbas.gov.tw/001/Upload/461/relfile/11525/230038/mp0101a07.xml",
        fmt="xml", freq="M", pub_lag_days=54,
        units="percent, not seasonally adjusted",
    ),
    "money": dict(
        dataset="data.gov.tw/6024 — 貨幣總計數 (中央銀行經研處, EF15M01)",
        url="https://www.cbc.gov.tw/public/data/OpenData/%E7%B6%93%E7%A0%94%E8%99%95/EF15M01.csv",
        fmt="csv", freq="M", pub_lag_days=56,
        units="M1A/M1B/M2 in 百萬元 (millions TWD), daily-average basis; *_yoy in percent",
    ),
    "pmi": dict(
        dataset="data.gov.tw/6100 — 臺灣採購經理人指數 (國發會 / 中華經濟研究院)",
        url="https://ws.ndc.gov.tw/Download.ashx?u=LzAwMS9hZG1pbmlzdHJhdG9yLzEwL3JlbGZpbGUvNTc4MS82MzkxL2JmOGE0ZWI3LTEwZmUtNGZhMC1iNjQ2LTMwZTg5MGQwMjE4YS5jc3Y%3d&n=6Ie654Gj5o6h6LO857aT55CG5Lq65oyH5pW4KHBtaeWPim5taSkuY3N2&icon=.csv",
        fmt="csv", freq="M", pub_lag_days=33,
        units="diffusion index, 50 = no change",
    ),
    "cycle": dict(
        dataset="data.gov.tw/6099 — 景氣指標及燈號 (國家發展委員會)",
        url="https://ws.ndc.gov.tw/Download.ashx?u=LzAwMS9hZG1pbmlzdHJhdG9yLzEwL3JlbGZpbGUvNTc4MS82MzkyL2VhMjM1YmQ5LWQwNTItNGE2OS1hYmZjLWQ1Yzc4NWQzZDBlMi56aXA%3d&n=5pmv5rCj5oyH5qiZ5Y%2bK54eI6JmfLnppcA%3d%3d&icon=.zip",
        fmt="zip", freq="M", pub_lag_days=58,
        units="composite indices; monitor_score 9-45; monitor_signal one of 藍/黃藍/綠/黃紅/紅",
    ),
}

# ws.dgbas.gov.tw and www.cbc.gov.tw serve an INCOMPLETE TLS chain (the leaf is
# issued by 'TWCA Secure SSL Certification Authority' but the presented chain is
# ePKI's).  The TWCA Global Root IS in the system store, so supplying just the
# missing intermediate restores full verification — we never disable it.
TW_CA_BUNDLE = "/home/ubuntu/mtx/tools/certs/tw_gov_bundle.pem"
TWCA_INTERMEDIATE_URL = "http://sslserver.twca.com.tw/cacert/secure_sha2_2023G3.crt"

# ── Japan ───────────────────────────────────────────────────────────────────
# NOTE: FRED's Japan CPI / industrial-production series are OECD-MEI vintages
# that were DISCONTINUED — JPNCPIALLMINMEI stops 2021-06, JPNPROINDMISMEI stops
# 2024-03, and TWN* series 404 outright.  They still return HTTP 200 and parse
# cleanly, which is exactly the "present but not complete" trap.  Japan CPI is
# therefore taken from the OECD SDMX COICOP-2018 dataflow, which is live.
JP_OECD_CPI = dict(
    dataset="OECD SDMX — DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL",
    base="https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL",
    key="JPN.M......",          # REF_AREA.FREQ.METHODOLOGY.MEASURE.UNIT_MEASURE.EXPENDITURE.ADJUSTMENT.TRANSFORMATION
    freq="M", pub_lag_days=57,
    # EXPENDITURE code -> column.  _T = all items, _TXCP01_NRG = ex food & energy
    expenditures={"_T": "cpi", "_TXCP01_NRG": "cpi_core",
                  "CP045_0722": "cpi_energy", "CP01": "cpi_food"},
    units="GY = year-on-year percent; IX = index level",
)

# Live FRED series only — each verified to have a 2026 observation.
JP_FRED_MONTHLY = {
    "unemployment_rate": dict(id="LRHUTTTTJPM156S", pub_lag_days=61, units="percent, SA"),
    "boj_assets":        dict(id="JPNASSETS",       pub_lag_days=41, units="100 million JPY"),
    "jgb10y":            dict(id="IRLTLT01JPM156N", pub_lag_days=46, units="percent per annum"),
}
JP_FRED_DAILY = {
    "nikkei225": dict(id="NIKKEI225", pub_lag_days=0, units="index points, close"),
    "usdjpy":    dict(id="DEXJPUS",   pub_lag_days=0, units="JPY per USD, NY noon"),
}

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
FRED_UA  = "python-requests/2.31.0"   # browser UAs get throttled — see us_macro_ingest
