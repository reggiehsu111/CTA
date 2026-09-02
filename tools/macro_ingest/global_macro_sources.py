"""
Source registry for the global / regional macro ingest (QNT-10, part 2).

Part 1 of this ticket landed Taiwan and Japan in the wide tables
`tw_macro_monthly`, `jp_macro_monthly`, `jp_markets_daily` (see
`asia_macro_sources.py`).  Part 2 covers the sources that were still missing:
Korea, China, the semiconductor / tech cycle, and the US and global series that
the pre-existing `us_*` tables do not carry.

WHY A TIDY TABLE AND NOT MORE WIDE TABLES
-----------------------------------------
The existing macro tables are wide (one column per series).  Adding a series to
a wide table requires ALTER TABLE, which is a banned operation for agents on
this box, so every future macro source would need a human migration.  Part 2
therefore lands in a tidy pair instead:

    macro_series_meta   one row per series: units, source, frequency, pub lag
    macro_series        (series_id, date, value)

New sources are then an INSERT, never a migration.  Nothing in the existing
wide tables is read, written, or altered by this ingest.

PUBLICATION LAG
---------------
`pub_lag_days` is a CONSERVATIVE, documented calendar-day lag from the FIRST DAY
of the reference period to the day the value is first public.  It is rounded
LATE on purpose.  It is NOT a scraped release date -- these publishers do not
expose a machine-readable release calendar, so a per-observation `release_date`
would be fabricated precision.  Use the lag, and state the assumption in any
write-up (see the point-in-time rule in the standing brief).

Daily market series (FX, yields, breakevens, spot commodities) are same-day
observations published that evening, so they carry lag 0 -- but note they are
US-close series, which for a 台指期 backtest lands them on the NEXT Taipei
session.  That is a shift decision for the signal, not a property of the data.

REJECTED SOURCES -- do not re-probe these, see REJECTED below.
"""

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd=1900-01-01"
FRED_UA  = "python-requests/2.31.0"   # browser UAs get throttled

# ---------------------------------------------------------------------------
# Every series below was probed on 2026-09-01 and confirmed to carry a 2026
# observation.  fmt: (fred_id, label, country, category, freq, units, pub_lag_days)
# ---------------------------------------------------------------------------
FRED_SERIES = {
# --- Asia FX (daily, US-close) ------------------------------------------------
 "twd_usd":            ("DEXTAUS",            "TWD per USD, NY close",                 "TW","fx",      "D","TWD per USD",                      0),
 "krw_usd":            ("DEXKOUS",            "KRW per USD, NY close",                 "KR","fx",      "D","KRW per USD",                      0),
 "cny_usd":            ("DEXCHUS",            "CNY per USD, NY close",                 "CN","fx",      "D","CNY per USD",                      0),
# --- Korea (the Asian tech cycle read) ---------------------------------------
 # UNITS TRAP, verified empirically 2026-09-01: the OECD *_664S series are in
 # NATIONAL CURRENCY and *_667S is in USD. kr_exports vs kr_exports_sa therefore
 # differ on TWO axes -- currency AND seasonal adjustment -- and their levels are
 # ~1500x apart. corr(kr_exports/kr_exports_sa, KRW-per-USD) = 0.9999. Never
 # compare or ratio them without converting; use yoy/log-diff instead.
 "kr_exports":         ("XTEXVA01KRM664S",    "Korea exports, value (KRW, NSA)",       "KR","trade",   "M","level, KRW, NSA",                 60),
 "kr_exports_sa":      ("XTEXVA01KRM667S",    "Korea exports, value (USD, SA) - NOT level-comparable to kr_exports","KR","trade","M","level, USD, SA", 60),
 "kr_imports":         ("XTIMVA01KRM664S",    "Korea imports, value (KRW, NSA)",       "KR","trade",   "M","level, KRW, NSA",                 60),
 "kr_unemployment":    ("LRHUTTTTKRM156S",    "Korea unemployment rate",               "KR","labour",  "M","percent, SA",                     45),
 "kr_kospi":           ("SPASTT01KRM661N",    "Korea KOSPI, monthly average",          "KR","equity",  "M","index, 2015=100",                 31),
# --- China -------------------------------------------------------------------
 "cn_exports":         ("XTEXVA01CNM664S",    "China exports, value (CNY, NSA)",       "CN","trade",   "M","level, CNY, NSA",                 60),
 "cn_imports":         ("XTIMVA01CNM664S",    "China imports, value (CNY, NSA)",       "CN","trade",   "M","level, CNY, NSA",                 60),
 "cn_leading_idx":     ("CHNLOLITOAASTSAM",   "China OECD composite leading indicator", "CN","cycle",   "M","amplitude-adjusted, 100=trend",   45),
 "cn_shanghai_comp":   ("SPASTT01CNM661N",    "China share prices, monthly average",   "CN","equity",  "M","index, 2015=100",                 31),
# --- Japan (extends jp_macro_monthly, which has CPI/unemp/BoJ/JGB) ------------
 "jp_exports":         ("XTEXVA01JPM664S",    "Japan exports, value (JPY, NSA)",       "JP","trade",   "M","level, JPY, NSA",                 60),
 "jp_imports":         ("XTIMVA01JPM664S",    "Japan imports, value (JPY, NSA)",       "JP","trade",   "M","level, JPY, NSA",                 60),
 "jp_policy_rate":     ("IRSTCI01JPM156N",    "Japan call money / interbank rate",     "JP","rates",   "M","percent per annum",               31),
# --- Semiconductor / tech cycle (the Taiwan-relevant real economy) ------------
 "us_semi_ip":         ("IPG3344S",           "US semiconductor & electronic component IP, SA","US","semis","M","index, 2017=100, SA",         46),
 "us_semi_ip_nsa":     ("IPG3344N",           "US semiconductor & electronic component IP, NSA","US","semis","M","index, 2017=100, NSA",       46),
 "us_semi_ppi":        ("PCU334413334413",    "US PPI: semiconductor manufacturing",   "US","semis",   "M","index",                           45),
 "us_electronics_ppi": ("WPU1178",            "US PPI: electronic components & accessories","US","semis","M","index, 1982=100",               45),
 "us_mfg_new_orders":  ("AMTMNO",             "US manufacturers' new orders, total",   "US","activity","M","millions USD, SA",                65),
 "us_retail_inv_sales":("RETAILIRSA",         "US retail inventories/sales ratio",     "US","activity","M","ratio, SA",                       75),
 "us_freight_tsi":     ("TSIFRGHT",           "US freight transportation services index","US","activity","M","index, 2000=100, SA",           90),
# --- US macro not carried by the existing us_* tables ------------------------
 "us_cfnai":           ("CFNAI",              "Chicago Fed national activity index",   "US","cycle",   "M","std dev from trend",              56),
 "us_m2":              ("M2SL",               "US M2 money stock",                     "US","money",   "M","billions USD, SA",                60),
 "us_consumer_credit": ("TOTALSL",            "US total consumer credit outstanding",  "US","money",   "M","billions USD, SA",                66),
 "us_philly_fed":      ("GACDFSA066MSFRBPHI", "Philadelphia Fed mfg survey, general activity","US","survey","M","diffusion index, SA",        21),
 "us_empire_state":    ("GACDISA066MSFRBNY",  "NY Fed Empire State mfg survey, general activity","US","survey","M","diffusion index, SA",     16),
 "us_recession_prob":  ("RECPROUSM156N",      "US smoothed recession probability",     "US","cycle",   "M","percent",                         75),
 "us_corp_profits":    ("CP",                 "US corporate profits after tax",        "US","activity","Q","billions USD, SAAR",             150),
# --- US financial conditions / rates not in us_rates_daily or us_risk_daily --
 "us_breakeven_10y":   ("T10YIE",             "US 10y breakeven inflation",            "US","rates",   "D","percent",                          0),
 "us_breakeven_5y5y":  ("T5YIFR",             "US 5y5y forward inflation expectation", "US","rates",   "D","percent",                          0),
 "us_real_10y":        ("DFII10",             "US 10y TIPS real yield",                "US","rates",   "D","percent",                          0),
 "us_term_premium_10y":("THREEFYTP10",        "US 10y ACM term premium",               "US","rates",   "D","percent",                          0),
 "us_dgs5":            ("DGS5",               "US 5y Treasury constant maturity",      "US","rates",   "D","percent",                          0),
 "us_dgs30":           ("DGS30",              "US 30y Treasury constant maturity",     "US","rates",   "D","percent",                          0),
 "us_iorb":            ("IORB",               "US interest on reserve balances",       "US","rates",   "D","percent",                          0),
 "us_dxy_broad":       ("DTWEXBGS",           "US dollar index, broad goods & services","US","fx",     "D","index, Jan2006=100",               0),
 "us_stlfsi":          ("STLFSI4",            "St. Louis Fed financial stress index",  "US","risk",    "W","index, 0 = normal",                7),
# --- Global / commodities ----------------------------------------------------
 "wti":                ("DCOILWTICO",         "WTI crude oil spot, Cushing",           "GL","commodity","D","USD per barrel",                  0),
 "copper":             ("PCOPPUSDM",          "Copper, global price",                  "GL","commodity","M","USD per tonne",                  35),
 # IQ12260 is an INDEX, not a USD/oz price -- last obs 154.8, which is not a
 # gold price in any currency. FRED's USD/oz series (PGOLDUSDM, GOLDAMGBD228NLBM)
 # both 404 on this mirror, so an index is the best available and it is labelled
 # as one. Use it for returns/momentum, never as a level.
 "gold_index":         ("IQ12260",            "Gold price index (NOT USD/oz)",         "GL","commodity","M","index, level is not a price",     35),
 "igrea":              ("IGREA",              "Index of global real economic activity (Kilian)","GL","cycle","M","percent deviation from trend",40),
 "epu_global":         ("GEPUCURRENT",        "Global economic policy uncertainty index","GL","risk",  "M","index, current-price GDP weights",31),
}

# ---------------------------------------------------------------------------
# REJECTED -- probed 2026-09-01, deliberately NOT ingested.  Recorded so the
# next session does not spend the same hour rediscovering them.
# ---------------------------------------------------------------------------
REJECTED = {
    # FRED's OECD-MEI mirrors were discontinued.  They return HTTP 200 and parse
    # cleanly -- the "present but not complete" trap from the standing brief.
    "KORPROINDMISMEI":  "Korea IP -- OECD MEI, frozen at 2024-03",
    "CPALTT01KRM659N":  "Korea CPI yoy -- OECD MEI, frozen at 2023-11",
    "CPALTT01CNM659N":  "China CPI yoy -- OECD MEI, frozen at 2025-04",
    "JPNPROINDMISMEI":  "Japan IP -- OECD MEI, frozen at 2024-03",
    "JPNPROINDAISMEI":  "Japan IP annual -- frozen at 2023-01",
    "JPNCPIALLMINMEI":  "Japan CPI -- frozen at 2021-06 (part 1 uses OECD SDMX instead)",
    "BSCICP03JPM665S":  "Japan business confidence (Tankan proxy) -- frozen at 2023-12",
    "JPNSARTMISMEI":    "Japan retail -- frozen at 2024-01",
    "MYAGM2CNM189N":    "China M2 -- frozen at 2019-08",
    "MABMM301CNM189S":  "China M2 alt -- frozen at 2018-12",
    "OECDLOLITOAASTSAM":"OECD total CLI -- frozen at 2022-11",
    "EA19LOLITOAASTSAM":"Euro area CLI -- frozen at 2022-11",
    "USSLIND":          "US leading index -- frozen at 2020-02",
    "TEDRATE":          "TED spread -- discontinued 2022-01",
    # 404 on FRED entirely
    "NAPM":             "ISM manufacturing PMI -- removed from FRED (ISM is proprietary)",
    "XTEXVA01TWM664S":  "Taiwan exports -- no FRED series; Taiwan is not an OECD member. "
                        "tw_macro_monthly.customs_exports covers this from 主計總處.",
    "TWNCPIALLMINMEI":  "Taiwan CPI -- no FRED series; tw_macro_monthly.cpi covers it.",
    "JPNMACHORD":       "Japan machinery orders -- no FRED series",
    # Reachable but gated
    "NBS_CHINA_PMI":    "data.stats.gov.cn easyquery returns HTTP 403 to this box. "
                        "Official NBS manufacturing PMI is the single biggest remaining "
                        "gap; it needs either a proxy or a key-gated mirror.",
    "KR_CUSTOMS_20DAY": "Korea Customs 20-day exports (관세청) -- portal reachable but the "
                        "data API requires a data.go.kr service key. This is the highest-value "
                        "missing series for Taiwan tech: 3x/month at ~1 day lag vs the 60-day "
                        "lag on the monthly OECD mirror we do ingest.",
    "BOK_ECOS":         "Bank of Korea ECOS API requires a registered key",
    "MOEA_E01":         "data.moea.gov.tw refuses connections from this box; 外銷訂單 value "
                        "is therefore still missing (only the DCI diffusion index is in "
                        "tw_macro_monthly.export_orders_dci).",
}
