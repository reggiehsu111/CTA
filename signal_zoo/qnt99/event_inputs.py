"""QNT-99 Part A — build MACRO EVENT inputs: the release calendar + the value
released at each event.

Every prior MTX macro grid (QNT-12 / -14 / -18 / -19 / -98) used a macro series
as a forward-filled LEVEL. Nothing has ever used
  (a) the surprise IN the release, or
  (b) the release CALENDAR itself as timing.
Those are the two new source axes here.

PIT
---
Each reference_date is mapped to the TW trading day it becomes actable through
`cta.us_macro._available_from_tw`, i.e. the first TAIFEX 08:45 day-session open
STRICTLY AFTER the exact ALFRED release_ts. So input[t] means "public before t's
08:45 open". The production pipeline only computes at 15:31 TPE
(project_mtx_pit_compute_schedule), so the harness's own
_SHIFT = {c2c:2, o2o:2, day:1, ongap:1} is the correct execution lag on top of
that labelling -- it is used unchanged.

KNOWN CONTAMINATION, stated up front: the DB stores FINAL revised values, not
first prints (only us_nfci has vintages). A surprise built from revised data
knows a little of the future. It biases the surprise TOWARD working, so a null
result here is safe and a positive one would need vintage data to confirm.
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta.us_macro as um
from db_utils import engine

# event -> (table, [fields]), and how to difference the level
EVENTS = {
    "CPI":                   ("us_prices",           ["cpi_all_sa", "cpi_core_sa"],            "pct"),
    "PCE":                   ("us_prices",           ["pce_core_sa"],                          "pct"),
    "PPI":                   ("us_prices",           ["ppi_final_demand_sa"],                  "pct"),
    "NFP":                   ("us_labor_monthly",    ["nfp_thousands", "unemployment_rate"],   "diff"),
    "JOLTS":                 ("us_labor_monthly",    ["jolts_openings_thousands"],             "diff"),
    "claims":                ("us_labor_weekly",     ["initial_claims"],                       "diff"),
    "retail_sales":          ("us_activity_monthly", ["retail_sales_sa"],                      "pct"),
    "industrial_production": ("us_activity_monthly", ["industrial_production"],                "pct"),
    "durable_goods":         ("us_activity_monthly", ["durable_goods_orders"],                 "pct"),
    "housing_starts":        ("us_activity_monthly", ["housing_starts"],                       "pct"),
    "new_home_sales":        ("us_activity_monthly", ["new_home_sales"],                       "pct"),
    "trade_balance":         ("us_activity_monthly", ["trade_balance"],                        "diff"),
    "umich_sentiment":       ("us_activity_monthly", ["umich_sentiment"],                      "diff"),
    "case_shiller":          ("us_activity_monthly", ["case_shiller_hpi"],                     "pct"),
    "GDP_advance":           ("us_gdp_quarterly",    ["gdp_real_growth"],                      "diff"),
}
# nfp_thousands is a LEVEL of payrolls in the DB? -> checked in build(): if it looks
# like a level (>100000) we diff it, else it is already a monthly change.

_TBL_CACHE = {}
def _tbl(t):
    if t not in _TBL_CACHE:
        _TBL_CACHE[t] = pd.read_sql(f"select * from {t} order by date", engine,
                                    parse_dates=["date"]).set_index("date")
    return _TBL_CACHE[t]


def release_calendar() -> pd.DataFrame:
    """release_name, reference_date, release_ts — the raw event table."""
    df = pd.read_sql("select release_name, reference_date, release_ts from us_macro_releases "
                     "where release_ts is not null", engine)
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["release_ts"] = pd.to_datetime(df["release_ts"], utc=True)
    return df


def event_tw_dates(event: str, trading_index) -> pd.Series:
    """reference_date -> the TW trading date on whose 08:45 open the event is public."""
    cal = um._load_release_calendar().get(event)
    if cal is None or not len(cal):
        return pd.Series(dtype="datetime64[ns]")
    return um._tw_actable(cal, pd.DatetimeIndex(trading_index)).dropna()


def surprise_series(event: str, field: str, kind: str, k: int) -> pd.Series:
    """Surprise at each reference_date: the release-over-release change of the
    published value, standardised by the trailing k changes THAT WERE ALREADY
    PUBLISHED. Uses only reference_dates strictly earlier than the current one."""
    tbl, _, _ = EVENTS[event]
    s = _tbl(tbl)[field].astype(float).dropna()
    if kind == "pct":
        chg = s.pct_change()
    else:
        chg = s.diff()
    mu = chg.shift(1).rolling(k, min_periods=max(6, k // 3)).mean()
    sd = chg.shift(1).rolling(k, min_periods=max(6, k // 3)).std()
    z = ((chg - mu) / sd.replace(0, np.nan))
    return z.dropna()


def impulse(z_by_ref: pd.Series, ev_dates: pd.Series, trading_index,
            hold: int) -> pd.Series:
    """Map a per-reference_date surprise onto the TW calendar and hold it for
    `hold` trading days starting at the actable date. Zero elsewhere (flat)."""
    ti = pd.DatetimeIndex(trading_index)
    out = pd.Series(0.0, index=ti)
    common = z_by_ref.index.intersection(ev_dates.index)
    if not len(common):
        return out
    loc = {d: i for i, d in enumerate(ti)}
    for ref in common:
        d = ev_dates.loc[ref]
        i = loc.get(pd.Timestamp(d))
        if i is None:
            continue
        out.iloc[i:i + hold] = float(z_by_ref.loc[ref])
    return out
