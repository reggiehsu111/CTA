"""
US macro loaders — CPI/PCE (monthly) and Fed Funds / Treasury yields (daily).

Source
------
FRED (fred.stlouisfed.org) → Lambda ingest → RDS. See:
  - QuantResearch/tools/lambda/us_prices_ingest      (CPI + PCE)
  - QuantResearch/tools/lambda/us_rates_daily_ingest (daily rates)

Both use ``python-requests/2.31.0`` UA (see [[fred-ua-quirk]] memory) and
mirror the NFCI pattern (chain-invoke mtx-signal-runner on new upserts).

PIT semantics
-------------
Monthly series (CPI/PCE) are dated at the FIRST DAY of the reference month
but released weeks later:
  - CPI headline / core: BLS releases ~13-15th of the FOLLOWING month at 08:30 ET
  - PCE:                 BEA releases ~28-30th of the FOLLOWING month at 08:30 ET

For PIT alignment onto a TW trading calendar:
  - The value dated ``YYYY-MM-01`` should NOT be visible until the actual
    release date. This loader can't verify the exact release date
    per-observation (would need BLS/BEA release calendars), so it uses a
    conservative default: **35 days after the reference-month date** for CPI
    and **45 days** for PCE. Override via ``pit_lag_days``.

Daily rates (Fed funds, Treasury yields):
  - Published next TW business day at 08:00-09:00 ET (= 20:00-21:00 TPE)
  - Standard shift(1) at TW-market close is safe.

Usage
-----
    >>> import cta
    >>> ASSET = cta.load_asset('mtx', '1d')
    >>>
    >>> # Monthly (raw, month-start dated)
    >>> cpi = cta.load_us_price('cpi_all_sa')
    >>>
    >>> # TW-aligned + forward-filled + PIT-safe
    >>> cpi_tw = cta.load_us_price_tw('cpi_all_sa', ASSET.index)
    >>>
    >>> # YoY convenience (12-month pct-change on the raw monthly series,
    >>> # then aligned to TW calendar PIT-safe)
    >>> cpi_yoy = cta.load_us_price_yoy_tw('cpi_all_sa', ASSET.index)
    >>>
    >>> # Daily rates
    >>> ffr = cta.load_us_rate_tw('fed_funds_eff', ASSET.index)
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pytz

_TPE = pytz.timezone("Asia/Taipei")
_TAIFEX_OPEN = "08:45"   # day-session open; a US release is actable on the first open strictly after it


# ── Column catalogs ─────────────────────────────────────────────────────────
_PRICE_FIELDS = (
    "cpi_all_sa", "cpi_core_sa",
    "cpi_all_nsa", "cpi_core_nsa",
    "cpi_energy_sa", "cpi_food_sa", "cpi_shelter_sa",
    "pce_all_sa", "pce_core_sa",
)
_RATE_FIELDS = (
    "fed_funds_eff", "sofr",
    "dgs10", "dgs2", "t10y2y", "dgs3mo",
)
_LABOR_MONTHLY_FIELDS = (
    "unemployment_rate", "nfp_thousands",
    "avg_hourly_earnings", "labor_force_pr",
    "avg_weekly_hours",   "jolts_openings_thousands",
)
_LABOR_WEEKLY_FIELDS = ("initial_claims", "continuing_claims")
_RISK_DAILY_FIELDS = (
    "vix", "vix_3m", "hy_oas", "ig_oas",
    "baa_10y_spread", "aaa_10y_spread",
)


def available_us_price_fields() -> list[str]:
    return list(_PRICE_FIELDS)


def available_us_rate_fields() -> list[str]:
    return list(_RATE_FIELDS)


def available_us_labor_monthly_fields() -> list[str]:
    return list(_LABOR_MONTHLY_FIELDS)


def available_us_labor_weekly_fields() -> list[str]:
    return list(_LABOR_WEEKLY_FIELDS)


def available_us_risk_fields() -> list[str]:
    return list(_RISK_DAILY_FIELDS)


# ── DB helpers ──────────────────────────────────────────────────────────────
_PRICE_CACHE: pd.DataFrame | None = None
_RATE_CACHE:  pd.DataFrame | None = None
_LABOR_M_CACHE: pd.DataFrame | None = None
_LABOR_W_CACHE: pd.DataFrame | None = None
_RISK_CACHE:  pd.DataFrame | None = None


def _engine():
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path:
        sys.path.insert(0, _LIBS)
    from db_utils import engine
    return engine


def _load_us_prices_all() -> pd.DataFrame:
    """Whole us_prices table, cached."""
    global _PRICE_CACHE
    if _PRICE_CACHE is None:
        df = pd.read_sql(
            "SELECT date, " + ", ".join(_PRICE_FIELDS) +
            " FROM us_prices ORDER BY date",
            _engine(),
        )
        df["date"] = pd.to_datetime(df["date"])
        for c in _PRICE_FIELDS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _PRICE_CACHE = df.set_index("date").sort_index()
    return _PRICE_CACHE


def _load_us_rates_all() -> pd.DataFrame:
    global _RATE_CACHE
    if _RATE_CACHE is None:
        df = pd.read_sql(
            "SELECT date, " + ", ".join(_RATE_FIELDS) +
            " FROM us_rates_daily ORDER BY date",
            _engine(),
        )
        df["date"] = pd.to_datetime(df["date"])
        for c in _RATE_FIELDS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _RATE_CACHE = df.set_index("date").sort_index()
    return _RATE_CACHE


def _load_generic(table: str, fields: tuple, cache_name: str) -> pd.DataFrame:
    """Whole-table cache loader for the labor/risk tables."""
    globals_ = globals()
    if globals_[cache_name] is None:
        df = pd.read_sql(
            "SELECT date, " + ", ".join(fields) + f" FROM {table} ORDER BY date",
            _engine(),
        )
        df["date"] = pd.to_datetime(df["date"])
        for c in fields:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        globals_[cache_name] = df.set_index("date").sort_index()
    return globals_[cache_name]


def _load_us_labor_monthly_all() -> pd.DataFrame:
    return _load_generic("us_labor_monthly", _LABOR_MONTHLY_FIELDS, "_LABOR_M_CACHE")


def _load_us_labor_weekly_all() -> pd.DataFrame:
    return _load_generic("us_labor_weekly", _LABOR_WEEKLY_FIELDS, "_LABOR_W_CACHE")


def _load_us_risk_all() -> pd.DataFrame:
    return _load_generic("us_risk_daily", _RISK_DAILY_FIELDS, "_RISK_CACHE")


# ── Public API ──────────────────────────────────────────────────────────────
def load_us_price(field: str) -> pd.Series:
    """Raw monthly Series, indexed by 1st-of-month date."""
    if field not in _PRICE_FIELDS:
        raise ValueError(f"unknown price field {field!r}; valid: {_PRICE_FIELDS}")
    return _load_us_prices_all()[field].dropna().rename(field)


def load_us_rate(field: str) -> pd.Series:
    """Raw daily Series, indexed by business-day date."""
    if field not in _RATE_FIELDS:
        raise ValueError(f"unknown rate field {field!r}; valid: {_RATE_FIELDS}")
    return _load_us_rates_all()[field].dropna().rename(field)


def _default_price_pit_lag(field: str) -> int:
    """Conservative PIT lag in days from month-start date to first usable TW date."""
    return 45 if field.startswith("pce_") else 35


# ── PIT release calendar (exact ALFRED release_ts -> TW actable tick) ─────────
# us_macro_releases holds, per (release_name, reference_date), the EXACT first-public timestamp
# (release_ts, tz-aware UTC) backfilled from FRED/ALFRED. A US release is only actable on the first
# TAIFEX day-session open (08:45 TPE) STRICTLY AFTER release_ts — no look-ahead. This replaces the
# old fixed-lag heuristic (which revealed CPI/PCE several days early). Fields missing from the
# calendar (pre-2009 history, un-archived series) fall back to the conservative fixed lag.
_RELEASE_CAL_CACHE: dict | None = None

# data column -> the release EVENT (release_name) whose schedule governs its PIT availability
_FIELD_TO_RELEASE = {
    "cpi_all_sa": "CPI", "cpi_core_sa": "CPI", "cpi_all_nsa": "CPI", "cpi_core_nsa": "CPI",
    "cpi_energy_sa": "CPI", "cpi_food_sa": "CPI", "cpi_shelter_sa": "CPI",
    "pce_all_sa": "PCE", "pce_core_sa": "PCE",
    "ppi_final_demand_sa": "PPI", "ppi_core_sa": "PPI", "ppi_all_commodities": "PPI",
    "nfp_thousands": "NFP", "unemployment_rate": "NFP", "avg_hourly_earnings": "NFP",
    "labor_force_pr": "NFP", "avg_weekly_hours": "NFP", "jolts_openings_thousands": "JOLTS",
    "initial_claims": "claims", "continuing_claims": "claims",
    "retail_sales_sa": "retail_sales", "retail_ex_food_sa": "retail_sales", "retail_control_sa": "retail_sales",
    "industrial_production": "industrial_production", "capacity_util": "industrial_production",
    "durable_goods_orders": "durable_goods", "core_capex_orders": "durable_goods",
    "housing_starts": "housing_starts", "building_permits": "housing_starts",
    "new_home_sales": "new_home_sales", "trade_balance": "trade_balance",
    "umich_sentiment": "umich_sentiment", "case_shiller_hpi": "case_shiller",
    "gdp_nominal": "GDP_advance", "gdp_real": "GDP_advance",
    "gdp_real_growth": "GDP_advance", "gdp_deflator": "GDP_advance",
}


def _load_release_calendar() -> dict:
    """{release_name: DataFrame(reference_date -> release_ts UTC)}, cached."""
    global _RELEASE_CAL_CACHE
    if _RELEASE_CAL_CACHE is None:
        df = pd.read_sql(
            "SELECT release_name, reference_date, release_ts FROM us_macro_releases "
            "WHERE release_ts IS NOT NULL", _engine())
        df["reference_date"] = pd.to_datetime(df["reference_date"]).astype("datetime64[ns]")
        df["release_ts"] = pd.to_datetime(df["release_ts"], utc=True)
        _RELEASE_CAL_CACHE = {n: g.set_index("reference_date")["release_ts"].sort_index()
                              for n, g in df.groupby("release_name")}
    return _RELEASE_CAL_CACHE


def _tw_actable(release_ts: pd.Series, trading_index: pd.DatetimeIndex) -> pd.Series:
    """Vectorized: each release_ts (UTC) -> the first TW trading date whose 08:45 TPE open is
    strictly after it. NaT for releases beyond the TW data."""
    tw = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_index).normalize().unique()))
    open_utc = pd.DatetimeIndex(
        [_TPE.localize(pd.Timestamp(f"{d.date()} {_TAIFEX_OPEN}")).tz_convert("UTC") for d in tw])
    idx = open_utc.searchsorted(pd.DatetimeIndex(release_ts), side="right")
    return pd.Series([tw[i] if i < len(tw) else pd.NaT for i in idx], index=release_ts.index)


def _available_from_tw(reference_dates: pd.DatetimeIndex, field: str,
                       trading_index: pd.DatetimeIndex, fallback_lag_days: int) -> pd.Series:
    """Per reference_date, the TW date the value becomes actable — from the exact release calendar,
    falling back to reference_date + fallback_lag for dates missing from the calendar."""
    ref = pd.DatetimeIndex(pd.to_datetime(reference_dates)).astype("datetime64[ns]")
    release_name = _FIELD_TO_RELEASE.get(field)
    cal = _load_release_calendar().get(release_name) if release_name else None
    avail = pd.Series(pd.NaT, index=ref, dtype="datetime64[ns]")
    in_cal = ref[:0]
    if cal is not None:
        common = ref.intersection(cal.index)
        if len(common):
            avail.loc[common] = _tw_actable(cal.loc[common], trading_index).values
            in_cal = common
    # Fallback ONLY for refs with NO calendar entry (genuinely unknown release, e.g. pre-2009 history).
    # A calendar ref mapping to NaT is a release BEYOND the TW data (future) -> keep NaT so it never
    # maps to a bar; a fixed-lag fallback there could land before the real release (look-ahead).
    no_cal = ~ref.isin(in_cal)
    if no_cal.any():
        avail.loc[ref[no_cal]] = (ref[no_cal] + pd.Timedelta(days=fallback_lag_days)).astype("datetime64[ns]")
    return avail.astype("datetime64[ns]")


def load_us_price_tw(
    field: str,
    trading_index: pd.DatetimeIndex,
    pit_lag_days: int | None = None,
) -> pd.Series:
    """Monthly price index aligned to a TW trading calendar, forward-filled,
    PIT-safe. Default ``pit_lag_days`` = 35 for CPI, 45 for PCE."""
    raw = load_us_price(field)
    lag = pit_lag_days if pit_lag_days is not None else _default_price_pit_lag(field)

    df = pd.DataFrame({
        "value":              raw.values,
        "available_from_tw":  _available_from_tw(raw.index, field, trading_index, lag).values,
    }).dropna(subset=["available_from_tw"]).sort_values("available_from_tw").reset_index(drop=True)

    tw = pd.DataFrame({"tw_date": pd.DatetimeIndex(trading_index).sort_values().astype("datetime64[ns]")})
    merged = pd.merge_asof(
        tw, df[["available_from_tw", "value"]],
        left_on="tw_date", right_on="available_from_tw",
        direction="backward",
    )
    out = pd.Series(merged["value"].values, index=merged["tw_date"],
                    name=f"{field}_tw").sort_index()
    return out.reindex(trading_index)


def load_us_price_yoy_tw(
    field: str,
    trading_index: pd.DatetimeIndex,
    pit_lag_days: int | None = None,
) -> pd.Series:
    """12-month YoY % change of the monthly price index, TW-aligned PIT-safe.

    Computes YoY on the monthly (raw) series first, then merges onto TW.
    Result is in percentage points (e.g., 3.25 means +3.25% YoY).
    """
    raw = load_us_price(field)
    yoy = (raw.pct_change(12) * 100.0).rename(f"{field}_yoy")
    lag = pit_lag_days if pit_lag_days is not None else _default_price_pit_lag(field)

    df = pd.DataFrame({
        "value":             yoy.values,
        "available_from_tw": _available_from_tw(yoy.index, field, trading_index, lag).values,
    }).dropna(subset=["value", "available_from_tw"]).sort_values("available_from_tw").reset_index(drop=True)

    tw = pd.DataFrame({"tw_date": pd.DatetimeIndex(trading_index).sort_values().astype("datetime64[ns]")})
    merged = pd.merge_asof(
        tw, df[["available_from_tw", "value"]],
        left_on="tw_date", right_on="available_from_tw",
        direction="backward",
    )
    out = pd.Series(merged["value"].values, index=merged["tw_date"],
                    name=f"{field}_yoy_tw").sort_index()
    return out.reindex(trading_index)


def load_us_rate_tw(
    field: str,
    trading_index: pd.DatetimeIndex,
    pit_lag_days: int = 1,
) -> pd.Series:
    """Daily US rate aligned to TW calendar, forward-filled.

    Default ``pit_lag_days=1``: FRED publishes ~08:00-09:00 ET the following
    business day, which is ~20:00-21:00 TPE. The value dated business-day D
    is safely usable on TW business day D+1.
    """
    raw = load_us_rate(field)

    df = pd.DataFrame({
        "value":             raw.values,
        "available_from_tw": raw.index + pd.Timedelta(days=pit_lag_days),
    }).sort_values("available_from_tw").reset_index(drop=True)

    tw = pd.DataFrame({"tw_date": pd.DatetimeIndex(trading_index).sort_values().astype("datetime64[ns]")})
    merged = pd.merge_asof(
        tw, df[["available_from_tw", "value"]],
        left_on="tw_date", right_on="available_from_tw",
        direction="backward",
    )
    out = pd.Series(merged["value"].values, index=merged["tw_date"],
                    name=f"{field}_tw").sort_index()
    return out.reindex(trading_index)


# ── Labor monthly ──────────────────────────────────────────────────────────
def load_us_labor_monthly(field: str) -> pd.Series:
    if field not in _LABOR_MONTHLY_FIELDS:
        raise ValueError(f"unknown labor monthly field {field!r}; valid: {_LABOR_MONTHLY_FIELDS}")
    return _load_us_labor_monthly_all()[field].dropna().rename(field)


def load_us_labor_monthly_tw(
    field: str, trading_index: pd.DatetimeIndex, pit_lag_days: int = 35,
) -> pd.Series:
    """NFP/UNRATE etc released ~1st Friday of following month at 08:30 ET.
    Default 35-day lag from month-start is safe (release is +5 to +14 days
    typically, +35 gives generous safety for any release-delay + TW timing)."""
    raw = load_us_labor_monthly(field)
    return _asof_pit(raw, trading_index, pit_lag_days, f"{field}_tw")


# ── Labor weekly ───────────────────────────────────────────────────────────
def load_us_labor_weekly(field: str) -> pd.Series:
    if field not in _LABOR_WEEKLY_FIELDS:
        raise ValueError(f"unknown labor weekly field {field!r}; valid: {_LABOR_WEEKLY_FIELDS}")
    return _load_us_labor_weekly_all()[field].dropna().rename(field)


def load_us_labor_weekly_tw(
    field: str, trading_index: pd.DatetimeIndex, pit_lag_days: int = 6,
) -> pd.Series:
    """ICSA/CCSA released Thursday 08:30 ET for prior Saturday's report week.
    The report date FRED shows is the Saturday of the reference week.
    Publication ~5 days after report Sat → default 6-day lag."""
    raw = load_us_labor_weekly(field)
    return _asof_pit(raw, trading_index, pit_lag_days, f"{field}_tw")


# ── Risk daily ─────────────────────────────────────────────────────────────
def load_us_risk(field: str) -> pd.Series:
    if field not in _RISK_DAILY_FIELDS:
        raise ValueError(f"unknown risk field {field!r}; valid: {_RISK_DAILY_FIELDS}")
    return _load_us_risk_all()[field].dropna().rename(field)


def load_us_risk_tw(
    field: str, trading_index: pd.DatetimeIndex, pit_lag_days: int = 1,
) -> pd.Series:
    """Daily VIX / credit spreads. Same convention as us_rate: value dated
    business-day D is usable on TW business-day D+1."""
    raw = load_us_risk(field)
    return _asof_pit(raw, trading_index, pit_lag_days, f"{field}_tw")


# ── shared as-of PIT helper (used by all 4 TW-aligners above) ─────────────
def _asof_pit(raw: pd.Series, trading_index: pd.DatetimeIndex,
              pit_lag_days: int, out_name: str) -> pd.Series:
    df = pd.DataFrame({
        "value":             raw.values,
        "available_from_tw": raw.index + pd.Timedelta(days=pit_lag_days),
    }).dropna(subset=["value"]).sort_values("available_from_tw").reset_index(drop=True)
    tw = pd.DataFrame({"tw_date": pd.DatetimeIndex(trading_index).sort_values().astype("datetime64[ns]")})
    merged = pd.merge_asof(
        tw, df[["available_from_tw", "value"]],
        left_on="tw_date", right_on="available_from_tw",
        direction="backward",
    )
    return pd.Series(merged["value"].values, index=merged["tw_date"],
                     name=out_name).sort_index().reindex(trading_index)


__all__ = [
    "load_us_price", "load_us_price_tw", "load_us_price_yoy_tw",
    "load_us_rate",  "load_us_rate_tw",
    "load_us_labor_monthly", "load_us_labor_monthly_tw",
    "load_us_labor_weekly",  "load_us_labor_weekly_tw",
    "load_us_risk",  "load_us_risk_tw",
    "available_us_price_fields", "available_us_rate_fields",
    "available_us_labor_monthly_fields",
    "available_us_labor_weekly_fields",
    "available_us_risk_fields",
]
