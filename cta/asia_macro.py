"""
Taiwan / Japan macro loaders — PIT-safe, TW-calendar aligned. (QNT-10)

Backs `tw_macro_monthly`, `jp_macro_monthly` and `jp_markets_daily`, ingested by
`tools/macro_ingest/ingest_asia_macro.py`.

Why the raw loaders are not enough
----------------------------------
Every monthly series here is stamped with its REFERENCE month, not its release
date, and the gap is large: Taiwan's 景氣指標 for month M is published around the
27th of M+1 — a 58-day lag from the first of M. Joining a monthly frame onto a
daily calendar by reference date therefore hands the backtest roughly two months
of look-ahead, which is more than enough to manufacture a Sharpe out of nothing.

So: `load_*_tw()` shifts each observation forward by a documented, deliberately
LATE publication lag (see `tools/macro_ingest/asia_macro_sources.py`) and then
merge-asof's backward onto the TW trading calendar. Prefer them. The raw
`load_*()` functions exist for coverage audits and event-mask construction, not
for building a tradable signal.

None of these lags is a scraped release date — no publisher here exposes a
machine-readable calendar — so they are conservative conventions, not truth.
State that assumption in any write-up, per the house rule on fill time.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

import pandas as pd

# Resolve tools/macro_ingest across machines and inside the Lambda package.
# This file sits one level below the mtx repo root, so the relative candidate
# covers both layouts (Mac: Research/mtx; EC2: ~/mtx). The bare repo-root entry
# covers the deployed /var/task layout, where prepare.sh vendors
# asia_macro_sources.py next to cta/ rather than under tools/.
# QNT-50: this was hardcoded to /home/ubuntu/mtx/tools/macro_ingest, which made
# `import cta` raise ModuleNotFoundError anywhere but this box - including in the
# mtx-signal-runner Lambda, whose handler imports cta at module scope.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC_CANDIDATES = [
    os.environ.get("MTX_MACRO_INGEST"),
    os.path.join(_ROOT, "tools", "macro_ingest"),
    _ROOT,
    "/home/ubuntu/mtx/tools/macro_ingest",
    "/Users/hsureggie/coding/Research/mtx/tools/macro_ingest",
]
for _cand in _SRC_CANDIDATES:
    if _cand and os.path.isfile(os.path.join(_cand, "asia_macro_sources.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
else:
    raise ModuleNotFoundError(
        "asia_macro_sources.py not found - tried: "
        + ", ".join(repr(c) for c in _SRC_CANDIDATES if c)
        + ". Set MTX_MACRO_INGEST to its directory, or vendor the file next to cta/."
    )
import asia_macro_sources as _S  # noqa: E402


def _engine():
    from db_utils import engine
    return engine


# Publication lag in calendar days from the FIRST DAY of the reference month.
TW_PUB_LAG = {
    **{c: _S.TW_SOURCES["cpi"]["pub_lag_days"] for c in (
        "cpi", "cpi_yoy", "cpi_food", "cpi_food_yoy", "cpi_apparel", "cpi_apparel_yoy",
        "cpi_housing", "cpi_housing_yoy", "cpi_transport", "cpi_transport_yoy",
        "cpi_health", "cpi_health_yoy", "cpi_education", "cpi_education_yoy",
        "cpi_misc", "cpi_misc_yoy")},
    "unemployment_rate": _S.TW_SOURCES["unemployment"]["pub_lag_days"],
    **{c: _S.TW_SOURCES["money"]["pub_lag_days"] for c in (
        "m1a", "m1a_yoy", "m1b", "m1b_yoy", "m2", "m2_yoy")},
    **{c: _S.TW_SOURCES["pmi"]["pub_lag_days"] for c in ("pmi", "nmi")},
    **{c: _S.TW_SOURCES["cycle"]["pub_lag_days"] for c in (
        "leading_idx", "leading_idx_nt", "coincident_idx", "coincident_idx_nt",
        "lagging_idx", "lagging_idx_nt", "monitor_score", "monitor_signal",
        "export_orders_dci", "semi_equip_imports", "industrial_production",
        "mfg_sales_idx", "customs_exports")},
}

JP_PUB_LAG = {
    **{c: _S.JP_OECD_CPI["pub_lag_days"] for c in (
        "cpi", "cpi_yoy", "cpi_core", "cpi_core_yoy",
        "cpi_energy", "cpi_energy_yoy", "cpi_food", "cpi_food_yoy")},
    **{c: cfg["pub_lag_days"] for c, cfg in _S.JP_FRED_MONTHLY.items()},
}


@lru_cache(maxsize=8)
def _read(table: str) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {table} ORDER BY date", _engine())
    if df.empty:
        raise RuntimeError(f"{table} is empty — run tools/macro_ingest/ingest_asia_macro.py")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").drop(columns=["created_at", "updated_at"], errors="ignore")


def available_tw_macro_fields() -> list[str]:
    return sorted(TW_PUB_LAG)


def available_jp_macro_fields() -> list[str]:
    return sorted(JP_PUB_LAG)


def load_tw_macro(field: str) -> pd.Series:
    """Raw monthly Taiwan series indexed by REFERENCE month-start.

    Not PIT-safe — see module docstring. Use `load_tw_macro_tw()` for signals.
    """
    if field not in TW_PUB_LAG:
        raise ValueError(f"Unknown TW field {field!r}. Valid: {available_tw_macro_fields()}")
    return _read("tw_macro_monthly")[field].dropna().rename(f"TW_{field}")


def load_jp_macro(field: str) -> pd.Series:
    """Raw monthly Japan series indexed by REFERENCE month-start. Not PIT-safe."""
    if field not in JP_PUB_LAG:
        raise ValueError(f"Unknown JP field {field!r}. Valid: {available_jp_macro_fields()}")
    return _read("jp_macro_monthly")[field].dropna().rename(f"JP_{field}")


def _align(raw: pd.Series, trading_index, lag_days: int, name: str) -> pd.Series:
    """Shift each observation to its earliest observable date, then merge-asof
    backward onto the TW trading calendar."""
    if trading_index is None or len(trading_index) == 0:
        return pd.Series([], dtype=raw.dtype, name=name)
    if raw.empty:
        return pd.Series(index=pd.DatetimeIndex(trading_index), dtype=float, name=name)

    src = pd.DataFrame({
        "available_from": pd.DatetimeIndex(raw.index) + pd.Timedelta(days=lag_days),
        "value": raw.values,
    }).sort_values("available_from").reset_index(drop=True)

    tw_idx = pd.DatetimeIndex(trading_index).sort_values()
    tw = pd.DataFrame({"tw_date": tw_idx})
    # pandas 3 infers datetime64[s] from psycopg2 dates while the TW calendar is
    # ns — merge_asof raises on mismatched resolutions, so cast both keys.
    tw["tw_date"] = tw["tw_date"].astype("datetime64[ns]")
    src["available_from"] = src["available_from"].astype("datetime64[ns]")

    merged = pd.merge_asof(tw, src, left_on="tw_date", right_on="available_from",
                           direction="backward")
    return pd.Series(merged["value"].values, index=tw_idx, name=name).reindex(
        pd.DatetimeIndex(trading_index))


def load_tw_macro_tw(field: str, trading_index, pub_lag_days: int | None = None) -> pd.Series:
    """PIT-safe Taiwan macro series on the TW trading calendar.

    Each reference month M is treated as first observable on
    `first_of_M + pub_lag_days`, then forward-filled. Override `pub_lag_days`
    only if you can justify a different fill-time assumption — and say so.
    """
    raw = load_tw_macro(field)
    lag = TW_PUB_LAG[field] if pub_lag_days is None else pub_lag_days
    return _align(raw, trading_index, lag, f"TW_{field}_tw")


def load_jp_macro_tw(field: str, trading_index, pub_lag_days: int | None = None) -> pd.Series:
    """PIT-safe Japan macro series on the TW trading calendar."""
    raw = load_jp_macro(field)
    lag = JP_PUB_LAG[field] if pub_lag_days is None else pub_lag_days
    return _align(raw, trading_index, lag, f"JP_{field}_tw")


def load_jp_market(field: str) -> pd.Series:
    """Daily Nikkei 225 / USDJPY indexed by its own date."""
    if field not in _S.JP_FRED_DAILY:
        raise ValueError(f"Unknown JP market field {field!r}. "
                         f"Valid: {sorted(_S.JP_FRED_DAILY)}")
    return _read("jp_markets_daily")[field].dropna().rename(f"JP_{field}")


def load_jp_market_tw(field: str, trading_index) -> pd.Series:
    """Daily Japan market series on the TW calendar.

    Lag 0 is NOT applied here: Tokyo closes 15:00 JST = 14:00 TPE, before the
    MTX day-session close, but USD/JPY on FRED is a NY-noon quote that lands
    after the TW close. Callers must apply their own `shift()` per the variant's
    execution window — this only aligns the calendar, it does not make the value
    executable. See the six-variant shift table in the standing brief.
    """
    return _align(load_jp_market(field), trading_index, 0, f"JP_{field}_tw")
