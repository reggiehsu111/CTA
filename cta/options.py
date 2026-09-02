"""
TAIFEX TXO options loader.

Reads the per-strike daily data scraped by
`QuantResearch/tools/scrape_taifex_options_daily.py` (default location
`mtx/taifex_txo_daily.csv`) and exposes it through three tiers of API:

1. **Direct lookup** — `load_option(strike, expiry, side)` returns the daily
   time series of one specific contract.
2. **Contract-selection helpers** — find the ATM strike per date, list the
   strikes/expiries available, identify front-month expiries.
3. **Aggregate loaders** — pre-baked common signals: PCR, ATM straddle
   premium, put skew, front-share of OI, etc. — all as date-indexed Series
   ready for `cta.Simulate` / factor-zoo pipelines.

CSV schema (one row per date × side × expiry × strike):

    date, market_code, side, expiry, strike,
    high, low, last, settlement, change, volume, oi

Sides are 'call' / 'put'. Market codes: 0 = 一般 (day), 1 = 盤後 (after-hours).
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Default CSV path — same directory as taifex_three_majors.csv
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "taifex_txo_daily.csv"

_NUMERIC_COLS = ["high", "low", "last", "settlement", "change", "volume", "oi"]
_SIDES        = ("call", "put")

# Lazy cache — raw source read once per process
_RAW_CACHE: pd.DataFrame | None = None
_RAW_CACHE_SOURCE: str | None = None


def _read_from_db() -> pd.DataFrame:
    """Query `tw_options_daily` into a DataFrame matching the CSV schema."""
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path: sys.path.insert(0, _LIBS)
    from db_utils import engine
    df = pd.read_sql(
        "SELECT date, market_code, side, expiry, strike, "
        "high, low, last, settlement, change, volume, oi "
        "FROM tw_options_daily ORDER BY date, market_code, side, expiry, strike",
        engine,
    )
    df["date"]   = pd.to_datetime(df["date"])
    df["expiry"] = df["expiry"].astype(str)
    df["side"]   = df["side"].astype(str).str.lower()
    # Cast numeric columns to float (CSV uses float; DB comes back mixed)
    for c in ("high", "low", "last", "settlement", "change"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ("strike", "volume", "oi", "market_code"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def _load_total_from_db(metric: str, side: str | None,
                        expiry_filter: str, market: str | int) -> pd.Series:
    """SQL-side equivalent of the groupby in `load_option_daily_total`.

    The WHERE clauses mirror the pandas filters EXACTLY, so switching to this
    path changes performance and nothing else:

        monthly  df["expiry"].str.match(r"^\\d{6}$")  ->  expiry ~ '^[0-9]{6}$'
        weekly   df["expiry"].str.contains("W")       ->  expiry LIKE '%W%'
        front    expiry == date.strftime("%Y%m")      ->  expiry = to_char(date,'YYYYMM')

    Note `weekly` deliberately keeps the literal "contains W" test. TAIFEX
    added F3/F4 expiry codes in 2026 which match neither the monthly nor the
    weekly predicate; they were already excluded from both under the pandas
    filters, and silently folding them in here would change what the live
    signals compute while we are supposed to be fixing only the data.

    ONE deliberate behaviour change: for a date whose rows are all NULL for
    `metric`, pandas' groupby-sum returns 0.0 while SQL SUM returns NULL
    (-> NaN). The NaN is the honest answer and is kept on purpose. The 0.0 is
    precisely what hid the 2026-07-28 outage: with OI missing for 20 trading
    days, both live options signals read a confident zero, z-scored it, and
    held a pinned position instead of going NaN and failing loudly.
    """
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path: sys.path.insert(0, _LIBS)
    from db_utils import engine

    where, params = [], {}
    if market == "regular" or market == 0:
        where.append("market_code = 0")
    elif market == "afterhours" or market == 1:
        where.append("market_code = 1")
    elif not (market == "both" or market is None):
        raise ValueError(
            f"market must be 'regular' / 'afterhours' / 'both' (got {market!r})")

    if side is not None:
        where.append("side = %(side)s")
        params["side"] = _validate_side(side)

    if expiry_filter == "monthly":
        where.append("expiry ~ '^[0-9]{6}$'")
    elif expiry_filter == "weekly":
        where.append("expiry LIKE '%%W%%'")
    elif expiry_filter == "front":
        where.append("expiry = to_char(date, 'YYYYMM')")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    df = pd.read_sql(
        f"SELECT date, SUM({metric}) AS v FROM tw_options_daily{clause} "
        "GROUP BY date ORDER BY date",
        engine, params=params,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["v"].astype(float)


def _read_raw(csv_path: str | Path | None = None, use_db: bool = True) -> pd.DataFrame:
    """Load the raw long-format table, cached after first call.

    Assumptions applied: expiry is str, date is Timestamp, market_code default 0.
    Source: DB (`tw_options_daily`) if ``use_db`` and no explicit ``csv_path``;
    otherwise the CSV.
    """
    global _RAW_CACHE, _RAW_CACHE_SOURCE
    source = "csv" if csv_path is not None else ("db" if use_db else "csv")
    if _RAW_CACHE is not None and csv_path is None and _RAW_CACHE_SOURCE == source:
        return _RAW_CACHE

    if source == "db":
        df = _read_from_db()
    else:
        path = Path(csv_path) if csv_path else _DEFAULT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"TXO options CSV not found at {path}. "
                "Scrape it first with "
                "QuantResearch/tools/scrape_taifex_options_daily.py."
            )
        df = pd.read_csv(path, dtype={"expiry": str},
                         parse_dates=["date"], low_memory=False)
        df["side"] = df["side"].astype(str).str.lower()

    if csv_path is None:
        _RAW_CACHE = df
        _RAW_CACHE_SOURCE = source
    return df


def _validate_side(side: str) -> str:
    side = str(side).lower()
    if side in ("c", "call"):  return "call"
    if side in ("p", "put"):   return "put"
    raise ValueError(f"side must be 'call' or 'put' (got {side!r})")


def _pit_shift(s: pd.Series, days: int) -> pd.Series:
    """Same convention as three_majors: shift daily series forward N days so
    values labeled D become D-1's published number."""
    return s.shift(days) if days > 0 else s


def _resolve_market(df: pd.DataFrame, market: str | int) -> pd.DataFrame:
    if market == "regular" or market == 0:  return df[df["market_code"] == 0]
    if market == "afterhours" or market == 1: return df[df["market_code"] == 1]
    if market == "both" or market is None:  return df
    raise ValueError(f"market must be 'regular' / 'afterhours' / 'both' (got {market!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Direct lookup — one specific contract
# ─────────────────────────────────────────────────────────────────────────────

def load_option(
    strike:  float | int,
    expiry:  str,
    side:    str,
    metric:  str = "settlement",
    market:  str | int = "regular",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """Daily time series of one specific TXO contract.

    Parameters
    ----------
    strike : int/float — strike price (index level, e.g. 17800)
    expiry : str — TAIFEX expiry code, e.g. '202506', '202506W1', '202506W2'
    side   : 'call' / 'put' (or 'c' / 'p')
    metric : which numeric column — one of {'volume','oi','settlement','last',
             'high','low','change'}. Default 'settlement' (marked close).
    market : 'regular' (一般, MarketCode=0), 'afterhours' (盤後, =1), or 'both'.
    pit_lag_days : shift by N days for intraday broadcasts. Daily backtests
                   with cta.Simulate should keep 0 (Simulate.shift(2) handles PIT).
    csv_path : override default file location.

    Returns
    -------
    pd.Series indexed by date. Name = f"TXO_{expiry}_{strike:g}{side_char}_{metric}"
    """
    side = _validate_side(side)
    if metric not in _NUMERIC_COLS:
        raise ValueError(f"metric must be one of {_NUMERIC_COLS} (got {metric!r})")

    df = _read_raw(csv_path, use_db=use_db)
    df = _resolve_market(df, market)
    sub = df[(df["strike"] == strike) & (df["expiry"] == expiry) & (df["side"] == side)]
    if sub.empty:
        raise KeyError(
            f"No rows for strike={strike}, expiry={expiry!r}, side={side!r}. "
            "Try `cta.option_strikes(expiry, side)` to see what's available."
        )
    name = f"TXO_{expiry}_{int(strike) if float(strike).is_integer() else strike}"
    name += "C" if side == "call" else "P"
    name += f"_{metric}"
    ser = sub.set_index("date")[metric].sort_index().rename(name)
    return _pit_shift(ser, pit_lag_days)


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Contract-selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def option_expiries(csv_path: str | Path | None = None, use_db: bool = True) -> list[str]:
    """Every expiry code present in the source (sorted)."""
    return sorted(_read_raw(csv_path, use_db=use_db)["expiry"].dropna().unique().tolist())


def option_strikes(
    expiry: str,
    side:   str | None = None,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> list[float]:
    """Sorted list of strikes available for one expiry (and optionally side)."""
    df = _read_raw(csv_path, use_db=use_db)
    sub = df[df["expiry"] == expiry]
    if side is not None:
        sub = sub[sub["side"] == _validate_side(side)]
    return sorted(sub["strike"].dropna().unique().tolist())


def front_month_expiry(
    on_date: str | pd.Timestamp | None = None,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> str:
    """Front-monthly expiry (YYYYMM only, no W) as of `on_date`.

    Convention: the earliest expiry whose YYYYMM is ≥ the date's YYYYMM.
    Uses monthly (not weekly) expiries so it's stable across the trading month.
    """
    df = _read_raw(csv_path, use_db=use_db)
    if on_date is None:
        on_date = df["date"].max()
    on_date = pd.Timestamp(on_date)
    cur_yyyymm = on_date.strftime("%Y%m")
    monthly_expiries = [e for e in df["expiry"].dropna().unique()
                        if re.fullmatch(r"\d{6}", str(e)) and str(e) >= cur_yyyymm]
    if not monthly_expiries:
        raise ValueError(f"no monthly expiries >= {cur_yyyymm}")
    return min(monthly_expiries)


def load_atm_option(
    expiry:  str,
    side:    str,
    spot:    pd.Series | None = None,
    metric:  str = "settlement",
    market:  str | int = "regular",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """For each date, pick the strike closest to `spot[date]` — daily
    'nearest-to-ATM' series. Useful for tracking ATM IV proxies over time.

    Parameters
    ----------
    spot : underlying spot Series, date-indexed. Defaults to `cta.load_asset('mtx','1d')`
           close price (MXF tracks TAIEX so it's a fine underlying proxy).
    """
    side = _validate_side(side)
    df = _read_raw(csv_path, use_db=use_db)
    df = _resolve_market(df, market)
    sub = df[(df["expiry"] == expiry) & (df["side"] == side)].copy()
    if sub.empty:
        raise KeyError(f"No rows for expiry={expiry!r}, side={side!r}")

    if spot is None:
        # Lazy import to avoid circular dependency with cta.simulate
        from .simulate import load_asset
        asset = load_asset("mtx", "1d")
        spot = asset["close"]

    sub = sub.merge(spot.rename("spot"), left_on="date", right_index=True, how="inner")
    sub["dist"] = (sub["strike"] - sub["spot"]).abs()
    nearest = sub.sort_values("dist").drop_duplicates("date", keep="first")
    ser = (nearest.set_index("date")[metric]
                  .sort_index()
                  .rename(f"TXO_{expiry}_ATM_{side}_{metric}"))
    return _pit_shift(ser, pit_lag_days)


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: Aggregate loaders — pre-baked common signals
# ─────────────────────────────────────────────────────────────────────────────

def load_option_daily_total(
    metric: str = "volume",
    side:   str | None = None,
    expiry_filter: str = "all",
    market: str | int = "regular",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """Sum a metric across strikes (and optionally sides / expiries) per date.

    Parameters
    ----------
    metric : 'volume' or 'oi' (or any numeric column).
    side   : 'call', 'put', or None for both.
    expiry_filter : 'all' — sum across every expiry present
                    'monthly' — YYYYMM-only (drop weeklies)
                    'weekly'  — YYYYMMWn only
                    'front'   — only the front-month YYYYMM per date
    """
    if metric not in _NUMERIC_COLS:
        raise ValueError(f"metric must be one of {_NUMERIC_COLS}")
    if expiry_filter not in ("all", "monthly", "weekly", "front"):
        raise ValueError("expiry_filter must be 'all'/'monthly'/'weekly'/'front'")

    # Aggregate server-side when reading the DB. `tw_options_daily` went from
    # ~440k rows (the ~20-strike 簡表) to 4.6M when the full-strike download
    # replaced it on 2026-08-24, and `_read_raw` pulls + caches the WHOLE
    # table. That is ~500MB of pandas: it times out interactively and would
    # OOM the 768MB mtx-signal-runner Lambda on its next scheduled run.
    # Summing in Postgres returns one row per date instead of millions.
    # A test asserts this path matches the pandas path exactly.
    if csv_path is None and use_db:
        ser = _load_total_from_db(metric, side, expiry_filter, market)
        name = f"TXO_{side or 'both'}_{expiry_filter}_{metric}"
        return _pit_shift(ser.rename(name), pit_lag_days)

    df = _read_raw(csv_path, use_db=use_db)
    df = _resolve_market(df, market)
    if side is not None:
        df = df[df["side"] == _validate_side(side)]

    if expiry_filter == "monthly":
        df = df[df["expiry"].str.match(r"^\d{6}$", na=False)]
    elif expiry_filter == "weekly":
        df = df[df["expiry"].str.contains("W", na=False)]
    elif expiry_filter == "front":
        cur_ym = df["date"].dt.strftime("%Y%m")
        df = df[df["expiry"] == cur_ym]
    elif expiry_filter != "all":
        raise ValueError("expiry_filter must be 'all'/'monthly'/'weekly'/'front'")

    ser = df.groupby("date")[metric].sum().sort_index()
    name = f"TXO_{side or 'both'}_{expiry_filter}_{metric}"
    return _pit_shift(ser.rename(name), pit_lag_days)


def load_pcr(
    kind:   str = "volume",
    expiry_filter: str = "all",
    market: str | int = "regular",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """Put/Call ratio, daily. `kind` = 'volume' or 'oi'."""
    if kind not in ("volume", "oi"):
        raise ValueError("kind must be 'volume' or 'oi'")
    calls = load_option_daily_total(kind, "call", expiry_filter, market, 0, csv_path)
    puts  = load_option_daily_total(kind, "put",  expiry_filter, market, 0, csv_path)
    pcr = (puts / calls.replace(0, np.nan)).rename(f"TXO_pcr_{kind}_{expiry_filter}")
    return _pit_shift(pcr, pit_lag_days)


def load_atm_straddle_pct(
    spot:    pd.Series | None = None,
    expiry:  str | None = None,
    market:  str | int = "regular",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """ATM straddle premium as % of spot — an implied-vol proxy.

    Straddle = ATM call settlement + ATM put settlement, on the front-month
    expiry (or a user-specified `expiry`). Divided by spot for scale-invariance.

    Rough conversion to annualized IV: multiply by sqrt(252 / days-to-expiry).
    """
    if spot is None:
        from .simulate import load_asset
        asset = load_asset("mtx", "1d")
        spot = asset["close"]

    df = _read_raw(csv_path, use_db=use_db)
    df = _resolve_market(df, market)
    if expiry is not None:
        df = df[df["expiry"] == expiry]
    else:
        # Front-month per date
        df = df[df["expiry"] == df["date"].dt.strftime("%Y%m")]

    df = df.merge(spot.rename("spot"), left_on="date", right_index=True, how="inner")
    df["dist"] = (df["strike"] - df["spot"]).abs()
    nearest = (df.sort_values("dist")
                 .drop_duplicates(["date", "side"], keep="first")
                 .pivot(index="date", columns="side", values="settlement")
                 .dropna())
    nearest = nearest.join(spot.rename("spot"), how="inner")
    straddle_pct = ((nearest["call"] + nearest["put"]) / nearest["spot"] * 100)
    return _pit_shift(straddle_pct.rename("TXO_atm_straddle_pct"), pit_lag_days)


def load_put_skew(
    otm_pct: float = 0.05,
    spot:    pd.Series | None = None,
    expiry:  str | None = None,
    market:  str | int = "regular",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """OTM put / OTM call settlement ratio at symmetric moneyness.

    For each date, pick the front-month put closest to `spot × (1 - otm_pct)`
    and the front-month call closest to `spot × (1 + otm_pct)`. The ratio is
    a "crash-fear" indicator — >1 means puts are richer than symmetric calls.
    """
    if spot is None:
        from .simulate import load_asset
        asset = load_asset("mtx", "1d")
        spot = asset["close"]

    df = _read_raw(csv_path, use_db=use_db)
    df = _resolve_market(df, market)
    if expiry is not None:
        df = df[df["expiry"] == expiry]
    else:
        df = df[df["expiry"] == df["date"].dt.strftime("%Y%m")]

    df = df.merge(spot.rename("spot"), left_on="date", right_index=True, how="inner")
    df = df[df["settlement"] > 0]
    df["target"] = np.where(df["side"] == "put",
                            df["spot"] * (1 - otm_pct),
                            df["spot"] * (1 + otm_pct))
    df["dist"] = (df["strike"] - df["target"]).abs()

    nearest = (df.sort_values("dist")
                 .drop_duplicates(["date", "side"], keep="first")
                 .pivot(index="date", columns="side", values="settlement")
                 .dropna())
    ratio = (nearest["put"] / nearest["call"].replace(0, np.nan))
    return _pit_shift(ratio.rename(f"TXO_put_skew_{int(otm_pct*100)}pct"), pit_lag_days)


def load_front_share_oi(
    market:  str | int = "regular",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """Share of total OI concentrated in the front-month expiry.

    Positioning-horizon indicator: high share = attention is on the near-term.
    """
    df = _read_raw(csv_path, use_db=use_db)
    df = _resolve_market(df, market)
    cur_ym = df["date"].dt.strftime("%Y%m")
    df = df.assign(is_front=(df["expiry"] == cur_ym))
    grouped = df.groupby(["date", "is_front"])["oi"].sum().unstack(fill_value=0)
    if True in grouped.columns and False in grouped.columns:
        share = (grouped[True] / (grouped[True] + grouped[False]).replace(0, np.nan))
    elif True in grouped.columns:
        share = pd.Series(1.0, index=grouped.index)
    else:
        share = pd.Series(0.0, index=grouped.index)
    return _pit_shift(share.rename("TXO_front_share_oi"), pit_lag_days)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — cta.Options() sugar (per the user's requested API)
# ─────────────────────────────────────────────────────────────────────────────

def Options(
    strike:  float | int,
    expiry:  str,
    put_call: str,
    metric:  str = "settlement",
    **kwargs,
) -> pd.Series:
    """Convenience alias for `load_option()` matching the sketch API
    `cta.Options(strike_price, expiry_date, put_call)`.

    All keyword args (metric, market, pit_lag_days, csv_path) forward to
    `load_option`.
    """
    return load_option(strike, expiry, put_call, metric, **kwargs)


__all__ = [
    "Options",
    "load_option",
    "load_atm_option",
    "option_strikes", "option_expiries", "front_month_expiry",
    "load_option_daily_total",
    "load_pcr",
    "load_atm_straddle_pct",
    "load_put_skew",
    "load_front_share_oi",
]
