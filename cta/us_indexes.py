"""
US index / ETF loader with **strict PIT-safe alignment** to a Taiwan trading
calendar.

Reads the CSV produced by `QuantResearch/tools/download_us_indexes.py`
(default location `mtx/us_indexes.csv`), long-format with one row per
`(date, ticker, open, high, low, close, volume)`.

Timezone / PIT rationale
------------------------
US market close is ~16:00 America/New_York, which is ~04:00 Asia/Taipei on
the **next** calendar day. TW day session opens 08:45 TPE.

So SOX/SPY close on US date `U` is:
  * NOT usable during TW day session on the same-calendar-day-as-U (that
    session already closed at 13:45 TPE, before the US market even opened).
  * NOT usable during TW night session that opens 15:00 TPE on that same
    calendar day (US market opens ~21:30 TPE, still ~7 hrs in the future).
  * FIRST usable on TW day session that opens 08:45 TPE the **next** TW
    trading calendar day after U — because that's the first TW session
    whose open comes AFTER the US close at 04:00 TPE.

The alignment rule this module enforces:

    sox_tw[TW date D] = SOX_close(latest US date U with U + pit_lag_days <= D)

With `pit_lag_days=1` (the default): SOX close from US date D-1 becomes
labeled at TW date D. When D-1 is a US holiday or weekend, it falls back
to the previous US trading day.

Interaction with `cta.Simulate`
-------------------------------
`cta.Simulate` applies `signal.shift(2)` internally: PnL[TW t] uses
signal[TW t-2]. With this loader's default alignment, signal[TW t-2] =
SOX from US date t-3 (or earlier for weekends/holidays). This is
extra-conservative (3+ calendar days of lag), safe for close-to-close
daily backtests.

If you want the "trade the next TW open on last night's US close" (only
1 US trading day of lag), pair this loader with an OPEN-based simulator
that does NOT further shift the signal — do not just pass the loader
output to `cta.Simulate` and expect that faster semantic.

Usage
-----
    >>> import cta
    >>> ASSET = cta.load_asset('mtx', '1d')
    >>>
    >>> # PIT-safe SOX close, TW-calendar aligned:
    >>> sox = cta.load_us_index_tw('^SOX', field='close', trading_index=ASSET.index)
    >>> # sox[Tue] == SOX close from Mon US
    >>> # sox[Mon] == SOX close from Fri US (weekend gap)
    >>>
    >>> # Raw US-indexed series (no alignment — for research/analysis):
    >>> sox_raw = cta.load_us_index('^SOX', field='close')
    >>>
    >>> # Log return of SPY, aligned:
    >>> spy_ret = cta.load_us_index_tw('SPY', field='close', trading_index=ASSET.index).pct_change()
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "us_indexes.csv"

_VALID_FIELDS = ("open", "high", "low", "close", "volume")

_RAW_CACHE: pd.DataFrame | None = None
_RAW_CACHE_SOURCE: str | None = None


def _read_from_db() -> pd.DataFrame:
    """Query `us_index_pv` into a DataFrame matching the CSV schema."""
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path: sys.path.insert(0, _LIBS)
    from db_utils import engine
    df = pd.read_sql(
        "SELECT date, ticker, open, high, low, close, volume "
        "FROM us_index_pv ORDER BY ticker, date",
        engine,
    )
    df["date"]   = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["date"]   = df["date"].dt.normalize() if hasattr(df["date"].dt, "normalize") else df["date"]
    df["ticker"] = df["ticker"].astype(str)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype(float)
    return df


def _read_raw(csv_path: str | Path | None = None, use_db: bool = True) -> pd.DataFrame:
    """Load the raw long-format table, cached after first call.

    Source: DB (`us_index_pv`) if ``use_db`` and no explicit ``csv_path``;
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
                f"US-indexes CSV not found at {path}. "
                "Run: python QuantResearch/tools/download_us_indexes.py"
            )
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
        df["ticker"] = df["ticker"].astype(str)

    if csv_path is None:
        _RAW_CACHE = df
        _RAW_CACHE_SOURCE = source
    return df


def available_us_tickers(csv_path=None, use_db: bool = True) -> list[str]:
    """Return every ticker present in the CSV."""
    return sorted(_read_raw(csv_path, use_db=use_db)["ticker"].unique().tolist())


def load_us_index(
    ticker: str,
    field: str = "close",
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """
    Load a US index / ETF field, indexed by **US trading date** (naive).

    **This is the raw series — no TW-calendar alignment.** Do NOT feed
    directly into `cta.Simulate` for TW-asset backtests; that would use
    US date D's close to trade TW date D, which is a look-ahead violation.
    Use `load_us_index_tw` for TW-aligned, PIT-safe output.

    Parameters
    ----------
    ticker : Yahoo Finance ticker (e.g. '^SOX', 'SPY', '^SPX').
    field  : One of 'open', 'high', 'low', 'close', 'volume'. Default 'close'.
    csv_path : override default CSV location.
    """
    if field not in _VALID_FIELDS:
        raise ValueError(f"Unknown field {field!r}. Valid: {_VALID_FIELDS}")
    df = _read_raw(csv_path, use_db=use_db)
    sub = df[df["ticker"] == ticker]
    if sub.empty:
        raise KeyError(
            f"Ticker {ticker!r} not present in CSV. "
            f"Available: {available_us_tickers(csv_path)}"
        )
    ser = sub.set_index("date")[field].astype(float).sort_index()
    return ser.rename(f"{ticker}_{field}")


def load_us_index_tw(
    ticker: str,
    trading_index: pd.DatetimeIndex,
    field: str = "close",
    pit_lag_days: int = 1,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """
    Load a US index / ETF field, aligned to a **Taiwan trading calendar**
    with strict PIT safety.

    For each TW date D in `trading_index`, returns the ticker's `field`
    value from the latest US trading date U satisfying
    ``U + pit_lag_days ≤ D`` (calendar-day comparison).

    Default `pit_lag_days=1` implements the "US close on date U is usable
    on TW day session D = U + 1 calendar day" rule (see module docstring).
    Increase to make the alignment more conservative; **do not** decrease
    below 1 — that would use a US close that is not yet observable at
    TW's day-session open on the target date.

    Parameters
    ----------
    ticker        : Yahoo Finance ticker (e.g. '^SOX', 'SPY').
    trading_index : pd.DatetimeIndex — the TW trading calendar (e.g.
                    `cta.load_asset('mtx','1d').index`).
    field         : 'open' / 'high' / 'low' / 'close' / 'volume'.
    pit_lag_days  : minimum calendar-day lag between US date and TW date.
                    Default 1 (safe for daily backtests). Must be ≥ 1.
    csv_path      : override default CSV location.

    Returns
    -------
    pd.Series indexed by `trading_index`. NaN for TW dates before the
    ticker's first available US date (accounting for the lag). Weekends
    and US holidays are ffilled from the previous US trading day.
    """
    if pit_lag_days < 1:
        raise ValueError(
            "pit_lag_days must be ≥ 1 to preserve PIT safety on TW day-session open. "
            f"Got {pit_lag_days}."
        )
    if trading_index is None or len(trading_index) == 0:
        return pd.Series([], dtype=float, name=f"{ticker}_{field}_tw")

    us = load_us_index(ticker, field=field, csv_path=csv_path, use_db=use_db)

    us_df = us.reset_index()
    us_df.columns = ["us_date", "value"]
    us_df["available_from_tw"] = us_df["us_date"] + pd.Timedelta(days=pit_lag_days)
    us_df = us_df.sort_values("available_from_tw").reset_index(drop=True)

    tw_df = pd.DataFrame(
        {"tw_date": pd.DatetimeIndex(trading_index).sort_values()}
    )

    # merge_asof requires BOTH keys at the same datetime resolution. They can
    # legitimately differ: pandas 2.x infers datetime64[ns] everywhere, while
    # pandas 3.x infers [s] from psycopg2 `date` objects and [us] from other
    # paths. Production runs 2.2.3 and research runs 3.0.1, so without this
    # normalization the same code merges fine in the Lambda and raises
    # MergeError in a notebook. Pin both to [ns] rather than trusting either.
    tw_df["tw_date"] = tw_df["tw_date"].astype("datetime64[ns]")
    us_df["available_from_tw"] = us_df["available_from_tw"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        tw_df, us_df[["available_from_tw", "value"]],
        left_on="tw_date", right_on="available_from_tw",
        direction="backward",
    )

    out = pd.Series(
        merged["value"].values, index=merged["tw_date"],
        name=f"{ticker}_{field}_tw",
    ).sort_index()
    return out.reindex(trading_index)


__all__ = [
    "load_us_index",
    "load_us_index_tw",
    "available_us_tickers",
]
