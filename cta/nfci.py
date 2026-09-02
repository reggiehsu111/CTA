"""
Chicago Fed National Financial Conditions Index (NFCI) — PIT-safe loader.

Data source & schedule (verified from chicagofed.org):
------------------------------------------------------
  * Cadence:    Weekly.
  * Value date: The **previous Friday**.
  * Published:  Wednesday of week+1 at 08:30 a.m. ET
                (= 20:30 TPE during US DST / 21:30 TPE during US Standard).
  * Fallback:   If Wed is a US federal holiday, published Thursday instead.
  * Publication lag: 5 calendar days (Fri value → next Wed publish) plus a
                     one-day safety buffer for TPE wall-clock timing and
                     Wed→Thu holiday shifts → **6 calendar days**.

We store weekly values in RDS `us_nfci` keyed by the value's Friday date.
For any TW date `D`, the freshest NFCI observable is dated `d` where
`d + pit_lag_days ≤ D`. Default `pit_lag_days=6`.

  D − d  ≥ 6 (calendar days)
  ⇔ NFCI[Fri d] usable on TW ≥ Thursday of week (d+1)

Interaction with cta.Simulate
------------------------------
`cta.Simulate` applies `signal.shift(2)`. Combined with our forward-filled
weekly series, the signal for TW date D uses NFCI dated (D − 6 − 2×TW_day)
or earlier — extremely conservative, safe for any variant.

Usage
-----
    >>> import cta
    >>> ASSET = cta.load_asset('mtx', '1d')
    >>>
    >>> # Raw weekly series (Friday-dated):
    >>> nfci_raw = cta.load_nfci('NFCI')     # or 'ANFCI', 'NFCICREDIT', ...
    >>>
    >>> # TW-calendar aligned + forward-filled, PIT-safe:
    >>> nfci = cta.load_nfci_tw('NFCI', ASSET.index)   # pit_lag_days=6 default
    >>>
    >>> # Tighter lag if you know your execution beats Wed 21:30 TPE:
    >>> nfci_tight = cta.load_nfci_tw('NFCI', ASSET.index, pit_lag_days=5)
"""
from __future__ import annotations

import pandas as pd


_VALID_FIELDS = ("nfci", "anfci", "credit", "risk", "leverage", "nonfin_lev")

_RAW_CACHE: pd.DataFrame | None = None


def _read_from_db() -> pd.DataFrame:
    """Query `us_nfci` into a Friday-indexed DataFrame."""
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path: sys.path.insert(0, _LIBS)
    from db_utils import engine
    df = pd.read_sql(
        "SELECT date, nfci, anfci, credit, risk, leverage, nonfin_lev "
        "FROM us_nfci ORDER BY date",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.set_index("date").sort_index()
    for c in _VALID_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def _read_raw() -> pd.DataFrame:
    """Cached raw NFCI table (weekly, Friday-dated)."""
    global _RAW_CACHE
    if _RAW_CACHE is None:
        _RAW_CACHE = _read_from_db()
    return _RAW_CACHE


def load_nfci(field: str = "nfci") -> pd.Series:
    """Raw weekly NFCI series, indexed by Friday value dates.

    Do NOT feed this directly into `cta.Simulate` for MTX backtests — the
    dates are US-Friday-dated with no PIT lag applied. Use `load_nfci_tw`
    to align to the TW calendar with the correct forward-fill + PIT lag.

    Parameters
    ----------
    field : one of 'nfci', 'anfci', 'credit', 'risk', 'leverage', 'nonfin_lev'.
    """
    if field not in _VALID_FIELDS:
        raise ValueError(f"Unknown NFCI field {field!r}. Valid: {_VALID_FIELDS}")
    return _read_raw()[field].rename(f"NFCI_{field}")


def load_nfci_pit(field: str = "nfci", as_of=None) -> pd.Series:
    """NFCI as it was actually known at `as_of` — the true point-in-time view.

    `load_nfci` returns the CURRENT vintage. That is not what you knew at the
    time: the Chicago Fed restates NFCI, and `us_macro_ingest` upserts a
    rolling 2-year window with DO UPDATE, so the stored history is rewritten
    with the latest vintage every week. Measured 2026-08-24, 66.9% of settled
    history had changed within one month (mean |rev| 0.0052 vs series std
    1.0). Because `nfci_loose_drift_d3_12` fires on a threshold, that moved
    its event set from 427 to 422 — 3.5% of events unstable.

    This reads `us_nfci_vintage`, an append-only log that records every value
    the ingest has ever seen, and returns the newest observation per date that
    was recorded at or before `as_of`.

    LIMITATION: vintage capture began 2026-08-24. For any `as_of` at or before
    that date this necessarily returns the seeded (i.e. then-current) vintage,
    so it cannot retroactively make older backtests point-in-time — recovering
    that needs ALFRED. It makes PIT drift measurable from here forward.

    Parameters
    ----------
    field : one of 'nfci', 'anfci', 'credit', 'risk', 'leverage', 'nonfin_lev'.
    as_of : timestamp-like, or None for the latest recorded vintage.
    """
    if field not in _VALID_FIELDS:
        raise ValueError(f"Unknown NFCI field {field!r}. Valid: {_VALID_FIELDS}")
    import sys as _sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in _sys.path: _sys.path.insert(0, _LIBS)
    from db_utils import engine

    params, clause = {}, ""
    if as_of is not None:
        params["as_of"] = pd.Timestamp(as_of).to_pydatetime()
        clause = "WHERE observed_at <= %(as_of)s"
    df = pd.read_sql(
        f"""SELECT DISTINCT ON (date) date, {field}
              FROM us_nfci_vintage {clause}
             ORDER BY date, observed_at DESC""",
        engine, params=params or None,
    )
    if df.empty:
        return pd.Series(dtype=float, name=f"NFCI_{field}_pit")
    df["date"] = pd.to_datetime(df["date"])
    return (df.set_index("date")[field].astype(float)
              .sort_index().rename(f"NFCI_{field}_pit"))


def load_nfci_tw(
    field: str,
    trading_index: pd.DatetimeIndex,
    pit_lag_days: int = 6,
) -> pd.Series:
    """
    NFCI series aligned to a **Taiwan trading calendar**, forward-filled,
    with strict PIT safety.

    For each TW date `D` in `trading_index`, returns the NFCI value from
    the latest Friday `d` satisfying `d + pit_lag_days ≤ D`.

    Default `pit_lag_days=6` means: NFCI[Fri d] usable on TW date `D ≥ Thu
    of the week after` — universally safe (covers Wed 20:30 TPE
    publication, plus Wed→Thu holiday shifts, plus DST edge cases).

    Rationale for pit_lag_days=6
    ----------------------------
    * NFCI[Fri d] published Wed (d+5 days) 08:30 ET = 20:30 TPE (DST) or
      21:30 TPE (Standard). All our MTX variants trade before that hour
      on Wed unless they specifically re-fire after Wed close.
    * Adding 1 day → NFCI[Fri d] first usable on TW-Thursday (d+6 days).
      That's before ALL possible MTX execution times on that Thursday.
    * On rare Wed holidays, Chicago Fed shifts publication to Thu — the
      value is still available by Fri (d+7 days), still ≤ our lag.

    Parameters
    ----------
    field         : 'nfci' / 'anfci' / 'credit' / 'risk' / 'leverage' /
                    'nonfin_lev'.
    trading_index : pd.DatetimeIndex of TW trading dates.
    pit_lag_days  : minimum calendar-day lag between value's Friday date
                    and the earliest TW date it can be used on. Default 6.

    Returns
    -------
    pd.Series indexed by ``trading_index``. Forward-filled (each TW date
    inherits the last observable NFCI). NaN before the first available.
    """
    if pit_lag_days < 5:
        raise ValueError(
            f"pit_lag_days must be ≥ 5 (NFCI Fri→Wed publish gap is 5 days). "
            f"Got {pit_lag_days}."
        )
    if field not in _VALID_FIELDS:
        raise ValueError(f"Unknown NFCI field {field!r}. Valid: {_VALID_FIELDS}")
    if trading_index is None or len(trading_index) == 0:
        return pd.Series([], dtype=float, name=f"NFCI_{field}_tw")

    raw = _read_raw()[field].dropna()

    # Shift each Friday value forward by pit_lag_days → the earliest calendar
    # date on which it's "observable". Then merge-asof onto the TW calendar
    # (backward direction: take the latest observable value ≤ TW date).
    df = pd.DataFrame({
        "friday_date":       raw.index,
        "value":             raw.values,
        "available_from_tw": raw.index + pd.Timedelta(days=pit_lag_days),
    }).sort_values("available_from_tw").reset_index(drop=True)

    tw = pd.DataFrame({"tw_date": pd.DatetimeIndex(trading_index).sort_values()})

    # pandas 3 infers datetime64[s] from psycopg2 dates while the TW calendar
    # is us/ns — merge_asof raises MergeError on mismatched resolutions, so
    # cast both keys. Same fix as global_macro._align. (QNT-15, 2026-09-01)
    tw["tw_date"] = tw["tw_date"].astype("datetime64[ns]")
    df["available_from_tw"] = df["available_from_tw"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        tw, df[["available_from_tw", "value"]],
        left_on="tw_date", right_on="available_from_tw",
        direction="backward",
    )
    out = pd.Series(
        merged["value"].values, index=merged["tw_date"],
        name=f"NFCI_{field}_tw",
    ).sort_index()
    return out.reindex(trading_index)


def available_nfci_fields() -> list[str]:
    """List the NFCI fields available in this loader."""
    return list(_VALID_FIELDS)


__all__ = ["load_nfci", "load_nfci_pit", "load_nfci_tw", "available_nfci_fields"]
