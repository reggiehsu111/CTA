"""
operators.py — time-series operators for building CTA signals.

All functions accept a pd.Series and return a pd.Series, mirroring
the f-package operator convention but for single-instrument data.

Context-aware functions
-----------------------
Prices(field) and Returns(field, i) work against the active asset set by
simulate() (analogous to how the f-package uses DataLoader.get_active_context).
Call set_active_asset(asset) to point them at a specific BaseAsset manually.

    from cta import operators as ops
    ops.set_active_asset(mtx)
    ret = ops.Returns('c', -1)      # close-to-close 1-bar return
    px  = ops.Prices('h')           # high prices

Usage (explicit)
----------------
    sig = ops.InstMean(20, asset.close) - ops.InstMean(60, asset.close)
    sig = ops.InstZScore(252, sig)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Active-asset context  (mirrors f's DataLoader.get_active_context)
# ─────────────────────────────────────────────────────────────────────────────

_active_asset = None   # set by simulate() or manually via set_active_asset()


def set_active_asset(asset) -> None:
    """Point Prices() / Returns() at *asset* (a BaseAsset instance)."""
    global _active_asset
    _active_asset = asset


def _require_asset():
    if _active_asset is None:
        raise RuntimeError(
            "No active asset. Call set_active_asset(asset) or use simulate()."
        )
    return _active_asset


# ─────────────────────────────────────────────────────────────────────────────
# Prices / Returns  (context-aware, match f-package signature style)
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_MAP = {
    'o': 'open',
    'h': 'high',
    'l': 'low',
    'c': 'close',
    'v': 'volume',
    # also accept full names
    'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume',
}


def Prices(field: str = 'c') -> pd.Series:
    """
    Return the price series for *field* from the active asset.

    Parameters
    ----------
    field : 'o' | 'h' | 'l' | 'c' | 'v'  (or full column names)

    Returns
    -------
    pd.Series indexed by the asset's trading dates.
    """
    asset = _require_asset()
    col = _FIELD_MAP.get(field)
    if col is None:
        raise ValueError(f"Unknown field '{field}'. Choose from: {list(_FIELD_MAP)}")
    return asset[col].rename(field)


def Returns(field: str = 'c', i: int = -1) -> pd.Series:
    """
    Return the i-bar return series of *field* from the active asset.

    Convention (matches f.Returns):
        i < 0  →  past return  : px[t] / px[t - |i|] - 1   (default i=-1)
        i > 0  →  future return: px[t + i] / px[t] - 1      (look-ahead — research only)

    Returns are clipped to ±50% to remove data errors.

    Parameters
    ----------
    field : price field, same as Prices()
    i     : lag/lead in bars; i=0 is not allowed
    """
    if i == 0:
        raise ValueError("i cannot be 0")
    px = Prices(field)
    if i < 0:
        ret = px / px.shift(-i) - 1          # px[t] / px[t - |i|] - 1
    else:
        ret = px.shift(-i) / px - 1          # px[t + i] / px[t] - 1
    return ret.clip(-0.5, 0.5).rename(f"ret_{field}_{i}")


# ─────────────────────────────────────────────────────────────────────────────
# Shift
# ─────────────────────────────────────────────────────────────────────────────

def Lag(n: int, s: pd.Series) -> pd.Series:
    """Shift s backward by n bars (no look-ahead)."""
    assert n > 0, "Lag requires n > 0. Use Lead for forward shifts."
    return s.shift(n)


def Lead(n: int, s: pd.Series) -> pd.Series:
    """
    Shift s forward by n bars (look-ahead bias).
    For research / labelling purposes only.
    """
    assert n > 0, "Lead requires n > 0. Use Lag for backward shifts."
    return s.shift(-n)


# ─────────────────────────────────────────────────────────────────────────────
# Fill
# ─────────────────────────────────────────────────────────────────────────────

def ForwardFill(limit: int, s: pd.Series, stop_events: pd.Series | None = None) -> pd.Series:
    """
    Forward-fill NaN values up to *limit* consecutive bars.

    Parameters
    ----------
    limit       : maximum consecutive bars to fill
    s           : signal series
    stop_events : optional 0/1 series; fill resets where this is 1
    """
    result = s.ffill(limit=limit)
    if stop_events is not None:
        stop_mask = stop_events.cumsum().astype(bool)
        result = result.where(~stop_mask, other=np.nan)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Rolling moments
# ─────────────────────────────────────────────────────────────────────────────

def InstMean(n: int, s: pd.Series, min_periods: int | None = None) -> pd.Series:
    """Rolling mean over n bars."""
    mp = min_periods if min_periods is not None else max(1, n // 2)
    return s.rolling(n, min_periods=mp).mean()


def InstStdev(n: int, s: pd.Series, min_periods: int | None = None) -> pd.Series:
    """Rolling standard deviation over n bars (ddof=1)."""
    mp = min_periods if min_periods is not None else max(1, n // 2)
    return s.rolling(n, min_periods=mp).std(ddof=1)


def InstSkew(n: int, s: pd.Series, min_periods: int | None = None) -> pd.Series:
    """Rolling skewness over n bars."""
    mp = min_periods if min_periods is not None else max(3, n // 2)
    return s.rolling(n, min_periods=mp).skew()


def InstSum(n: int, s: pd.Series, min_periods: int | None = None) -> pd.Series:
    """Rolling sum over n bars."""
    mp = min_periods if min_periods is not None else max(1, n // 2)
    return s.rolling(n, min_periods=mp).sum()


# ─────────────────────────────────────────────────────────────────────────────
# Rolling rank / z-score
# ─────────────────────────────────────────────────────────────────────────────

def InstRank(n: int, s: pd.Series) -> pd.Series:
    """
    Rolling percentile rank of the current bar within the last n bars.
    Returns values in [0, 1].
    """
    return s.rolling(n, min_periods=n // 2).apply(
        lambda x: float(np.sum(x <= x[-1]) / len(x)), raw=True
    )


def InstZScore(n: int, s: pd.Series, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score: (s - mean) / std over the last n bars."""
    mu  = InstMean(n, s, min_periods=min_periods)
    std = InstStdev(n, s, min_periods=min_periods).replace(0, np.nan)
    return (s - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
# Difference / returns (explicit — no active-asset context needed)
# ─────────────────────────────────────────────────────────────────────────────

def Diff(n: int, s: pd.Series) -> pd.Series:
    """s[t] - s[t-n]."""
    return s.diff(n)


def PctChange(n: int, s: pd.Series) -> pd.Series:
    """(s[t] - s[t-n]) / s[t-n]."""
    return s.pct_change(n)


# ─────────────────────────────────────────────────────────────────────────────
# Event study
# ─────────────────────────────────────────────────────────────────────────────

def _taifex_expiry_dates(trading_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Third Wednesday of every month, snapped to the nearest trading day."""
    cur    = trading_index[0].replace(day=1)
    end    = trading_index[-1]
    dates  = []
    while cur <= end:
        days_to_wed = (2 - cur.weekday()) % 7
        third_wed   = cur + pd.Timedelta(days=days_to_wed) + pd.Timedelta(weeks=2)
        if third_wed <= end:
            pos = trading_index.searchsorted(third_wed, side="right") - 1
            if pos >= 0:
                dates.append(trading_index[pos])
        cur = (cur + pd.DateOffset(months=1)).replace(day=1)
    return pd.DatetimeIndex(sorted(set(dates)))


def Event(name: str) -> pd.Series:
    """
    Signed trading-day distance to the nearest recurring event.

    Parameters
    ----------
    name : event type. Currently supported:
        'optexp' — TAIFEX monthly option/futures expiry (third Wednesday)

    Returns
    -------
    pd.Series of signed integers indexed by the active asset's trading calendar:
         0  = expiry day itself
        -N  = N trading days BEFORE the nearest upcoming expiry
        +N  = N trading days AFTER the most recent past expiry

    Example
    -------
        cta.load_asset('mtx', '1d')
        ev = cta.Event('optexp')
        # ev looks like: ..., -3, -2, -1, 0, 1, 2, 3, ..., 10, ..., -3, -2, -1, 0, ...
    """
    asset         = _require_asset()
    trading_index = asset.index

    if name == "optexp":
        expiry_dates = _taifex_expiry_dates(trading_index)
        exp_pos = np.sort(np.array([
            trading_index.searchsorted(ed, side="right") - 1
            for ed in expiry_dates
        ]))

        result = np.empty(len(trading_index), dtype=int)
        for i in range(len(trading_index)):
            ins  = np.searchsorted(exp_pos, i)
            best = None
            if ins > 0:
                d    = i - exp_pos[ins - 1]          # positive: days after prev expiry
                best = d
            if ins < len(exp_pos):
                d    = i - exp_pos[ins]              # negative (or 0): days before next
                if best is None or abs(d) < abs(best):
                    best = d
            result[i] = best if best is not None else 0

        return pd.Series(result, index=trading_index, name="event_optexp")

    raise ValueError(f"Unknown event name '{name}'. Supported: 'optexp'")


def Caar(
    window: int,
    returns: pd.Series,
    event_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Cumulative Average Abnormal Return around event dates.

    Mirrors f.Caar(window, returns).

    Parameters
    ----------
    window      : bars on each side of the event (total window = 2*window+1)
    returns     : daily return / PnL series, date-indexed
                  e.g. Lag(2, signal) * Returns('c', -1)
    event_dates : dates to centre on; defaults to TAIFEX monthly expiry dates
                  (third Wednesday of each month) from the series' own index

    Returns
    -------
    pd.DataFrame indexed by relative day (-window … +window) with columns:
        casr  — cumulative average return
        mean  — average return per bar
        se    — standard error of the mean per bar
        n     — number of events used
    """
    trading_index = returns.index

    if event_dates is None:
        event_dates = _taifex_expiry_dates(trading_index)

    ret_vals = returns.fillna(0.0).values
    slices   = []
    for ed in event_dates:
        pos = trading_index.searchsorted(ed, side="right") - 1
        if pos < window or pos + window >= len(trading_index):
            continue
        slices.append(ret_vals[pos - window : pos + window + 1])

    if not slices:
        raise ValueError("No valid event windows found in the series.")

    mat  = np.array(slices)                           # (n_events, 2*window+1)
    avg  = mat.mean(axis=0)
    se   = mat.std(axis=0, ddof=1) / np.sqrt(len(slices))

    return pd.DataFrame(
        {"casr": np.cumsum(avg), "mean": avg, "se": se, "n": len(slices)},
        index=np.arange(-window, window + 1),
    )


__all__ = [
    "set_active_asset",
    "Prices", "Returns",
    "Lag", "Lead",
    "ForwardFill",
    "InstMean", "InstStdev", "InstSkew", "InstSum",
    "InstRank", "InstZScore",
    "Diff", "PctChange",
    "Event", "Caar",
]
