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
# Filter / mask
# ─────────────────────────────────────────────────────────────────────────────

def Filter(signal: pd.Series, mask) -> pd.Series:
    """
    Filter a signal by a 0/1 or boolean mask, mirroring f-package
    `BaseMatrix / mask` semantics.

        signal / mask  →  signal where mask is truthy, NaN elsewhere

    Internally: any 0s in the mask are first replaced with NaN, then
    we divide so that filtered-out positions become NaN (not inf, not 0).
    NaN means "no position" downstream — `exec_sig.shift(2) * returns`
    propagates the NaN, and `pnl.cumsum()` skips it.

    Parameters
    ----------
    signal : pd.Series
        Date-indexed signal series.
    mask : pd.Series | array-like
        Same length / index as signal. Truthy (1, True) keeps the signal,
        falsy (0, False) drops it.

    Examples
    --------
        # Only trade when realised vol is in the top quintile
        vol      = cta.InstStdev(20, cta.Returns('c', -1))
        high_vol = vol > vol.rolling(252).quantile(0.8)
        sig_hv   = cta.Filter(my_sig, high_vol)

        # Equivalent inline form using the / operator on pandas:
        sig_hv   = my_sig / high_vol.astype(float).replace(0, np.nan)
    """
    if isinstance(mask, pd.Series):
        m = mask.reindex(signal.index).astype(float)
    else:
        m = pd.Series(mask, index=signal.index).astype(float)
    m = m.where(m != 0)                                  # zeros → NaN
    return (signal / m).replace([np.inf, -np.inf], np.nan)


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

def load_tsmc_ea_dates(
    path,
    trading_index: pd.DatetimeIndex,
    midnight_to_next_day: bool = True,
) -> pd.DatetimeIndex:
    """
    Load TSMC earnings-announcement / investor-relations event dates from the
    TWSE-format CSV `tsmc_ea.csv` and snap them to trading days.

    The CSV is Big5-encoded with ROC dates (e.g. '115/04/25' = 2026-04-25).
    Column layout (zero-indexed):
        0 公司代號     1 公司名稱   2 召開法人說明會日期 (date)
        3 召開法人說明會時間 (time)  4 召開法人說明會地點    ...

    Date handling
    -------------
    * ROC year + 1911 → Gregorian year.
    * Date ranges like '115/03/03 至 115/03/06' use the **start** date.
    * If `midnight_to_next_day` and the time field begins with '00:00',
      the event is shifted forward by one calendar day. This implements
      the convention "midnight events map to the next day", on the
      assumption that 00:00 entries represent overnight events whose
      market impact lands on the next session.
    * Each event is then snapped *forward* to the next available trading
      day in `trading_index` (skipping weekends / holidays).
    * Events outside `trading_index` are dropped.

    Returns
    -------
    pd.DatetimeIndex of unique, sorted trading-day-aligned event dates.
    """
    import csv
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSMC EA CSV not found: {path}")

    if trading_index is None or len(trading_index) == 0:
        return pd.DatetimeIndex([])

    events: list[pd.Timestamp] = []
    with open(path, encoding="big5", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)            # skip header
        except StopIteration:
            return pd.DatetimeIndex([])
        for row in reader:
            if len(row) < 4:
                continue
            date_field = row[2].strip()
            time_field = row[3].strip()
            if not date_field:
                continue

            # Range like "115/03/03 至 115/03/06" → take the start date.
            date_str = date_field.split()[0]

            parts = date_str.split("/")
            if len(parts) != 3:
                continue
            try:
                year_roc, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            try:
                event_date = pd.Timestamp(year=year_roc + 1911, month=month, day=day)
            except (ValueError, OverflowError):
                continue

            if midnight_to_next_day and time_field.startswith("00:00"):
                event_date = event_date + pd.Timedelta(days=1)

            events.append(event_date)

    if not events:
        return pd.DatetimeIndex([])

    events_idx = pd.DatetimeIndex(sorted(set(events)))

    # Snap each event forward to the next available trading day.
    snapped: list[pd.Timestamp] = []
    last_pos = len(trading_index)
    for ev in events_idx:
        pos = trading_index.searchsorted(ev, side="left")
        if pos < last_pos:
            snapped.append(trading_index[pos])

    return pd.DatetimeIndex(sorted(set(snapped)))


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


def _event_positions(event_dates: pd.DatetimeIndex,
                     trading_index: pd.DatetimeIndex) -> np.ndarray:
    """Integer positions of event dates within `trading_index` (sorted, deduped)."""
    if len(event_dates) == 0:
        return np.array([], dtype=int)
    positions = np.array([
        trading_index.searchsorted(ed, side="right") - 1
        for ed in event_dates
    ])
    positions = positions[(positions >= 0) & (positions < len(trading_index))]
    return np.sort(np.unique(positions))


def _signed_event_distance(trading_index: pd.DatetimeIndex,
                           positions:     np.ndarray) -> np.ndarray:
    """
    Signed trading-day distance from each bar to the nearest event position.

    Convention:
        0  = on an event day
        -N = N trading days BEFORE the nearest upcoming event
        +N = N trading days AFTER the most recent past event
    """
    n   = len(trading_index)
    out = np.zeros(n, dtype=int)
    if len(positions) == 0:
        return out
    for i in range(n):
        ins  = np.searchsorted(positions, i)
        best = None
        if ins > 0:
            d    = i - positions[ins - 1]           # positive: days after prev event
            best = d
        if ins < len(positions):
            d    = i - positions[ins]               # negative (or 0): days before next
            if best is None or abs(d) < abs(best):
                best = d
        out[i] = best if best is not None else 0
    return out


def Event(name: str) -> pd.Series:
    """
    Signed trading-day distance to the nearest event of type `name`.

    Parameters
    ----------
    name : event type. Currently supported:
        'optexp'   — TAIFEX monthly futures/options expiry (3rd Wednesday).
        'tsmc_ea'  — TSMC earnings-announcement / investor-relations events,
                     loaded from the repo's `tsmc_ea.csv` with midnight events
                     mapped to the next day and snapped to trading days.

    Returns
    -------
    pd.Series of signed integers indexed by the active asset's trading calendar:
         0  = on an event day
        -N  = N trading days BEFORE the nearest upcoming event
        +N  = N trading days AFTER the most recent past event

    Example
    -------
        cta.load_asset('mtx', '1d')
        ev_exp = cta.Event('optexp')
        ev_ea  = cta.Event('tsmc_ea')
        # near a TSMC EA: ..., -3, -2, -1, 0, 1, 2, 3, ..., -3, -2, -1, 0, ...
    """
    asset         = _require_asset()
    trading_index = asset.index

    if name == "optexp":
        event_dates = _taifex_expiry_dates(trading_index)
        positions   = _event_positions(event_dates, trading_index)
        result      = _signed_event_distance(trading_index, positions)
        return pd.Series(result, index=trading_index, name="event_optexp")

    if name == "tsmc_ea":
        from pathlib import Path
        default_path = Path(__file__).parent.parent / "tsmc_ea.csv"
        if not default_path.exists():
            raise FileNotFoundError(
                f"TSMC EA CSV not found at {default_path}. "
                "Use cta.load_tsmc_ea_dates(path, trading_index) directly "
                "if your file lives elsewhere."
            )
        event_dates = load_tsmc_ea_dates(default_path, trading_index)
        positions   = _event_positions(event_dates, trading_index)
        result      = _signed_event_distance(trading_index, positions)
        return pd.Series(result, index=trading_index, name="event_tsmc_ea")

    raise ValueError(f"Unknown event name '{name}'. Supported: 'optexp', 'tsmc_ea'")


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
    "Filter",
    "ForwardFill",
    "InstMean", "InstStdev", "InstSkew", "InstSum",
    "InstRank", "InstZScore",
    "Diff", "PctChange",
    "Event", "Caar",
    "load_tsmc_ea_dates",
]
