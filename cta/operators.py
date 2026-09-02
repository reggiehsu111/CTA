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


def InstCorr(
    n: int,
    s1: pd.Series,
    s2: pd.Series,
    method: str = "pearson",
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling correlation between two series over the last n bars.

    Useful as a cross-feature signal — e.g. correlation of `volume` with
    past close-to-close returns measures whether large bars are bought up
    (positive ρ) or sold into (negative ρ).

    Parameters
    ----------
    n           : window length in bars.
    s1, s2      : two date-indexed pandas Series. Auto-aligned to their
                  intersection.
    method      : 'pearson' (default, linear)  — fast, uses pandas rolling.corr.
                  'spearman' (rank-based)     — robust to outliers and monotone
                                                  non-linearity. Vectorised with
                                                  `sliding_window_view`, chunked
                                                  to keep peak memory bounded.
    min_periods : minimum window size for a non-NaN result. Default n // 2.

    Notes
    -----
    The Spearman implementation uses within-window ranks. NaN values are
    placed at the highest ranks by `argsort`; pre-clean inputs if that's
    a concern.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")

    mp = min_periods if min_periods is not None else max(3, n // 2)
    s1, s2 = s1.align(s2, join="inner")

    if method == "pearson":
        return s1.rolling(n, min_periods=mp).corr(s2).rename(f"corr_{n}")

    # ── Spearman: within-window ranks → Pearson on ranks ─────────────────
    from numpy.lib.stride_tricks import sliding_window_view

    arr1 = s1.values.astype(np.float64)
    arr2 = s2.values.astype(np.float64)
    N    = len(arr1)
    out  = np.full(N, np.nan, dtype=np.float64)

    if N < n:
        return pd.Series(out, index=s1.index, name=f"corr_{n}")

    CHUNK = 100_000
    for out_start in range(n - 1, N, CHUNK):
        out_end   = min(out_start + CHUNK, N)
        inp_start = out_start - n + 1
        a1 = arr1[inp_start:out_end]
        a2 = arr2[inp_start:out_end]

        w1 = sliding_window_view(a1, n)
        w2 = sliding_window_view(a2, n)

        # Within-window ranks; .astype(float32) halves the memory footprint
        r1 = np.argsort(np.argsort(w1, axis=1), axis=1).astype(np.float32)
        r2 = np.argsort(np.argsort(w2, axis=1), axis=1).astype(np.float32)

        m1 = r1.mean(axis=1, keepdims=True)
        m2 = r2.mean(axis=1, keepdims=True)
        num = ((r1 - m1) * (r2 - m2)).sum(axis=1)
        den = np.sqrt(((r1 - m1) ** 2).sum(axis=1) * ((r2 - m2) ** 2).sum(axis=1))
        out[out_start:out_end] = np.where(den > 0, num / den, np.nan)

    return pd.Series(out, index=s1.index, name=f"corr_{n}")


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


def Sign(s: pd.Series) -> pd.Series:
    """Sign of each element; preserves index/name. 0 stays 0, NaN stays NaN."""
    return pd.Series(np.sign(s.values), index=s.index, name=getattr(s, "name", None))


def Abs(s: pd.Series) -> pd.Series:
    """Absolute value, index/name preserved."""
    return s.abs()


def Date(field: str) -> pd.Series:
    """Return a date/time-of-index feature from the active asset as a pd.Series.

    Convenient for building calendar-aware filters, e.g.
        mask = (cta.Date('dom') < 17) | (cta.Date('dom') > 23)
        filtered = cta.Filter(signal, mask)

    Fields
    ------
    'dom' | 'day'       : day of month (1..31)
    'dow' | 'weekday'   : day of week (Mon=0 .. Sun=6)
    'month'             : month (1..12)
    'year'              : four-digit year
    'quarter'           : 1..4
    'week' | 'isoweek'  : ISO week number (1..53)
    'doy'               : day of year (1..366)
    'hour'              : 0..23 (intraday)
    'minute'            : 0..59 (intraday)
    'time_of_day'       : minute since midnight (intraday)
    """
    asset = _require_asset()
    idx = asset.index
    key = field.lower()
    if key in ("dom", "day"):     out = idx.day
    elif key in ("dow", "weekday"): out = idx.dayofweek
    elif key == "month":          out = idx.month
    elif key == "year":           out = idx.year
    elif key == "quarter":        out = idx.quarter
    elif key in ("week", "isoweek"): out = idx.isocalendar().week
    elif key == "doy":            out = idx.dayofyear
    elif key == "hour":           out = idx.hour
    elif key == "minute":         out = idx.minute
    elif key == "time_of_day":    out = idx.hour * 60 + idx.minute
    else:
        raise ValueError(
            f"Unknown date field {field!r}. Choose from: "
            "dom, dow, month, year, quarter, week, doy, hour, minute, time_of_day"
        )
    return pd.Series(np.asarray(out), index=idx, name=f"date_{key}")


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


_TSMC_CATEGORY_ALIASES = {
    "tsmc_quarterly_call":    "quarterly_call",
    "tsmc_earnings":          "quarterly_call",    # convenience alias
    "tsmc_investor_day":      "investor_day",
    "tsmc_broker_conference": "broker_conference",
    "tsmc_broker":            "broker_conference", # convenience alias
    "tsmc_other":             "other",
}

# expiry aliases
_EXPIRY_ALIASES = {"optexp", "futures_expire", "futures_expiry", "expiry"}


def Event(name: str) -> pd.Series:
    """
    Signed trading-day distance to the nearest event of type `name`.

    Parameters
    ----------
    name : event type. Supported:
        'optexp' / 'futures_expire'
                        — TAIFEX monthly futures/options expiry (3rd Wednesday).
        'tsmc_ea'       — ALL TSMC investor-relations events lumped together
                          (back-compat with the original `load_tsmc_ea_dates`).
        'tsmc_quarterly_call' / 'tsmc_earnings'
                        — the 4x/year 法說會 official earnings announcement (~86 events).
        'tsmc_investor_day'
                        — TSMC-hosted capital-markets days (Investor Day, ~4 events).
        'tsmc_broker_conference' / 'tsmc_broker'
                        — sell-side broker/bank conference re-broadcast (~116 events).
        'tsmc_other'    — unclassified TSMC IR rows.

    Returns
    -------
    pd.Series of signed integers indexed by the active asset's trading calendar:
         0  = on an event day
        -N  = N trading days BEFORE the nearest upcoming event
        +N  = N trading days AFTER the most recent past event

    Example
    -------
        cta.load_asset('mtx', '1d')
        ev_exp = cta.Event('futures_expire')
        ev_qc  = cta.Event('tsmc_quarterly_call')
        ev_id  = cta.Event('tsmc_investor_day')
    """
    asset         = _require_asset()
    trading_index = asset.index

    if name in _EXPIRY_ALIASES:
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

    if name in _TSMC_CATEGORY_ALIASES:
        from .tsmc_events import load_tsmc_event_dates
        category    = _TSMC_CATEGORY_ALIASES[name]
        event_dates = load_tsmc_event_dates(category, trading_index)
        positions   = _event_positions(event_dates, trading_index)
        result      = _signed_event_distance(trading_index, positions)
        return pd.Series(result, index=trading_index, name=f"event_{name}")

    supported = ["optexp", "futures_expire", "tsmc_ea"] + list(_TSMC_CATEGORY_ALIASES)
    raise ValueError(f"Unknown event name '{name}'. Supported: {supported}")


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


def _events_to_positions(events: pd.Series, target_index: pd.Index) -> np.ndarray:
    """Convert an event mask (0/1/bool/NaN) aligned to `target_index` → int positions where mask==1."""
    ev = events.reindex(target_index)
    if ev.dtype == bool:
        return np.where(ev.fillna(False).values)[0]
    return np.where(np.isclose(ev.fillna(0).astype(float).values, 1.0))[0]


def EventFFill(
    feature: pd.Series,
    events: pd.Series,
    offset: int = 0,
    limit: int = 5,
) -> pd.Series:
    """
    Form-1 event signal: sample `feature` at each event bar (optionally shifted
    by `offset`) and forward-fill the value for up to `limit` bars.

    Every event contributes ONE sample; the value persists for at most `limit`
    trading days after being set and resets on the next event. Between events
    (or beyond the fill window) the signal is NaN, so downstream code sees an
    explicit "not active" state instead of stale data.

    Parameters
    ----------
    feature  : the base feature series (date-indexed on active asset).
    events   : mask series aligned to feature, 0/1/NaN or booleans. 1 = event.
    offset   : sample-bar offset from the event date. Examples —
                 0  → sample ON the event day  (Form 1 baseline)
                -1  → sample 1 day BEFORE the event
                +1  → sample 1 day AFTER  the event
                +K  → sample K days AFTER  the event
    limit    : forward-fill horizon (trading days). 5 = signal is live for
               the first 5 days after being sampled, then NaN.

    Returns
    -------
    pd.Series indexed like `feature`.

    Example
    -------
        ev_qc = cta.Event('tsmc_quarterly_call')
        # daily-return on event day, held for 5 trading days
        sig = cta.EventFFill(cta.Returns('c', -1), ev_qc == 0, offset=0, limit=5)
    """
    feature = feature.astype(float)
    idx     = feature.index
    positions = _events_to_positions(events, idx)
    if len(positions) == 0:
        return pd.Series(np.nan, index=idx, name=f"eventffill_o{offset}_l{limit}")

    shifted = positions + int(offset)
    shifted = shifted[(shifted >= 0) & (shifted < len(idx))]
    if len(shifted) == 0:
        return pd.Series(np.nan, index=idx, name=f"eventffill_o{offset}_l{limit}")

    out = pd.Series(np.nan, index=idx)
    out.iloc[shifted] = feature.values[shifted]
    return out.ffill(limit=limit).rename(f"eventffill_o{offset}_l{limit}")


def EventRollingFFill(
    feature: pd.Series,
    events: pd.Series,
    window: int,
    offset: int = 0,
    limit: int = 5,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Form-2 / Form-3 event signal: sample `InstMean(window, feature)` at each
    event bar (optionally shifted by `offset`), forward-fill for up to `limit`
    bars, reset on the next event.

    Choosing `offset` decides which rolling window sits inside the event
    neighborhood:
        offset =  0            → mean over feature[event-window+1 .. event]
                                 (Form 2: rolling-mean AT event)
        offset =  window       → mean over feature[event+1  .. event+window]
                                 (Form 3: mean over the `window` post-event bars,
                                  sampled the first day after the window closes)
        offset =  window - 1   → mean over feature[event    .. event+window-1]
                                 (post-event including event day itself)
        offset = -1            → mean over feature[event-window .. event-1]
                                 (mean of pre-event bars, sampled the day before)

    All parameters otherwise mirror `EventFFill`.

    Example
    -------
        ev_qc = cta.Event('tsmc_quarterly_call')
        # Form 2: 5-day trailing mean sampled ON event, held for 5 days
        sig2 = cta.EventRollingFFill(feature, ev_qc==0, window=5,
                                      offset=0,  limit=5)
        # Form 3: 5-day POST-event mean, sampled at event+5, held for 5 days
        sig3 = cta.EventRollingFFill(feature, ev_qc==0, window=5,
                                      offset=5,  limit=5)
    """
    feature = feature.astype(float)
    idx     = feature.index
    mp      = min_periods if min_periods is not None else max(1, window // 2)
    rolled  = feature.rolling(window, min_periods=mp).mean()

    positions = _events_to_positions(events, idx)
    if len(positions) == 0:
        return pd.Series(np.nan, index=idx,
                         name=f"eventroll_w{window}_o{offset}_l{limit}")

    shifted = positions + int(offset)
    shifted = shifted[(shifted >= 0) & (shifted < len(idx))]
    if len(shifted) == 0:
        return pd.Series(np.nan, index=idx,
                         name=f"eventroll_w{window}_o{offset}_l{limit}")

    out = pd.Series(np.nan, index=idx)
    out.iloc[shifted] = rolled.values[shifted]
    return out.ffill(limit=limit).rename(
        f"eventroll_w{window}_o{offset}_l{limit}"
    )


def Casr(
    days: int,
    signal: pd.Series,
    events: pd.Series,
    *,
    show: bool = True,
    figsize: tuple[float, float] = (13.0, 4.6),
    exec_lag: int = 2,
    title: str | None = None,
):
    """
    Plot two subplots for an event study around user-supplied event centers.

    Left  : average signal LEVEL from t=-days..+days, ±1 SE band.
    Right : CASR — cumulative average signal-return around the same events,
            ±1 SE band. Signal-return = signal.shift(exec_lag) * asset.returns.

    Parameters
    ----------
    days   : half-window in trading days. Total window = 2*days + 1.
    signal : the signal series (date-indexed). Its index defines the trading
             calendar for the event lookup.
    events : mask series aligned to `signal`, containing 0/1/NaN (or booleans).
             Every bar where the mask == 1 is treated as an event center.
             Example — study the day BEFORE a quarterly call:
                 ev_qc = cta.Event('tsmc_quarterly_call')
                 cta.Casr(10, signal, ev_qc == -1)
             The full ±days window is averaged around each 1 in the mask,
             independent of how close other 1s are (nearby events overlap
             naturally in the averaging).
    show   : if True, call `plt.show()` at the end; if False, return the
             figure without displaying so the caller can compose further.
    exec_lag : lag applied to the signal for the CASR (right) panel.
               Default 2 matches `cta.Simulate` — signal at day D drives
               PnL on D+2. Set to 0 to disable.
    title  : optional overall figure title.

    Returns
    -------
    dict with keys:
        'signal_avg'  DataFrame(index=[-days..+days], cols=['mean','se','n'])
        'casr'        DataFrame(index=[-days..+days], cols=['casr','mean','se','n'])
        'fig'         matplotlib Figure
        'axes'        (ax_left, ax_right)
    """
    import matplotlib.pyplot as plt

    signal = signal.astype(float)
    trading_index = signal.index

    # Align events to the signal calendar and pick event positions.
    ev = events.reindex(trading_index)
    if ev.dtype == bool:
        ev_positions = np.where(ev.fillna(False).values)[0]
    else:
        ev_positions = np.where(np.isclose(ev.fillna(0).astype(float).values, 1.0))[0]

    if len(ev_positions) == 0:
        raise ValueError("`events` mask has no True (==1) entries.")

    n_bars = len(trading_index)
    valid_positions = ev_positions[(ev_positions >= days) &
                                    (ev_positions + days < n_bars)]
    if len(valid_positions) == 0:
        raise ValueError(
            f"None of the {len(ev_positions)} events have a full ±{days}-day "
            "window inside the signal's index."
        )

    # ── Left panel data: signal level averaged over -days..+days ─────────
    sig_vals = signal.values                                 # float, NaN preserved
    sig_mat  = np.full((len(valid_positions), 2 * days + 1), np.nan)
    for i, pos in enumerate(valid_positions):
        sig_mat[i, :] = sig_vals[pos - days: pos + days + 1]

    import warnings as _w
    with np.errstate(invalid="ignore"), _w.catch_warnings():
        _w.filterwarnings("ignore", message="Mean of empty slice")
        _w.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        sig_mean = np.nanmean(sig_mat, axis=0)
        sig_std  = np.nanstd (sig_mat, axis=0, ddof=1)
        sig_n    = np.sum(~np.isnan(sig_mat), axis=0)
    sig_se = sig_std / np.sqrt(np.maximum(sig_n, 1))

    x = np.arange(-days, days + 1)
    signal_avg = pd.DataFrame(
        {"mean": sig_mean, "se": sig_se, "n": sig_n}, index=x
    )

    # ── Right panel data: CASR of signal-return around events ─────────────
    asset      = _require_asset()
    asset_ret  = asset.returns.reindex(trading_index)
    exec_sig   = signal.shift(exec_lag) if exec_lag else signal
    sig_pnl    = (exec_sig * asset_ret).reindex(trading_index)

    event_dates = trading_index[valid_positions]
    casr_df     = Caar(days, sig_pnl, event_dates=event_dates)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

    axL.plot(x, sig_mean, color="#1565c0", lw=1.5, marker="o", ms=4,
             label=f"avg signal  (n={len(valid_positions)} events)")
    axL.fill_between(x, sig_mean - sig_se, sig_mean + sig_se,
                     color="#1565c0", alpha=0.20, label="±1 SE")
    axL.axvline(0, color="#c62828", lw=1.0, ls="--", alpha=0.75, label="event day")
    axL.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    axL.set_xlabel("Days relative to event")
    axL.set_ylabel("Average signal value")
    axL.set_title(f"Signal level  (±{days}d around event)")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)

    x2       = casr_df.index.values
    casr     = casr_df["casr"].values
    cum_lo   = np.cumsum(casr_df["mean"].values - casr_df["se"].values)
    cum_hi   = np.cumsum(casr_df["mean"].values + casr_df["se"].values)
    n_events = int(casr_df["n"].iloc[0])

    axR.plot(x2, casr, color="#e65100", lw=1.4, marker="o", ms=4,
             label=f"signal CASR  (n={n_events})")
    axR.fill_between(x2, cum_lo, cum_hi, alpha=0.20, color="#e65100",
                     label="±1 SE")
    axR.axvline(0, color="#c62828", lw=1.0, ls="--", alpha=0.75, label="event day")
    axR.axhline(0, color="black", lw=0.6, ls="--", alpha=0.5)
    axR.set_xlabel("Days relative to event")
    axR.set_ylabel("Cumulative avg return")
    axR.set_title(f"CASR  (±{days}d around event, exec_lag={exec_lag})")
    axR.legend(fontsize=8); axR.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    if show:
        plt.show()

    return {
        "signal_avg": signal_avg,
        "casr": casr_df,
        "fig": fig,
        "axes": (axL, axR),
    }


__all__ = [
    "set_active_asset",
    "Prices", "Returns",
    "Lag", "Lead",
    "Filter",
    "ForwardFill",
    "InstMean", "InstStdev", "InstSkew", "InstSum", "InstCorr",
    "InstRank", "InstZScore",
    "Diff", "PctChange",
    "Sign", "Abs", "Date",
    "Event", "Caar", "Casr",
    "EventFFill", "EventRollingFFill",
    "load_tsmc_ea_dates",
]
