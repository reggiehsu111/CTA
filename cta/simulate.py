"""
simulate.py — single-asset CTA simulation for MTX futures.

Entry point:
    simulate(sig, time_granularity='1d', asset='mtx', normalize=True)

P&L model:
    exec_sig[t]  = lag(2, signal[t])          # 2-bar execution lag
    pnl[t]       = exec_sig[t] * returns[t]   # returns = close.pct_change()

Normalization
-------------
When normalize=True the raw signal is passed through a rolling z-score → tanh
pipeline so that every value lands in (-1, 1).

    z[t]      = (sig[t] - μ[t]) / σ[t]   (rolling, no look-ahead)
    normed[t] = tanh(z[t] * sensitivity)

Pass normalize='rank' to use a rolling percentile-rank approach instead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .asset import BaseAsset
from . import operators as _ops


# ─────────────────────────────────────────────────────────────────────────────
# Global date-range state — used by Simulate / SimulateAll / simulate_by_dollars
# / simulate_midday_short_night_long when their own start_date/end_date args
# are left as None. Set once at the top of a notebook for train/test splits.
# ─────────────────────────────────────────────────────────────────────────────

_global_date_range: dict = {"start": None, "end": None}


def set_date_range(start_date=None, end_date=None) -> None:
    """
    Set the default evaluation window used by every Simulate / SimulateAll /
    simulate_by_dollars / simulate_midday_short_night_long call that does not
    explicitly pass `start_date` / `end_date`.

    Pass `None` for either bound to mean 'no limit on that side'.

        cta.set_date_range(start_date="2017-01-01", end_date="2022-12-31")
        cta.Simulate(sig)        # train window
        cta.set_date_range(start_date="2023-01-01", end_date=None)
        cta.Simulate(sig)        # test window
    """
    _global_date_range["start"] = pd.Timestamp(start_date) if start_date is not None else None
    _global_date_range["end"]   = pd.Timestamp(end_date)   if end_date   is not None else None


def get_date_range() -> tuple:
    """Return (start, end) — the globally set defaults."""
    return _global_date_range["start"], _global_date_range["end"]


def _resolve_dates(start_date, end_date) -> tuple:
    """Explicit args win; fall back to global state. Both default to None."""
    start = start_date if start_date is not None else _global_date_range["start"]
    end   = end_date   if end_date   is not None else _global_date_range["end"]
    if start is not None:
        start = pd.Timestamp(start)
    if end is not None:
        end = pd.Timestamp(end)
    return start, end


def _subset_asset(asset_obj: BaseAsset, eval_idx: pd.DatetimeIndex) -> BaseAsset:
    """Return a new BaseAsset restricted to `eval_idx`, preserving metadata."""
    df = pd.DataFrame(asset_obj).loc[eval_idx]
    return BaseAsset.from_df(
        df, symbol=asset_obj.symbol, time_granularity=asset_obj.time_granularity,
    )


def _slice_index(full_idx: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    """Return the subset of full_idx within [start, end] (inclusive). Either may be None."""
    idx = full_idx
    if start is not None:
        idx = idx[idx >= start]
    if end is not None:
        idx = idx[idx <= end]
    return idx

# Module-level matplotlib state mutations have been removed — they were
# being re-applied on every importlib.reload(cta.simulate), which corrupted
# the Jupyter inline backend across cells.
#
# Set the CJK font in your notebook once if needed:
#     import matplotlib as mpl
#     mpl.rcParams["font.family"] = ["Hiragino Sans GB", "Heiti TC",
#                                    "Songti SC", "PingFang HK",
#                                    "Arial Unicode MS", "DejaVu Sans"]


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_tanh(
    sig: pd.Series,
    window: int = 252,
    sensitivity: float = 1.0,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(2, window // 4)
    mu  = sig.rolling(window, min_periods=min_periods).mean()
    std = sig.rolling(window, min_periods=min_periods).std(ddof=1)
    z   = (sig - mu) / std.replace(0, np.nan)
    return np.tanh(z * sensitivity).rename(sig.name)


def _normalize_rank(
    sig: pd.Series,
    window: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    if min_periods is None:
        min_periods = max(2, window // 4)

    def _rank_last(arr):
        return float(pd.Series(arr).rank(pct=True).iloc[-1])

    pct = sig.rolling(window, min_periods=min_periods).apply(_rank_last, raw=False)
    return (pct * 2 - 1).rename(sig.name)


def normalize_signal(
    sig: pd.Series,
    method: str = "tanh",
    window: int = 252,
    sensitivity: float = 1.0,
    force: bool = True,        # deprecated — kept for back-compat, always True now
) -> pd.Series:
    """
    Normalize a signal to (-1, 1) without look-ahead.

    Applies `method` unconditionally — no idempotence guard. Callers who
    don't want re-normalization should pass their signal through `Simulate`
    with `normalize=False` (the pipeline flag), rather than relying on this
    helper to detect already-normalized inputs.

    Parameters
    ----------
    sig         : signal, arbitrary scale, date-indexed
    method      : 'tanh' (default) or 'rank'
    window      : rolling look-back window in bars
    sensitivity : tanh only — controls saturation speed (default 1.0)
    force       : deprecated — no-op. Behaviour is now equivalent to force=True.
    """
    sig = sig.astype(float)
    if method == "tanh":
        return _normalize_tanh(sig, window=window, sensitivity=sensitivity)
    if method == "rank":
        return _normalize_rank(sig, window=window)
    raise ValueError(f"Unknown normalization method '{method}'. Choose 'tanh' or 'rank'.")


# ─────────────────────────────────────────────────────────────────────────────
# Asset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_asset(
    asset: str = "mtx",
    time_granularity: str = "1d",
    contract: str = "front",
) -> BaseAsset:
    """
    Load any TAIFEX futures contract, set it as the active context, and return it.

    Parameters
    ----------
    asset            : contract code, e.g. 'MTX', 'TX', 'GDF', 'TMF' (case-insensitive).
                       Call cta.available_assets() to list all available codes.
    time_granularity : '1d'
    contract         : 'front' (近月, default) or 'back' (次月)

    Call this before building signals with Prices() / Returns() / etc.

        cta.load_asset('TX', '1d')
        sig = cta.InstZScore(252, cta.Returns('c', -20))
        cta.SimulateAll(sig, normalize=False)
    """
    asset_obj = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj)
    return asset_obj


_HISTORY_DIR = None   # resolved lazily


def _history_dir() -> "Path":
    from pathlib import Path
    global _HISTORY_DIR
    if _HISTORY_DIR is None:
        _HISTORY_DIR = Path(__file__).parent.parent / "history_data"
    return _HISTORY_DIR


def available_assets() -> list[str]:
    """Return all contract codes that have a history CSV in history_data/."""
    return sorted(
        p.stem.replace("_history", "").upper()
        for p in _history_dir().glob("*_history.csv")
    )


_INTRADAY_GRANULARITIES = {"1m", "5m", "15m", "30m", "1h"}


def _resample_rule(granularity: str) -> str:
    # Lowercase 'h' / 'min' required by recent pandas.
    return {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}[granularity]


def _load_intraday(asset: str, time_granularity: str) -> BaseAsset:
    """Load intraday bars for `asset`, unioned across every source we have.

    Delegates to cta.bars.load_1min, which merges the legacy MXFR1k.csv with the
    Shioaji parquet (local and S3). This used to read MXFR1k.csv alone, which meant
    MXF only and nothing after 2026-05-22 — the file's last bar — so any intraday
    backtest silently stopped there while the daily series ran to today.

    Coarser granularities resample from 1m. The 13:45 -> 15:00 break produces empty
    buckets rather than bogus bars, and those are dropped.
    """
    from .bars import load_1min, resolve_symbol

    sym = resolve_symbol(asset)
    df = load_1min(sym)

    if time_granularity != "1m":
        rule = _resample_rule(time_granularity)
        df = (df.resample(rule)
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna(subset=["close"]))

    return BaseAsset.from_df(df, symbol=asset.upper(), time_granularity=time_granularity)


# ─── RDS-backed 1d loader for TAIFEX index futures ────────────────────────
# The Lambda ingest lambdas keep tw_index_futures_pv fresh through the day.
# Reading from RDS here means every runner invocation sees the newest bar
# without a Lambda zip rebuild.
_RDS_INDEX_FUTURES_TICKERS = frozenset({"MTX", "TX", "TE", "TF"})


class _RdsEmptyError(RuntimeError):
    """Raised when tw_index_futures_pv returns no rows for the ticker."""


def _load_asset_from_rds(asset: str, contract: str = "front") -> BaseAsset:
    """Build a 1d BaseAsset for a TAIFEX index-future ticker from RDS.

    Mirrors the CSV loader's output columns exactly (front-month selection
    by monthly-expiry rank, back-month close for rollover, night-session
    columns merged onto the same date).
    """
    import os as _os
    import sys as _sys

    # Resolve QuantResearch/Libs across machines. The relative candidate covers
    # both layouts (Mac: Research/{mtx,QuantResearch}; EC2: ~/{mtx,QuantResearch}),
    # since this file sits three levels below that shared root.
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _candidates = [
        _os.environ.get("QUANTRESEARCH_LIBS"),
        _os.path.join(_os.path.dirname(_os.path.dirname(_here)), "QuantResearch", "Libs"),
        "/home/ubuntu/QuantResearch/Libs",
        "/Users/hsureggie/coding/Research/QuantResearch/Libs",
    ]
    for _cand in _candidates:
        if _cand and _os.path.isdir(_cand):
            if _cand not in _sys.path:
                _sys.path.insert(0, _cand)
            break
    else:
        raise RuntimeError(
            "QuantResearch/Libs not found - tried: "
            + ", ".join(repr(c) for c in _candidates if c)
            + ". Set QUANTRESEARCH_LIBS to its path."
        )
    from db_utils import engine as _engine  # local import — Libs path required

    code = asset.upper()
    raw = pd.read_sql(
        "SELECT date, expiry_month, open, high, low, close, volume, "
        "settle, open_interest, bid, ask, "
        "night_open, night_high, night_low, night_close, night_volume "
        "FROM tw_index_futures_pv "
        "WHERE ticker = %(t)s ORDER BY date, expiry_month",
        _engine,
        params={"t": code},
    )
    if raw.empty:
        raise _RdsEmptyError(f"tw_index_futures_pv has no rows for {code!r}")

    raw["date"] = pd.to_datetime(raw["date"])
    for col in ("open", "high", "low", "close", "volume",
                "settle", "open_interest", "bid", "ask",
                "night_open", "night_high", "night_low", "night_close", "night_volume"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Rank by expiry within each date; weekly expiries (e.g. '202608W1')
    # → NaN via to_numeric and sort to the end, so front-month = smallest
    # monthly integer, matching the CSV loader's semantics.
    raw["_expiry_int"] = pd.to_numeric(raw["expiry_month"], errors="coerce")
    raw = raw.sort_values(["date", "_expiry_int"])
    raw["_rank"] = raw.groupby("date").cumcount()

    front_rank = 0 if contract == "front" else 1
    back_rank  = front_rank + 1

    front = raw[raw["_rank"] == front_rank].copy()
    back  = raw[raw["_rank"] == back_rank][["date", "expiry_month", "open", "close"]].rename(
        columns={"expiry_month": "back_expiry", "open": "back_open", "close": "back_close"}
    )

    front = front.rename(columns={
        "expiry_month":  "front_expiry",
        "settle":        "settlement",
        "open_interest": "oi",
    })

    keep_cols = [
        "date", "front_expiry",
        "open", "high", "low", "close", "volume", "settlement", "oi", "bid", "ask",
        "night_open", "night_high", "night_low", "night_close", "night_volume",
    ]
    front = front[keep_cols].merge(back, on="date", how="left")

    series = front.set_index("date").sort_index()
    series = series.dropna(subset=["open", "high", "low", "close"])

    symbol = code if contract == "front" else f"{code}_back"
    return BaseAsset.from_df(series, symbol=symbol, time_granularity="1d")


def _load_asset(asset: str, time_granularity: str, contract: str = "front") -> BaseAsset:
    if time_granularity in _INTRADAY_GRANULARITIES:
        return _load_intraday(asset, time_granularity)
    if time_granularity != "1d":
        raise NotImplementedError(f"Granularity '{time_granularity}' is not yet supported.")

    # TAIFEX index futures are stored in RDS (tw_index_futures_pv) — always fresh.
    # Falls back to CSV history for tickers not present in RDS.
    if asset.upper() in _RDS_INDEX_FUTURES_TICKERS:
        try:
            return _load_asset_from_rds(asset, contract)
        except _RdsEmptyError:
            pass  # fall through to CSV loader below

    import csv

    code = asset.upper()
    hdir = _history_dir()

    # Case-insensitive file lookup (mtx_history.csv vs MTX_history.csv)
    csv_file = next(
        (p for p in hdir.glob("*_history.csv")
         if p.stem.upper() == f"{code}_HISTORY"),
        None,
    )
    if csv_file is None:
        raise FileNotFoundError(
            f"No history CSV for '{code}' in {hdir}.\n"
            f"Available: {available_assets()}"
        )

    records = []
    with open(csv_file, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            n = len(row)
            if n < 16:
                continue
            session = row[17].strip() if n >= 18 else "一般"
            records.append({
                "date":       row[0].strip(),
                "expiry":     row[2].strip(),
                "open":       row[3].strip(),
                "high":       row[4].strip(),
                "low":        row[5].strip(),
                "close":      row[6].strip(),
                "volume":     row[9].strip(),
                "settlement": row[10].strip() if n > 10 else "",
                "oi":         row[11].strip() if n > 11 else "",
                "bid":        row[12].strip() if n > 12 else "",
                "ask":        row[13].strip() if n > 13 else "",
                "session":    session,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    for col in ["open", "high", "low", "close", "volume", "settlement", "oi", "bid", "ask"]:
        df[col] = pd.to_numeric(df[col].replace("-", np.nan), errors="coerce")

    # ── Day session (一般) — rank contracts by expiry per date ──────────────
    day = df[df["session"] == "一般"].copy()
    day["expiry_int"] = pd.to_numeric(day["expiry"], errors="coerce")
    day = day.sort_values(["date", "expiry_int"])
    day["_rank"] = day.groupby("date").cumcount()
    front_rank = 0 if contract == "front" else 1

    selected = day[day["_rank"] == front_rank].drop(columns=["expiry_int", "_rank", "session"])

    # ── Back-month (next-rank) close — needed for rollover-adjusted returns ──
    back_rank = front_rank + 1
    back = day[day["_rank"] == back_rank][["date", "expiry", "open", "close"]].rename(columns={
        "expiry": "back_expiry",
        "open":   "back_open",
        "close":  "back_close",
    })
    selected = selected.merge(back, on="date", how="left")

    # ── Night session (盤後) — match to the same (date, expiry) ─────────────
    night_cols = ["date", "expiry", "open", "high", "low", "close", "volume"]
    night = df[df["session"] == "盤後"][night_cols].rename(columns={
        "open":   "night_open",
        "high":   "night_high",
        "low":    "night_low",
        "close":  "night_close",
        "volume": "night_volume",
    })
    selected = selected.merge(night, on=["date", "expiry"], how="left")

    # Rename the merge-key "expiry" column to the more explicit "front_expiry".
    selected = selected.rename(columns={"expiry": "front_expiry"})

    series = selected.set_index("date").sort_index()
    series = series.dropna(subset=["open", "high", "low", "close"])

    symbol = code if contract == "front" else f"{code}_back"
    return BaseAsset.from_df(series, symbol=symbol, time_granularity="1d")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _active_periods_per_year(default: int = 252) -> int:
    """Look up periods/year from the active asset (intraday-aware)."""
    a = _ops._active_asset
    if a is None:
        return default
    return int(getattr(a, "periods_per_year", default))


def _sharpe(pnl: pd.Series, periods: "int | None" = None) -> float:
    """
    Annualised Sharpe. `periods` defaults to the active asset's
    `periods_per_year` so intraday Sharpe is annualised correctly
    (e.g. 252×1140 for 1-minute MXF data).
    """
    if periods is None:
        periods = _active_periods_per_year()
    r = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty or r.std(ddof=0) == 0:
        return np.nan
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=0))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def Simulate(
    sig: pd.Series,
    time_granularity: str = "1d",
    asset: str = "mtx",
    contract: str = "front",
    normalize: "bool | str" = True,
    norm_window: int = 252,
    norm_sensitivity: float = 1.0,
    point_value: float = 50.0,
    fixed_per_side: float = 20.0,
    fee_rate: float = 0.00002,
    start_date=None,
    end_date=None,
) -> dict:
    """
    Backtest a signal and plot a multi-panel diagnostic dashboard.

    Parameters
    ----------
    sig              : raw signal, date-indexed, arbitrary scale.
                       Positive = bullish bias, negative = bearish bias.
    time_granularity : '1d' (daily). Intraday granularities coming soon.
    asset            : 'mtx' — which asset to load.
    contract         : 'front' (近月, default) or 'back' (次月).
    normalize        : True → tanh, 'rank' → percentile-rank, False → as-is.
    norm_window      : rolling look-back window for normalization.
    norm_sensitivity : tanh saturation speed.
    point_value      : TWD per index point per contract (MTX = 50).
    fixed_per_side   : fixed commission in TWD per side per contract (default 20).
    fee_rate         : futures transaction tax per side as fraction of notional (default 0.00002).
    start_date,
    end_date         : restrict the evaluation window. If None, fall back to
                       whatever ``cta.set_date_range`` set. Signal normalisation
                       is still computed on the FULL history (so rolling stats
                       have a proper warmup), only the displayed P&L is sliced.

    Returns
    -------
    dict with keys: asset_obj, signal, exec_sig, pnl, cum_pnl, tcost, pnl_net, cum_pnl_net
    """
    # ── 1. Load full asset & set context ────────────────────────────────────
    asset_obj_full = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj_full)

    # ── 2. Align signal to full asset index ─────────────────────────────────
    sig = sig.reindex(asset_obj_full.index).astype(float)

    # ── 3. Normalize on FULL history (gives rolling stats a real warmup) ────
    if normalize is True:
        signal = normalize_signal(sig, method="tanh", window=norm_window, sensitivity=norm_sensitivity)
    elif isinstance(normalize, str):
        signal = normalize_signal(sig, method=normalize, window=norm_window, sensitivity=norm_sensitivity)
    else:
        signal = sig

    # ── 4. P&L:  lag(2, signal) * returns  (on full history) ────────────────
    exec_sig = signal.shift(2)
    ret      = asset_obj_full.returns
    pnl_full = (exec_sig * ret).rename("pnl")

    # ── 5. Transaction cost (on full history) ───────────────────────────────
    # turnover = |Δexec_sig| is already the count of trades (one side each),
    # so multiplying by cost_pct (per-side cost) gives the correct total —
    # NO extra factor of 2.
    turnover    = exec_sig.fillna(0).diff().abs()
    cost_pct    = fixed_per_side / (asset_obj_full.close * point_value) + fee_rate
    tcost_full  = (turnover * cost_pct).rename("tcost")
    pnl_net_full = (pnl_full - tcost_full).rename("pnl_net")

    # ── 6. Load 次月 for comparison ─────────────────────────────────────────
    try:
        asset_obj_back_full = _load_asset(asset, time_granularity, "back")
    except Exception:
        asset_obj_back_full = None

    # ── 7. Resolve & apply date range  (signal/PnL already on full data) ────
    start, end = _resolve_dates(start_date, end_date)
    if start is not None or end is not None:
        eval_idx = _slice_index(asset_obj_full.index, start, end)
        if len(eval_idx) == 0:
            raise ValueError(f"No data in range [{start}, {end}]")
        asset_obj      = _subset_asset(asset_obj_full, eval_idx)
        asset_obj_back = (_subset_asset(asset_obj_back_full, eval_idx)
                          if asset_obj_back_full is not None else None)
        _ops.set_active_asset(asset_obj)
        signal   = signal.reindex(eval_idx)
        exec_sig = exec_sig.reindex(eval_idx)
        pnl      = pnl_full.reindex(eval_idx)
        tcost    = tcost_full.reindex(eval_idx)
        pnl_net  = pnl_net_full.reindex(eval_idx)
    else:
        asset_obj      = asset_obj_full
        asset_obj_back = asset_obj_back_full
        pnl     = pnl_full
        tcost   = tcost_full
        pnl_net = pnl_net_full

    # Recompute cumulatives so they start at 0 within the eval window
    cum_pnl     = pnl.fillna(0).cumsum().rename("cum_pnl")
    cum_pnl_net = pnl_net.fillna(0).cumsum().rename("cum_pnl_net")

    # ── 8. Plot ─────────────────────────────────────────────────────────────
    _plot(
        asset_obj=asset_obj, asset_obj_back=asset_obj_back,
        signal=signal, exec_sig=exec_sig,
        pnl=pnl, cum_pnl=cum_pnl,
        tcost=tcost, pnl_net=pnl_net, cum_pnl_net=cum_pnl_net,
    )

    return {
        "asset_obj":      asset_obj,
        "asset_obj_back": asset_obj_back,
        "signal":         signal,
        "exec_sig":       exec_sig,
        "pnl":            pnl,
        "cum_pnl":        cum_pnl,
        "tcost":          tcost,
        "pnl_net":        pnl_net,
        "cum_pnl_net":    cum_pnl_net,
    }


# Backward-compatible alias — older notebooks call cta.simulate(...).
simulate = Simulate


# ─────────────────────────────────────────────────────────────────────────────
# Plot — layout  (3 cols × 5 rows)
#
#   (0,0) Cum PnL (close)         (0,1) Multi-price cum PnL    (0,2) Turnover
#   (1,0) Signal vs buy-hold      (1,1) Front vs back          (1,2) By weekday
#   (2,0) By day of month         (2,1) By year (doy)          (2,2) PnL vs tcost
#   (3,0) CASR around expiry      (3,1) Cum active return      (3,2) Vol-matched
#   (4,0) CASR around TSMC EA     (4,1) PnL by session         (4,2) —
# ─────────────────────────────────────────────────────────────────────────────

_FS  = 15   # subplot title fontsize
_FSS = 13   # axis label / legend fontsize

# Default location of the TSMC earnings-announcement CSV (TWSE format).
from pathlib import Path as _Path
_DEFAULT_TSMC_EA_PATH = _Path(__file__).parent.parent / "tsmc_ea.csv"


def _plot(asset_obj, asset_obj_back, signal, exec_sig, pnl, cum_pnl,
          tcost, pnl_net, cum_pnl_net):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(30, 40))
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.30)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(5)]

    _plot_cum_pnl(axes[0][0], pnl, cum_pnl, pnl_net, cum_pnl_net)
    _plot_price_comparison(axes[0][1], asset_obj, exec_sig)
    _plot_turnover(axes[0][2], exec_sig)

    _plot_signal_vs_asset(axes[1][0], asset_obj, pnl, cum_pnl)
    _plot_front_vs_back(axes[1][1], asset_obj, asset_obj_back, exec_sig, pnl, cum_pnl)
    _plot_by_weekday(axes[1][2], pnl)

    _plot_by_dom(axes[2][0], pnl)
    _plot_by_doy(axes[2][1], pnl)
    _plot_cost_summary(axes[2][2], pnl, cum_pnl, pnl_net, cum_pnl_net, tcost)

    _plot_casr(axes[3][0], signal, asset_obj)
    _plot_active_return(axes[3][1], asset_obj, pnl)
    _plot_vol_matched(axes[3][2], asset_obj, pnl)

    _plot_tsmc_ea_casr(axes[4][0], signal, asset_obj, _DEFAULT_TSMC_EA_PATH)
    _plot_session_breakdown(axes[4][1], asset_obj, exec_sig)
    axes[4][2].axis("off")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Individual subplot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cum_pnl(ax, pnl: pd.Series, cum_pnl: pd.Series,
                  pnl_net: pd.Series | None = None,
                  cum_pnl_net: pd.Series | None = None) -> None:
    """(0,0) Cumulative PnL — gross + net (post-cost) overlaid with both Sharpes."""
    sr_g = _sharpe(pnl)
    ax.plot(cum_pnl.index, cum_pnl.values, linewidth=1.2, color="#1565c0",
            label=f"gross  (SR {sr_g:+.2f})")
    if cum_pnl_net is not None and pnl_net is not None:
        sr_n = _sharpe(pnl_net)
        ax.plot(cum_pnl_net.index, cum_pnl_net.values, linewidth=1.2, color="#c62828",
                label=f"net (post-cost)  (SR {sr_n:+.2f})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(cum_pnl.index, cum_pnl.values, 0,
                    where=(cum_pnl.values >= 0), alpha=0.15, color="#00897b")
    ax.fill_between(cum_pnl.index, cum_pnl.values, 0,
                    where=(cum_pnl.values <  0), alpha=0.15, color="#c62828")
    ax.set_title("Cum PnL — gross vs net (close execution)", fontsize=_FS)
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=_FSS, loc="upper left"); ax.grid(True, alpha=0.3)


def _plot_price_comparison(ax, asset_obj, exec_sig: pd.Series) -> None:
    """(0,1) Cumulative PnL using close / open / vwap≈ execution price assumptions."""
    close_ret = asset_obj.close.pct_change()
    open_ret  = asset_obj.open.pct_change()
    tp        = (asset_obj.high + asset_obj.low + asset_obj.close) / 3
    vwap_ret  = tp / tp.shift(1) - 1

    series = {
        "close":  (exec_sig * close_ret).fillna(0).cumsum(),
        "open":   (exec_sig * open_ret).fillna(0).cumsum(),
        "vwap≈":  (exec_sig * vwap_ret).fillna(0).cumsum(),
    }
    colors = {
        "close":  "#1565c0",
        "open":   "#e65100",
        "vwap≈":  "#6a1b9a",
    }
    for name, cum in series.items():
        ax.plot(cum.index, cum.values, linewidth=1.0,
                color=colors[name], label=name, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Cum PnL by execution price", fontsize=_FS)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=_FSS, loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_turnover(ax, exec_sig: pd.Series) -> None:
    """(0,2) Rolling average turnover = |Δexec_sig|."""
    to_daily = exec_sig.fillna(0).diff().abs()
    to_20    = to_daily.rolling(20, min_periods=1).mean()
    to_252   = to_daily.rolling(252, min_periods=20).mean()

    ax.plot(to_20.index,  to_20.values,  linewidth=0.9, color="#e65100", alpha=0.8, label="20d avg")
    ax.plot(to_252.index, to_252.values, linewidth=1.4, color="#1565c0", alpha=0.9, label="252d avg")
    ax.set_title("Turnover  (|Δexec_sig|)", fontsize=_FS)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.legend(fontsize=_FSS)
    ax.grid(True, alpha=0.3)


def _bar_colors(vals):
    return ["#00897b" if v >= 0 else "#c62828" for v in vals]


def _plot_by_weekday(ax, pnl: pd.Series) -> None:
    """(1,0) Average daily PnL by day of week (Mon=0 … Fri=4)."""
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    grp    = pnl.groupby(pnl.index.dayofweek).mean()
    vals   = [float(grp.get(i, 0.0)) for i in range(7)]
    # drop weekend bins if all zero
    max_idx = max((i for i, v in enumerate(vals) if v != 0.0), default=4)
    vals   = vals[: max_idx + 1]
    lbls   = labels[: max_idx + 1]

    ax.bar(lbls, vals, color=_bar_colors(vals), alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Avg PnL by day of week", fontsize=_FS)
    ax.set_ylabel("Avg daily return")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_by_dom(ax, pnl: pd.Series) -> None:
    """(1,1) Average daily PnL by day of month (1–31)."""
    grp  = pnl.groupby(pnl.index.day).mean()
    days = grp.index.tolist()
    vals = grp.values.tolist()

    ax.bar(days, vals, color=_bar_colors(vals), alpha=0.85, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Avg PnL by day of month", fontsize=_FS)
    ax.set_xlabel("Day of month")
    ax.set_ylabel("Avg daily return")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_by_doy(ax, pnl: pd.Series) -> None:
    """(1,2) Cumulative PnL within each calendar year, plotted against day-of-year."""
    import matplotlib.pyplot as plt

    years = sorted(pnl.index.year.unique())
    cmap  = plt.cm.get_cmap("tab20", len(years))

    for i, yr in enumerate(years):
        yr_pnl = pnl[pnl.index.year == yr]
        sr     = _sharpe(yr_pnl)
        sr_str = f"{sr:.2f}" if not np.isnan(sr) else "n/a"
        doys   = yr_pnl.index.dayofyear
        cum    = yr_pnl.cumsum().values
        ax.plot(doys, cum, linewidth=0.8, color=cmap(i),
                label=f"{yr}  SR {sr_str}", alpha=0.75)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Cum PnL by year (day of year)", fontsize=_FS)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=9, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), framealpha=0.7)
    ax.grid(True, alpha=0.3)


def _alpha_beta(pnl: pd.Series, raw_ret: pd.Series, periods: int = 252) -> tuple[float, float]:
    """
    Annualised alpha (intercept) and beta (slope) from OLS:
        pnl_t = alpha/periods + beta · raw_ret_t + ε_t

    Returns (alpha_annualised, beta).
    """
    aligned = pd.concat([pnl, raw_ret], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan"), float("nan")
    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    var_x = x.var(ddof=0)
    if var_x == 0:
        return float("nan"), float("nan")
    beta  = float(((x - x.mean()) * (y - y.mean())).mean() / var_x)
    alpha_daily = float(y.mean() - beta * x.mean())
    return alpha_daily * periods, beta


def _plot_signal_vs_asset(ax, asset_obj, pnl, cum_pnl) -> None:
    """(1,0) Signal cumulative PnL vs buy-and-hold, with alpha/beta decomposition."""
    raw_ret    = asset_obj.returns
    cum_raw    = raw_ret.cumsum().rename("buy_hold")
    sharpe_sig = _sharpe(pnl)
    sharpe_raw = _sharpe(raw_ret)

    aligned    = pd.concat([pnl, raw_ret], axis=1).dropna()
    corr       = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    alpha_ann, beta = _alpha_beta(pnl, raw_ret)

    ax.plot(cum_pnl.index, cum_pnl.values, linewidth=1.2, color="#1565c0",
            label=f"signal  (SR {sharpe_sig:.2f})")
    ax.plot(cum_raw.index, cum_raw.values, linewidth=1.2, color="#e65100",
            label=f"buy & hold  (SR {sharpe_raw:.2f})", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Signal vs buy-and-hold  (corr {corr:+.2f}, "
        f"β {beta:+.2f}, α {alpha_ann*100:+.2f}%/yr)",
        fontsize=_FS,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=_FSS, loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_active_return(ax, asset_obj, pnl) -> None:
    """
    (3,1) Cumulative active return  cumsum(signal_pnl − buy_hold_ret).

    Diagnostic only — not a metric to optimize. Flat-or-short days on
    rising markets appear as drawdowns here even when the signal made
    the correct call.
    """
    raw_ret  = asset_obj.returns
    aligned  = pd.concat([pnl, raw_ret], axis=1).dropna()
    active   = (aligned.iloc[:, 0] - aligned.iloc[:, 1])
    cum_act  = active.cumsum()
    ir       = _sharpe(active)

    ax.plot(cum_act.index, cum_act.values, linewidth=1.2, color="#6a1b9a")
    ax.fill_between(cum_act.index, cum_act.values, 0,
                    where=(cum_act.values >= 0), alpha=0.20, color="#00897b")
    ax.fill_between(cum_act.index, cum_act.values, 0,
                    where=(cum_act.values <  0), alpha=0.20, color="#c62828")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Cum active return (signal − B&H)  IR {ir:+.2f}\n"
        f"⚠ misleading when signal is flat/short — diagnostic only",
        fontsize=_FS,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative excess vs B&H")
    ax.grid(True, alpha=0.3)


def _plot_vol_matched(ax, asset_obj, pnl, target_vol: float = 0.15,
                       periods: int = 252) -> None:
    """
    (3,2) Vol-matched cumulative PnL — both signal and buy-and-hold
    rescaled (ex-post, constant leverage) to the same annualised vol.
    This is the honest 'did the signal beat the market?' comparison.
    """
    raw_ret  = asset_obj.returns
    aligned  = pd.concat([pnl, raw_ret], axis=1).dropna()
    sig_r    = aligned.iloc[:, 0]
    bh_r     = aligned.iloc[:, 1]

    sig_vol = float(sig_r.std() * np.sqrt(periods))
    bh_vol  = float(bh_r.std()  * np.sqrt(periods))

    sig_scale = target_vol / sig_vol if sig_vol > 0 else 0.0
    bh_scale  = target_vol / bh_vol  if bh_vol  > 0 else 0.0

    cum_sig = (sig_r * sig_scale).cumsum()
    cum_bh  = (bh_r  * bh_scale).cumsum()

    sr_sig = _sharpe(sig_r)
    sr_bh  = _sharpe(bh_r)

    ax.plot(cum_sig.index, cum_sig.values, linewidth=1.2, color="#1565c0",
            label=f"signal × {sig_scale:.2f}  (SR {sr_sig:+.2f})")
    ax.plot(cum_bh.index,  cum_bh.values,  linewidth=1.2, color="#e65100",
            label=f"B&H × {bh_scale:.2f}  (SR {sr_bh:+.2f})", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Vol-matched cum PnL  (both ≈ {target_vol*100:.0f}% ann. vol)\n"
        f"Sharpe is leverage-invariant → this is the honest comparison",
        fontsize=_FS,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (vol-matched)")
    ax.legend(fontsize=_FSS, loc="upper left")
    ax.grid(True, alpha=0.3)




def _plot_front_vs_back(ax, asset_obj, asset_obj_back, exec_sig, pnl, cum_pnl) -> None:
    """(1,3) Signal PnL on 近月 vs 次月 futures."""
    sharpe_front = _sharpe(pnl)
    ax.plot(cum_pnl.index, cum_pnl.values, linewidth=1.2, color="#1565c0",
            label=f"近月  (SR {sharpe_front:.2f})")

    if asset_obj_back is not None:
        ret_back  = asset_obj_back.returns.reindex(exec_sig.index)
        pnl_back  = (exec_sig * ret_back).rename("pnl_back")
        cum_back  = pnl_back.fillna(0).cumsum()
        sharpe_back = _sharpe(pnl_back)
        ax.plot(cum_back.index, cum_back.values, linewidth=1.2, color="#e65100",
                label=f"次月  (SR {sharpe_back:.2f})", alpha=0.85)
    else:
        ax.text(0.5, 0.5, "次月 data unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=_FS)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("近月 vs 次月 signal PnL", fontsize=_FS)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=_FSS, loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_casr(ax, signal: pd.Series, asset_obj,
               window: int = 30) -> None:
    """
    (3,0) CASR around TAIFEX expiry — signal PnL CASR (blue, with ±1 SE band).
    """
    trading_index = asset_obj.index
    casr_pnl = (signal.shift(2) * asset_obj.returns).reindex(trading_index)
    try:
        result = _ops.Caar(window, casr_pnl)
    except ValueError as e:
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                ha="center", va="center", fontsize=_FS)
        return

    x      = result.index
    casr   = result["casr"]
    se     = result["se"]
    n      = int(result["n"].iloc[0])
    cum_lo = np.cumsum(result["mean"] - se)
    cum_hi = np.cumsum(result["mean"] + se)

    ax.plot(x, casr, linewidth=1.4, color="#1565c0",
            marker="o", markersize=5, markerfacecolor="#1565c0",
            markeredgecolor="white", markeredgewidth=0.6,
            label=f"signal CASR  (n={n} expiries)")
    ax.fill_between(x, cum_lo, cum_hi, alpha=0.2, color="#1565c0", label="±1 SE")
    ax.axvline(0, color="#c62828", linewidth=1.2, linestyle="--", alpha=0.8, label="expiry")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"CASR ±{window}d around expiry", fontsize=_FS)
    ax.set_xlabel("Days relative to expiry")
    ax.set_ylabel("Cumulative avg return")
    ax.legend(fontsize=_FSS)
    ax.grid(True, alpha=0.3)


def _plot_tsmc_ea_casr(ax, signal: pd.Series, asset_obj,
                        ea_path, window: int = 10) -> None:
    """
    (4,0) CASR around TSMC official quarterly earnings calls
    (`tsmc_quarterly_call` — the 4x/year 法說會 only, NOT broker
    conferences or investor days).

    `ea_path` is kept in the signature for backward compatibility but is
    now ignored: dates come from `cta.load_tsmc_event_dates('quarterly_call',
    trading_index)`, which parses the same underlying CSV and filters to
    the quarterly-call category.
    """
    trading_index = asset_obj.index

    try:
        from .tsmc_events import load_tsmc_event_dates
        ea_dates = load_tsmc_event_dates("quarterly_call", trading_index)
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, f"TSMC EA CSV not found:\n{e}",
                transform=ax.transAxes, ha="center", va="center", fontsize=_FSS)
        ax.set_title("CASR — TSMC quarterly earnings calls", fontsize=_FS)
        return
    except Exception as e:
        ax.text(0.5, 0.5, f"Failed to load TSMC quarterly calls: {e}",
                transform=ax.transAxes, ha="center", va="center", fontsize=_FSS)
        ax.set_title("CASR — TSMC quarterly earnings calls", fontsize=_FS)
        return

    if len(ea_dates) == 0:
        ax.text(0.5, 0.5, "No TSMC quarterly calls in the asset's trading range",
                transform=ax.transAxes, ha="center", va="center", fontsize=_FSS)
        ax.set_title("CASR — TSMC quarterly earnings calls", fontsize=_FS)
        return

    casr_pnl = (signal.shift(2) * asset_obj.returns).reindex(trading_index)

    try:
        result = _ops.Caar(window, casr_pnl, event_dates=ea_dates)
    except ValueError as e:
        ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                ha="center", va="center", fontsize=_FSS)
        return

    x      = result.index
    casr   = result["casr"]
    se     = result["se"]
    n      = int(result["n"].iloc[0])
    cum_lo = np.cumsum(result["mean"] - se)
    cum_hi = np.cumsum(result["mean"] + se)

    ax.plot(x, casr, linewidth=1.4, color="#e65100",
            marker="o", markersize=5, markerfacecolor="#e65100",
            markeredgecolor="white", markeredgewidth=0.6,
            label=f"signal CASR  (n={n} quarterly calls)")
    ax.fill_between(x, cum_lo, cum_hi, alpha=0.2, color="#e65100", label="±1 SE")
    ax.axvline(0, color="#c62828", linewidth=1.2, linestyle="--", alpha=0.8, label="earnings call")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"CASR ±{window}d around TSMC quarterly earnings calls", fontsize=_FS)
    ax.set_xlabel("Days relative to earnings call")
    ax.set_ylabel("Cumulative avg return")
    ax.legend(fontsize=_FSS)
    ax.grid(True, alpha=0.3)


def _plot_session_breakdown(ax, asset_obj, exec_sig: pd.Series) -> None:
    """
    (4,1) Cumulative signal PnL decomposed into four intraday windows.

        ret[t] = close[t] / close[t-1] - 1     (close-to-close)
              ≈ mid[t] + night[t] + over[t] + day[t]

    TAIFEX labeling convention: ``盤後[X]`` is the night session that
    ENDED on day X (started 15:00 of X−1, ended ~05:00 of X). So at index t:

        mid[t]   = night_open[t]  / close[t-1]     - 1   (close[t-1] @ 13:45 → 15:00 of t-1)
        night[t] = night_close[t] / night_open[t]  - 1   (15:00 of t-1 → 05:00 of t)
        over[t]  = open[t]        / night_close[t] - 1   (05:00 → 08:45 of t)
        day[t]   = close[t]       / open[t]        - 1   (08:45 → 13:45 of t)

    Pre-2017 (no 夜盤 data) `mid` and `night` are 0 and `over` falls back
    to `open[t] / close[t-1] - 1` so the four components still sum to ret.
    """
    close = asset_obj.close
    open_ = asset_obj.open
    # Use the rollover-aware previous close so the mid/over components don't
    # pick up the calendar-spread jump on rollover days.
    prev_close = asset_obj.continuous_prev_close

    has_night = ("night_open" in asset_obj.columns and "night_close" in asset_obj.columns
                 and asset_obj["night_open"].notna().any())

    if has_night:
        no  = asset_obj.night_open
        nc  = asset_obj.night_close
        mid_ret   = (no / prev_close - 1).fillna(0)
        night_ret = (nc / no          - 1).fillna(0)
        # Use night_close where available, fall back to prev day's close for pre-夜盤 era
        over_base = nc.fillna(prev_close)
        over_ret  = (open_ / over_base - 1).fillna(0)
    else:
        mid_ret   = pd.Series(0.0, index=close.index)
        night_ret = pd.Series(0.0, index=close.index)
        over_ret  = (open_ / prev_close - 1).fillna(0)

    day_ret = (close / open_ - 1).fillna(0)

    day_pnl   = (exec_sig * day_ret).fillna(0)
    mid_pnl   = (exec_sig * mid_ret).fillna(0)
    night_pnl = (exec_sig * night_ret).fillna(0)
    over_pnl  = (exec_sig * over_ret).fillna(0)

    cum_day   = day_pnl.cumsum()
    cum_mid   = mid_pnl.cumsum()
    cum_night = night_pnl.cumsum()
    cum_over  = over_pnl.cumsum()
    cum_total = (day_pnl + mid_pnl + night_pnl + over_pnl).cumsum()

    ax.plot(cum_day.index,   cum_day.values,   color="#1565c0", linewidth=1.2,
            label=f"日盤 day session  (SR {_sharpe(day_pnl):+.2f})")
    ax.plot(cum_mid.index,   cum_mid.values,   color="#e65100", linewidth=1.2,
            label=f"Mid-day gap  (SR {_sharpe(mid_pnl):+.2f})")
    ax.plot(cum_night.index, cum_night.values, color="#6a1b9a", linewidth=1.2,
            label=f"夜盤 night session  (SR {_sharpe(night_pnl):+.2f})")
    ax.plot(cum_over.index,  cum_over.values,  color="#00897b", linewidth=1.2,
            label=f"Overnight gap  (SR {_sharpe(over_pnl):+.2f})")
    ax.plot(cum_total.index, cum_total.values, color="black", linewidth=0.8,
            linestyle="--", alpha=0.5, label="sum (≈ close-to-close)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Cumulative PnL by intraday session", fontsize=_FS)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return contribution")
    ax.legend(fontsize=_FSS, loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_cost_summary(ax, pnl, cum_pnl, pnl_net, cum_pnl_net, tcost) -> None:
    """(2,0) Cumulative gross PnL, post-cost PnL, and cumulative transaction cost."""
    sharpe_gross = _sharpe(pnl)
    sharpe_net   = _sharpe(pnl_net)
    cum_tc       = tcost.fillna(0).cumsum()

    ax.plot(cum_pnl.index,     cum_pnl.values,     linewidth=1.2, color="#1565c0",
            label=f"cum PnL  (SR {sharpe_gross:.2f})")
    ax.plot(cum_pnl_net.index, cum_pnl_net.values, linewidth=1.2, color="#00897b",
            label=f"post-cost PnL  (SR {sharpe_net:.2f})")
    ax.plot(cum_tc.index,      cum_tc.values,       linewidth=1.2, color="#c62828",
            label="cum transaction cost", linestyle="--")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("PnL vs transaction cost", fontsize=_FS)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(fontsize=_FSS, loc="upper left")
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# SimulateAll
# ─────────────────────────────────────────────────────────────────────────────

def SimulateAll(
    *sigs: pd.Series,
    names: "list[str] | None" = None,
    time_granularity: str = "1d",
    asset: str = "mtx",
    contract: str = "front",
    normalize: "bool | str" = True,
    norm_window: int = 252,
    norm_sensitivity: float = 1.0,
    start_date=None,
    end_date=None,
) -> None:
    """
    Compare multiple signals side-by-side in a compact grid.

    Each signal occupies two columns:
      Left  — cumulative PnL with Sharpe in the title
      Right — raw daily PnL bars + 20-bar rolling turnover (twinx)

    Parameters
    ----------
    *sigs            : one or more raw signal Series
    names            : optional list of labels (defaults to "Signal 1", "Signal 2", …)
    time_granularity : bar size passed to the asset loader
    asset            : which asset to load ('mtx')
    contract         : 'front' (近月, default) or 'back' (次月)
    normalize        : same as Simulate() — True / 'rank' / False
    norm_window      : normalization look-back window
    norm_sensitivity : tanh saturation speed
    start_date,
    end_date         : restrict the evaluation window. If None, fall back to
                       whatever ``cta.set_date_range`` set.
    """
    import matplotlib.pyplot as plt

    if not sigs:
        raise ValueError("Provide at least one signal.")

    # ── load asset once on FULL history and set context ─────────────────────
    asset_obj_full = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj_full)
    ret_full = asset_obj_full.returns

    # Resolve & apply date range (signal normalisation still uses full history)
    start, end = _resolve_dates(start_date, end_date)
    if start is not None or end is not None:
        eval_idx = _slice_index(asset_obj_full.index, start, end)
        if len(eval_idx) == 0:
            raise ValueError(f"No data in range [{start}, {end}]")
    else:
        eval_idx = asset_obj_full.index

    n      = len(sigs)
    n_rows = math.ceil(n / 2)           # 2 signals per row, 4 cols total
    fig, axs = plt.subplots(
        n_rows, 4,
        figsize=(24, 5 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axs = axs[np.newaxis, :]        # always shape (n_rows, 4)

    for i, sig in enumerate(sigs):
        row      = i // 2
        col_base = (i % 2) * 2          # 0 for left signal, 2 for right signal

        ax_pnl = axs[row, col_base]
        ax_to  = axs[row, col_base + 1]

        name = names[i] if (names and i < len(names)) else f"Signal {i + 1}"

        # normalize on FULL history (so rolling stats have proper warmup)
        sig = sig.reindex(asset_obj_full.index).astype(float)
        if normalize is True:
            signal = normalize_signal(sig, method="tanh", window=norm_window, sensitivity=norm_sensitivity)
        elif isinstance(normalize, str):
            signal = normalize_signal(sig, method=normalize, window=norm_window, sensitivity=norm_sensitivity)
        else:
            signal = sig

        exec_sig_full = signal.shift(2)
        pnl_full      = (exec_sig_full * ret_full).rename("pnl")
        # Slice to eval window, then restart the cumulative sum at 0
        exec_sig = exec_sig_full.reindex(eval_idx)
        pnl      = pnl_full.reindex(eval_idx)
        cum_pnl  = pnl.fillna(0).cumsum()
        turnover = exec_sig.fillna(0).diff().abs().rolling(20, min_periods=1).mean()

        # ── left: cumulative PnL ─────────────────────────────────────────────
        sharpe = _sharpe(pnl)
        ax_pnl.plot(cum_pnl.index, cum_pnl.values, linewidth=1.2, color="#1565c0")
        ax_pnl.fill_between(cum_pnl.index, cum_pnl.values, 0,
                            where=(cum_pnl.values >= 0), alpha=0.20, color="#00897b")
        ax_pnl.fill_between(cum_pnl.index, cum_pnl.values, 0,
                            where=(cum_pnl.values <  0), alpha=0.20, color="#c62828")
        ax_pnl.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_pnl.set_title(f"{name}  —  Cum PnL  (Sharpe {sharpe:.2f})", fontsize=_FS)
        ax_pnl.set_ylabel("Cum return")
        ax_pnl.grid(True, alpha=0.3)

        # ── right: daily PnL bars + turnover ─────────────────────────────────
        bar_colors = ["#00897b" if v >= 0 else "#c62828" for v in pnl.fillna(0)]
        ax_to.bar(pnl.index, pnl.values, color=bar_colors, alpha=0.75, width=1)
        ax_to.set_ylabel("Daily PnL", fontsize=_FSS)
        ax_to.grid(True, alpha=0.2)

        ax2 = ax_to.twinx()
        ax2.plot(turnover.index, turnover.values,
                 color="#e65100", linewidth=0.9, alpha=0.85, label="Turnover (20d avg)")
        ax2.set_ylabel("Turnover", fontsize=_FSS, color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.legend(loc="upper right", fontsize=_FSS)

        avg_to = float(exec_sig.diff().abs().mean())
        ax_to.set_title(f"{name}  —  Daily PnL  (avg turnover {avg_to:.3f})", fontsize=_FS)

    # hide any unused axes in the last row
    for j in range(n, n_rows * 2):
        row      = j // 2
        col_base = (j % 2) * 2
        axs[row, col_base].axis("off")
        axs[row, col_base + 1].axis("off")

    plt.show()
