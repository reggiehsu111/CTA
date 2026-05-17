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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from .asset import BaseAsset
from . import operators as _ops

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
) -> pd.Series:
    """
    Normalize a raw signal to (-1, 1) without look-ahead.

    Idempotent: if `sig` already lies entirely in [-1, 1] (e.g. a previously
    normalized or pre-bounded signal), it is returned unchanged. This prevents
    re-normalization from warping pre-normalized inputs and stops filtered
    series (with NaN gaps) from picking up biased rolling stats over only the
    kept dates.

    Parameters
    ----------
    sig         : raw signal, arbitrary scale, date-indexed
    method      : 'tanh' (default) or 'rank'
    window      : rolling look-back window in bars
    sensitivity : tanh only — controls saturation speed (default 1.0)
    """
    sig = sig.astype(float)

    # ── Idempotence guard ────────────────────────────────────────────────────
    # Skip re-normalization only when the signal looks like a previous
    # normalization OUTPUT: bounded in [-1, 1] AND actually using the full
    # range (reaches at least 0.9 in absolute value somewhere). A signal
    # that's bounded but tiny (e.g. cal_spread ≈ ±0.05) is a raw input
    # that needs to be spread out, so we still normalize it.
    finite = sig.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        max_abs = float(finite.abs().max())
        if 0.9 <= max_abs <= 1.0 + 1e-9:
            return sig

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


def _load_asset(asset: str, time_granularity: str, contract: str = "front") -> BaseAsset:
    if time_granularity != "1d":
        raise NotImplementedError(f"Granularity '{time_granularity}' is not yet supported.")

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

    regular = df[df["session"] == "一般"].copy()
    regular["expiry_int"] = pd.to_numeric(regular["expiry"], errors="coerce")

    # Sort by expiry ascending per date; rank 0 = 近月, rank 1 = 次月
    regular = regular.sort_values(["date", "expiry_int"])
    regular["_rank"] = regular.groupby("date").cumcount()
    rank = 0 if contract == "front" else 1
    selected = regular[regular["_rank"] == rank].drop(columns=["expiry_int", "_rank"])

    series = selected.set_index("date").sort_index()
    series = series.dropna(subset=["open", "high", "low", "close"])

    symbol = code if contract == "front" else f"{code}_back"
    return BaseAsset.from_df(series, symbol=symbol, time_granularity="1d")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(pnl: pd.Series, periods: int = 252) -> float:
    r = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty or r.std(ddof=0) == 0:
        return np.nan
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=0))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def simulate(
    sig: pd.Series,
    time_granularity: str = "1d",
    asset: str = "mtx",
    contract: str = "front",
    normalize: bool | str = True,
    norm_window: int = 252,
    norm_sensitivity: float = 1.0,
    point_value: float = 50.0,
    fixed_per_side: float = 20.0,
    fee_rate: float = 0.00002,
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

    Returns
    -------
    dict with keys: asset_obj, signal, exec_sig, pnl, cum_pnl, tcost, pnl_net, cum_pnl_net
    """
    # ── 1. Load & set context ─────────────────────────────────────────────────
    asset_obj = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj)

    # ── 2. Align ─────────────────────────────────────────────────────────────
    sig = sig.reindex(asset_obj.index).astype(float)

    # ── 3. Normalize ──────────────────────────────────────────────────────────
    if normalize is True:
        signal = normalize_signal(sig, method="tanh", window=norm_window, sensitivity=norm_sensitivity)
    elif isinstance(normalize, str):
        signal = normalize_signal(sig, method=normalize, window=norm_window, sensitivity=norm_sensitivity)
    else:
        signal = sig

    # ── 4. P&L:  lag(2, signal) * returns ────────────────────────────────────
    exec_sig = signal.shift(2)                  # 2-bar execution lag
    ret      = asset_obj.returns                # close.pct_change()
    pnl      = (exec_sig * ret).rename("pnl")
    cum_pnl  = pnl.fillna(0).cumsum().rename("cum_pnl")  # NaN → 0 contrib (continuous)

    # ── 5. Transaction cost ───────────────────────────────────────────────────
    # cost per side (as fraction of contract value) = fixed_TWD/(close×point_value) + fee_rate
    # total daily cost = turnover × cost_per_side × 2 sides
    # NaN exec_sig (e.g. filtered-out days) → 0 exposure, so transitioning in/out counts as a real trade.
    turnover   = exec_sig.fillna(0).diff().abs()
    cost_pct   = fixed_per_side / (asset_obj.close * point_value) + fee_rate
    tcost      = (turnover * cost_pct * 2).rename("tcost")
    pnl_net    = (pnl - tcost).rename("pnl_net")
    cum_pnl_net = pnl_net.fillna(0).cumsum().rename("cum_pnl_net")

    # ── 6. Load 次月 for comparison ───────────────────────────────────────────
    try:
        asset_obj_back = _load_asset(asset, time_granularity, "back")
    except Exception:
        asset_obj_back = None

    # ── 7. Plot ───────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Plot — layout  (3 cols × 4 rows)
#
#   (0,0) Cum PnL (close)         (0,1) Multi-price cum PnL    (0,2) Turnover
#   (1,0) Signal vs buy-hold      (1,1) Front vs back          (1,2) By weekday
#   (2,0) By day of month         (2,1) By year (doy)          (2,2) PnL vs tcost
#   (3,0) CASR around expiry      (3,1) —                      (3,2) —
# ─────────────────────────────────────────────────────────────────────────────

_FS  = 15   # subplot title fontsize
_FSS = 13   # axis label / legend fontsize


def _plot(asset_obj, asset_obj_back, signal, exec_sig, pnl, cum_pnl,
          tcost, pnl_net, cum_pnl_net):
    fig = plt.figure(figsize=(30, 32))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.30)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(4)]

    _plot_cum_pnl(axes[0][0], pnl, cum_pnl)
    _plot_price_comparison(axes[0][1], asset_obj, exec_sig)
    _plot_turnover(axes[0][2], exec_sig)

    _plot_signal_vs_asset(axes[1][0], asset_obj, pnl, cum_pnl)
    _plot_front_vs_back(axes[1][1], asset_obj, asset_obj_back, exec_sig, pnl, cum_pnl)
    _plot_by_weekday(axes[1][2], pnl)

    _plot_by_dom(axes[2][0], pnl)
    _plot_by_doy(axes[2][1], pnl)
    _plot_cost_summary(axes[2][2], pnl, cum_pnl, pnl_net, cum_pnl_net, tcost)

    _plot_casr(axes[3][0], signal, asset_obj)
    axes[3][1].axis("off")
    axes[3][2].axis("off")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Individual subplot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cum_pnl(ax, pnl: pd.Series, cum_pnl: pd.Series) -> None:
    """(0,0) Cumulative PnL — close-price execution."""
    sharpe = _sharpe(pnl)
    ax.plot(cum_pnl.index, cum_pnl.values, linewidth=1.2, color="#1565c0")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_between(cum_pnl.index, cum_pnl.values, 0,
                    where=(cum_pnl.values >= 0), alpha=0.20, color="#00897b")
    ax.fill_between(cum_pnl.index, cum_pnl.values, 0,
                    where=(cum_pnl.values <  0), alpha=0.20, color="#c62828")
    ax.set_title(f"Cum PnL — close  (Sharpe {sharpe:.2f})", fontsize=_FS)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.grid(True, alpha=0.3)


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


def _plot_signal_vs_asset(ax, asset_obj, pnl, cum_pnl) -> None:
    """(0,3) Signal cumulative PnL vs raw buy-and-hold (weight=1 everywhere)."""
    raw_ret    = asset_obj.returns
    cum_raw    = raw_ret.cumsum().rename("buy_hold")
    sharpe_sig = _sharpe(pnl)
    sharpe_raw = _sharpe(raw_ret)

    aligned    = pd.concat([pnl, raw_ret], axis=1).dropna()
    corr       = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    ax.plot(cum_pnl.index, cum_pnl.values, linewidth=1.2, color="#1565c0",
            label=f"signal  (SR {sharpe_sig:.2f})")
    ax.plot(cum_raw.index, cum_raw.values, linewidth=1.2, color="#e65100",
            label=f"buy & hold  (SR {sharpe_raw:.2f})", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"Signal vs buy-and-hold  (corr {corr:.2f})", fontsize=_FS)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
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
    """(3,0) CASR — Lag(2, normalized_signal) * Returns(-1) around expiry dates."""
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
            label=f"CASR  (n={n} expiries)")
    ax.fill_between(x, cum_lo, cum_hi, alpha=0.2, color="#1565c0", label="±1 SE")
    ax.axvline(0, color="#c62828", linewidth=1.2, linestyle="--", alpha=0.8, label="expiry")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(f"CASR ±{window}d around expiry", fontsize=_FS)
    ax.set_xlabel("Days relative to expiry")
    ax.set_ylabel("Cumulative avg signal return")
    ax.legend(fontsize=_FSS)
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
    names: list[str] | None = None,
    time_granularity: str = "1d",
    asset: str = "mtx",
    contract: str = "front",
    normalize: bool | str = True,
    norm_window: int = 252,
    norm_sensitivity: float = 1.0,
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
    normalize        : same as simulate() — True / 'rank' / False
    norm_window      : normalization look-back window
    norm_sensitivity : tanh saturation speed
    """
    if not sigs:
        raise ValueError("Provide at least one signal.")

    # ── load asset once and set context ──────────────────────────────────────
    asset_obj = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj)
    ret = asset_obj.returns

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

        # normalize
        sig = sig.reindex(asset_obj.index).astype(float)
        if normalize is True:
            signal = normalize_signal(sig, method="tanh", window=norm_window, sensitivity=norm_sensitivity)
        elif isinstance(normalize, str):
            signal = normalize_signal(sig, method=normalize, window=norm_window, sensitivity=norm_sensitivity)
        else:
            signal = sig

        exec_sig = signal.shift(2)
        pnl      = (exec_sig * ret).rename("pnl")
        cum_pnl  = pnl.fillna(0).cumsum()   # NaN days contribute 0 → continuous line
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
