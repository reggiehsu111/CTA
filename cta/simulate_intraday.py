"""
simulate_intraday.py — diagnostic dashboard for intraday MXF/MTX backtests.

Entry point:
    SimulateIntraday(sig, time_granularity='1m')

Like `Simulate` but the panel layout is built for high-frequency data:
session-aware cumulative PnL, hour-of-day profile, daily-aggregated
distribution. Sharpe ratios are annualised with the active asset's
`periods_per_year`, so a 1-minute strategy's Sharpe is comparable to a
daily strategy's.
"""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .asset import BaseAsset
from . import operators as _ops
from .simulate import (
    _load_asset, _sharpe, normalize_signal,
    _resolve_dates, _subset_asset, _slice_index,
    _active_periods_per_year, _alpha_beta,
)

_FS, _FSS = 14, 12


def SimulateIntraday(
    sig: pd.Series,
    time_granularity: str = "1m",
    asset: str = "mtx",
    normalize: "bool | str" = False,
    norm_window: int = 252,
    norm_sensitivity: float = 1.0,
    point_value: float = 50.0,
    fixed_per_side: float = 20.0,
    fee_rate: float = 0.00002,
    target_vol: float = 0.15,
    exec_lag: int = 2,
    start_date=None,
    end_date=None,
) -> dict:
    """
    Backtest an intraday signal on MXF/MTX 1m / 5m / 15m / 30m / 1h data.

    Default `normalize=False` because intraday signals (z-scores, binary
    cross-overs, etc.) are usually already in [-1, 1]; flip to True if
    you're passing a raw factor.

    `exec_lag` controls the position-to-return alignment:
      *  +2 (default)  — signal at t is acted on for the bar [t+1, t+2]
                          → no look-ahead, the realistic backtest.
      *   0             — same-bar (assume zero latency execution).
      *  -1             — **lead-1 ORACLE test**: position at t uses the
                          signal from bar t+1. This is look-ahead bias
                          on purpose, so use only as a sanity check —
                          if even the oracle can't make money, the
                          signal carries no information.

    Subplot layout (3 × 3):
        (0,0) Cum PnL gross + net   (0,1) Signal vs B&H  (0,2) Vol-matched
        (1,0) Cum PnL by session    (1,1) Position       (1,2) Rolling turnover
        (2,0) Avg PnL by hour       (2,1) Daily PnL dist (2,2) Cumulative cost
    """
    # ── 1. Load asset on full history & set context ────────────────────────
    asset_obj_full = _load_asset(asset, time_granularity)
    _ops.set_active_asset(asset_obj_full)

    # ── 2. Align signal to full asset index ────────────────────────────────
    sig = sig.reindex(asset_obj_full.index).astype(float)

    # ── 3. Normalize on FULL history if requested ──────────────────────────
    if normalize is True:
        signal = normalize_signal(sig, method="tanh",
                                  window=norm_window, sensitivity=norm_sensitivity)
    elif isinstance(normalize, str):
        signal = normalize_signal(sig, method=normalize,
                                  window=norm_window, sensitivity=norm_sensitivity)
    else:
        signal = sig

    # ── 4. P&L on full data with chosen exec lag (default 2 = realistic) ───
    exec_sig_full = signal.shift(exec_lag)
    ret_full      = asset_obj_full.returns
    pnl_full      = (exec_sig_full * ret_full).rename("pnl")

    # ── 5. Transaction cost (turnover-based) ───────────────────────────────
    # turnover = |Δexec_sig| is the per-bar trade count; cost_pct is per-side.
    # The product is total cost — no extra ×2 (which would double-count).
    turnover    = exec_sig_full.fillna(0).diff().abs()
    cost_pct    = fixed_per_side / (asset_obj_full.close * point_value) + fee_rate
    tcost_full  = (turnover * cost_pct).rename("tcost")
    pnl_net_full = (pnl_full - tcost_full).rename("pnl_net")

    # ── 6. Resolve & apply date range ──────────────────────────────────────
    start, end = _resolve_dates(start_date, end_date)
    if start is not None or end is not None:
        eval_idx = _slice_index(asset_obj_full.index, start, end)
        if len(eval_idx) == 0:
            raise ValueError(f"No data in range [{start}, {end}]")
        asset_obj = _subset_asset(asset_obj_full, eval_idx)
        _ops.set_active_asset(asset_obj)
        signal    = signal.reindex(eval_idx)
        exec_sig  = exec_sig_full.reindex(eval_idx)
        pnl       = pnl_full.reindex(eval_idx)
        tcost     = tcost_full.reindex(eval_idx)
        pnl_net   = pnl_net_full.reindex(eval_idx)
    else:
        asset_obj = asset_obj_full
        exec_sig  = exec_sig_full
        pnl       = pnl_full
        tcost     = tcost_full
        pnl_net   = pnl_net_full

    # Recompute cumulatives so they start at 0 in the eval window
    cum_pnl     = pnl.fillna(0).cumsum().rename("cum_pnl")
    cum_pnl_net = pnl_net.fillna(0).cumsum().rename("cum_pnl_net")

    # ── 7. Plot ────────────────────────────────────────────────────────────
    _plot_intraday(asset_obj, signal, exec_sig,
                    pnl, cum_pnl, tcost, pnl_net, cum_pnl_net,
                    target_vol=target_vol)

    return {
        "asset_obj":   asset_obj,
        "signal":      signal,
        "exec_sig":    exec_sig,
        "pnl":         pnl,
        "cum_pnl":     cum_pnl,
        "tcost":       tcost,
        "pnl_net":     pnl_net,
        "cum_pnl_net": cum_pnl_net,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_intraday(asset_obj, signal, exec_sig,
                    pnl, cum_pnl, tcost, pnl_net, cum_pnl_net,
                    target_vol: float = 0.15):
    fig = plt.figure(figsize=(24, 18))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.28)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    _plot_cum_pnl(axes[0][0], pnl, cum_pnl, pnl_net, cum_pnl_net)
    _plot_signal_vs_asset(axes[0][1], asset_obj, pnl, cum_pnl)
    _plot_vol_matched(axes[0][2], asset_obj, pnl, target_vol=target_vol)

    _plot_by_session(axes[1][0], asset_obj, pnl)
    _plot_position(axes[1][1], exec_sig)
    _plot_turnover(axes[1][2], exec_sig, asset_obj)

    _plot_by_hour(axes[2][0], pnl)
    _plot_daily_dist(axes[2][1], pnl)
    _plot_cost(axes[2][2], tcost)

    plt.show()


def _plot_cum_pnl(ax, pnl, cum_pnl, pnl_net, cum_pnl_net):
    sr_g = _sharpe(pnl)
    sr_n = _sharpe(pnl_net)
    ax.plot(cum_pnl.index,     cum_pnl.values     * 100, color="#1565c0", linewidth=1.1,
            label=f"Gross  (SR {sr_g:+.2f})")
    ax.plot(cum_pnl_net.index, cum_pnl_net.values * 100, color="#c62828", linewidth=1.1,
            label=f"Net    (SR {sr_n:+.2f})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Cumulative PnL", fontsize=_FS)
    ax.set_xlabel("Date"); ax.set_ylabel("Cum return (%)")
    ax.legend(fontsize=_FSS, loc="upper left"); ax.grid(True, alpha=0.3)


def _plot_signal_vs_asset(ax, asset_obj, pnl, cum_pnl):
    raw_ret = asset_obj.returns
    cum_raw = raw_ret.fillna(0).cumsum()
    sr_sig  = _sharpe(pnl)
    sr_raw  = _sharpe(raw_ret)
    aligned = pd.concat([pnl, raw_ret], axis=1).dropna()
    corr    = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 1 else float("nan")
    alpha_ann, beta = _alpha_beta(pnl, raw_ret, periods=asset_obj.periods_per_year)

    ax.plot(cum_pnl.index, cum_pnl.values * 100, color="#1565c0", linewidth=1.1,
            label=f"signal  (SR {sr_sig:+.2f})")
    ax.plot(cum_raw.index, cum_raw.values * 100, color="#e65100", linewidth=1.1,
            alpha=0.85, label=f"buy & hold  (SR {sr_raw:+.2f})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Signal vs buy-and-hold  (corr {corr:+.2f}, β {beta:+.2f}, α {alpha_ann*100:+.2f}%/yr)",
        fontsize=_FS,
    )
    ax.set_xlabel("Date"); ax.set_ylabel("Cum return (%)")
    ax.legend(fontsize=_FSS, loc="upper left"); ax.grid(True, alpha=0.3)


def _plot_vol_matched(ax, asset_obj, pnl, target_vol: float = 0.15):
    periods = asset_obj.periods_per_year
    raw_ret = asset_obj.returns
    aligned = pd.concat([pnl, raw_ret], axis=1).dropna()
    if len(aligned) < 2:
        ax.text(0.5, 0.5, "not enough overlap", transform=ax.transAxes,
                ha="center", va="center")
        return
    sig_r, bh_r = aligned.iloc[:, 0], aligned.iloc[:, 1]
    sig_vol = float(sig_r.std() * np.sqrt(periods))
    bh_vol  = float(bh_r.std() * np.sqrt(periods))
    sig_scale = target_vol / sig_vol if sig_vol > 0 else 0.0
    bh_scale  = target_vol / bh_vol  if bh_vol  > 0 else 0.0
    cum_sig = (sig_r * sig_scale).cumsum() * 100
    cum_bh  = (bh_r  * bh_scale).cumsum()  * 100

    ax.plot(cum_sig.index, cum_sig.values, color="#1565c0", linewidth=1.1,
            label=f"signal × {sig_scale:.2f}  (SR {_sharpe(sig_r):+.2f})")
    ax.plot(cum_bh.index,  cum_bh.values,  color="#e65100", linewidth=1.1, alpha=0.85,
            label=f"B&H × {bh_scale:.2f}  (SR {_sharpe(bh_r):+.2f})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(f"Vol-matched ({int(target_vol*100)}% ann.)  — Sharpe-invariant comparison",
                 fontsize=_FS)
    ax.set_xlabel("Date"); ax.set_ylabel("Cum return (%)")
    ax.legend(fontsize=_FSS, loc="upper left"); ax.grid(True, alpha=0.3)


def _plot_by_session(ax, asset_obj, pnl):
    sess = asset_obj.session
    pnl_d = pnl.where(sess == "day")
    pnl_n = pnl.where(sess == "night")
    cum_d = pnl_d.fillna(0).cumsum() * 100
    cum_n = pnl_n.fillna(0).cumsum() * 100
    sr_d  = _sharpe(pnl_d.dropna())
    sr_n  = _sharpe(pnl_n.dropna())

    ax.plot(cum_d.index, cum_d.values, color="#1565c0", linewidth=1.1,
            label=f"日盤 day    (SR {sr_d:+.2f})")
    ax.plot(cum_n.index, cum_n.values, color="#6a1b9a", linewidth=1.1,
            label=f"夜盤 night  (SR {sr_n:+.2f})")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Cum PnL by trading session", fontsize=_FS)
    ax.set_xlabel("Date"); ax.set_ylabel("Cum return (%)")
    ax.legend(fontsize=_FSS, loc="upper left"); ax.grid(True, alpha=0.3)


def _plot_position(ax, exec_sig):
    pos = exec_sig.fillna(0)
    avg_pos = float(pos.abs().mean())
    max_pos = float(pos.abs().max())
    pct_held = float((pos.abs() > 0.01).mean()) * 100
    ax.plot(pos.index, pos.values, color="#1565c0", linewidth=0.5, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Position (exec_sig)  —  avg |pos| {avg_pos:.3f}  max {max_pos:.2f}  "
        f"held {pct_held:.0f}% of bars",
        fontsize=_FS,
    )
    ax.set_xlabel("Date"); ax.set_ylabel("Position")
    ax.grid(True, alpha=0.3)


def _plot_turnover(ax, exec_sig, asset_obj):
    # rolling window ≈ 1 trading day worth of bars
    bars_per_day = max(int(round(asset_obj.periods_per_year / 252)), 1)
    win_short    = bars_per_day                      # ≈ 1 trading day
    win_long     = bars_per_day * 20                 # ≈ 1 trading month

    to_per_bar = exec_sig.fillna(0).diff().abs()
    short = to_per_bar.rolling(win_short, min_periods=1).mean()
    long_ = to_per_bar.rolling(win_long,  min_periods=int(win_short)).mean()

    ax.plot(short.index, short.values, color="#e65100", linewidth=0.6, alpha=0.6,
            label=f"≈ 1d  ({win_short} bars)")
    ax.plot(long_.index, long_.values, color="#1565c0", linewidth=1.0,
            label=f"≈ 1m  ({win_long} bars)")
    ax.set_title(
        f"Rolling |Δposition|  —  total avg/bar {float(to_per_bar.mean()):.4f}",
        fontsize=_FS,
    )
    ax.set_xlabel("Date"); ax.set_ylabel("Mean |Δexec_sig|")
    ax.legend(fontsize=_FSS, loc="upper right"); ax.grid(True, alpha=0.3)


def _plot_by_hour(ax, pnl):
    valid = pnl.dropna()
    if valid.empty:
        ax.text(0.5, 0.5, "no PnL", transform=ax.transAxes, ha="center", va="center")
        return
    hr   = valid.index.hour
    mean = valid.groupby(hr).mean() * 1e4                # bps
    n    = valid.groupby(hr).count()
    colors = ["#00897b" if v >= 0 else "#c62828" for v in mean.values]

    ax.bar(mean.index, mean.values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(f"Average PnL by hour of day (bps)  —  total bars {int(n.sum()):,}",
                 fontsize=_FS)
    ax.set_xlabel("Hour (TW)"); ax.set_ylabel("Mean PnL (bps)")
    ax.set_xticks(range(0, 24, 2)); ax.grid(True, alpha=0.3, axis="y")


def _plot_daily_dist(ax, pnl):
    # Aggregate to per-calendar-day P&L for the distribution view
    daily = pnl.fillna(0).groupby(pnl.index.date).sum() * 100   # %
    if daily.empty:
        ax.text(0.5, 0.5, "no daily PnL", transform=ax.transAxes, ha="center", va="center")
        return
    m, s = float(daily.mean()), float(daily.std())
    sr   = float(np.sqrt(252) * m / s) if s > 0 else float("nan")

    ax.hist(daily.values, bins=60, color="#1565c0", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(m, color="#c62828", linewidth=1.2, linestyle="--",
               label=f"mean {m:+.3f}%   SR_daily {sr:+.2f}")
    ax.set_title(f"Daily-aggregated PnL distribution  (N={len(daily):,} days, σ {s:.3f}%)",
                 fontsize=_FS)
    ax.set_xlabel("Daily P&L (%)"); ax.set_ylabel("Count")
    ax.legend(fontsize=_FSS); ax.grid(True, alpha=0.3, axis="y")


def _plot_cost(ax, tcost):
    cum_tc = tcost.fillna(0).cumsum() * 100
    if cum_tc.empty:
        ax.text(0.5, 0.5, "no cost data", transform=ax.transAxes, ha="center", va="center")
        return
    ax.fill_between(cum_tc.index, 0, cum_tc.values, color="#c62828", alpha=0.6,
                    label=f"cum cost  end {cum_tc.iloc[-1]:+.2f}%")
    ax.plot(cum_tc.index, cum_tc.values, color="#c62828", linewidth=0.8)
    ax.set_title("Cumulative transaction cost", fontsize=_FS)
    ax.set_xlabel("Date"); ax.set_ylabel("Cum cost (%)")
    ax.legend(fontsize=_FSS, loc="upper left"); ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-signal comparison — same role as `SimulateAll` but cheap at intraday scale
# ─────────────────────────────────────────────────────────────────────────────

def SimulateAllIntraday(
    *sigs: pd.Series,
    names: "list[str] | None" = None,
    time_granularity: str = "1m",
    asset: str = "mtx",
    normalize: "bool | str" = False,
    norm_window: int = 252,
    norm_sensitivity: float = 1.0,
    point_value:     float = 50.0,
    fixed_per_side:  float = 20.0,
    fee_rate:        float = 0.00002,
    exec_lag:        int = 2,
    start_date=None,
    end_date=None,
) -> None:
    """
    Compare multiple intraday signals side-by-side, like `SimulateAll` but
    designed for high bar counts.

    Each signal occupies two columns:
      Left  — gross + net cumulative PnL line (cheap to render)
      Right — *daily-aggregated* PnL bars (~hundreds of bars, not the millions
              you'd get from per-bar bars at 1m)

    All Sharpes are annualised with the active asset's `periods_per_year`,
    so 1m and 1d strategies are directly comparable.

    `exec_lag` controls signal-to-return alignment (see `SimulateIntraday`
    docstring). Set to -1 for the look-ahead oracle test.
    """
    import math
    if not sigs:
        raise ValueError("Provide at least one signal.")

    # ── load asset on FULL history & set context ────────────────────────────
    asset_obj_full = _load_asset(asset, time_granularity)
    _ops.set_active_asset(asset_obj_full)
    ret_full = asset_obj_full.returns

    start, end = _resolve_dates(start_date, end_date)
    if start is not None or end is not None:
        eval_idx = _slice_index(asset_obj_full.index, start, end)
        if len(eval_idx) == 0:
            raise ValueError(f"No data in range [{start}, {end}]")
    else:
        eval_idx = asset_obj_full.index

    n      = len(sigs)
    n_rows = math.ceil(n / 2)
    fig, axs = plt.subplots(
        n_rows, 4, figsize=(24, 4.5 * n_rows), constrained_layout=True,
    )
    if n_rows == 1:
        axs = axs[np.newaxis, :]

    sess_full  = asset_obj_full.session
    close_full = asset_obj_full.close
    cost_pct_full = fixed_per_side / (close_full * point_value) + fee_rate

    for i, sig in enumerate(sigs):
        row      = i // 2
        col_base = (i % 2) * 2
        ax_pnl   = axs[row, col_base]
        ax_d     = axs[row, col_base + 1]
        name     = names[i] if (names and i < len(names)) else f"Signal {i + 1}"

        sig_aligned = sig.reindex(asset_obj_full.index).astype(float)
        if normalize is True:
            signal = normalize_signal(sig_aligned, method="tanh",
                                      window=norm_window, sensitivity=norm_sensitivity)
        elif isinstance(normalize, str):
            signal = normalize_signal(sig_aligned, method=normalize,
                                      window=norm_window, sensitivity=norm_sensitivity)
        else:
            signal = sig_aligned

        exec_sig_full = signal.shift(exec_lag)
        pnl_full      = exec_sig_full * ret_full

        turnover_full = exec_sig_full.fillna(0).diff().abs()
        tcost_full    = turnover_full * cost_pct_full          # no ×2
        pnl_net_full  = pnl_full - tcost_full

        # Slice to eval window
        pnl      = pnl_full.reindex(eval_idx)
        pnl_net  = pnl_net_full.reindex(eval_idx)
        exec_sig = exec_sig_full.reindex(eval_idx)
        sess     = sess_full.reindex(eval_idx)

        cum_pnl     = pnl.fillna(0).cumsum() * 100        # %
        cum_pnl_net = pnl_net.fillna(0).cumsum() * 100

        sr_g = _sharpe(pnl)
        sr_n = _sharpe(pnl_net)
        sr_d = _sharpe(pnl.where(sess == "day"))
        sr_nt = _sharpe(pnl.where(sess == "night"))

        # ── Left: cumulative gross + net PnL ────────────────────────────────
        ax_pnl.plot(cum_pnl.index,     cum_pnl.values,     color="#1565c0", linewidth=1.0,
                    label=f"gross  (SR {sr_g:+.2f})")
        ax_pnl.plot(cum_pnl_net.index, cum_pnl_net.values, color="#c62828", linewidth=1.0,
                    label=f"net    (SR {sr_n:+.2f})")
        ax_pnl.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        lag_tag = ("lag" if exec_lag > 0 else "lead" if exec_lag < 0 else "same-bar")
        ax_pnl.set_title(
            f"{name}  —  Cum PnL  [{lag_tag} {abs(exec_lag) if exec_lag != 0 else ''}]  "
            f"(day {sr_d:+.2f}  /  night {sr_nt:+.2f})",
            fontsize=11,
        )
        ax_pnl.set_ylabel("Cum return (%)")
        ax_pnl.legend(fontsize=9, loc="upper left")
        ax_pnl.grid(True, alpha=0.3)

        # ── Right: daily-aggregated PnL bars ────────────────────────────────
        # Aggregate to ~one bar per calendar day to avoid plotting all minutes
        daily_gross = pnl.fillna(0).groupby(pnl.index.date).sum() * 100   # %
        bar_colors  = ["#00897b" if v >= 0 else "#c62828" for v in daily_gross.values]
        ax_d.bar(pd.to_datetime(daily_gross.index), daily_gross.values,
                 color=bar_colors, alpha=0.85, width=0.9)
        ax_d.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_d.set_title(
            f"{name}  —  Daily-aggregated gross PnL  "
            f"(σ {daily_gross.std():.3f}%/d,  N={len(daily_gross)})",
            fontsize=12,
        )
        ax_d.set_ylabel("Daily PnL (%)")
        ax_d.grid(True, alpha=0.3, axis="y")

    # hide unused axes
    for j in range(n, n_rows * 2):
        r, c = j // 2, (j % 2) * 2
        axs[r, c].axis("off")
        axs[r, c + 1].axis("off")

    plt.show()


__all__ = ["SimulateIntraday", "SimulateAllIntraday"]
