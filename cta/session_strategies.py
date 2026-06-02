"""
session_strategies.py — fixed time-of-day strategies on TAIFEX index futures.

Each function simulates a deterministic intraday trading schedule (no signal
involved) so you can isolate the structural P&L of session-window biases.

Functions
---------
simulate_midday_short_night_long  Short on day-session close, cover and go
                                  long at night-session open, exit at night
                                  close. 4 transactions per active day.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .asset import BaseAsset
from . import operators as _ops
from .simulate import _load_asset, _sharpe, _resolve_dates


def simulate_midday_short_night_long(
    asset: "str | BaseAsset" = "mtx",
    *,
    contract:         str = "front",
    time_granularity: str = "1d",
    contracts:        int = 1,
    point_value:      float = 50.0,
    fixed_per_side:   float = 20.0,
    fee_rate:         float = 0.00002,
    spread_fraction:  float = 0.5,
    start_date:       "str | pd.Timestamp | None" = None,
    end_date:         "str | pd.Timestamp | None" = None,
) -> dict:
    """
    Simulate the schedule:

        13:45 of day t-1 (日盤 close[t-1])  →  SHORT `contracts` contracts
        15:00 of day t-1 (夜盤 open[t])     →  BUY 2 × `contracts`
                                                (close short + open long)
        05:00 of day t   (夜盤 close[t])    →  SELL `contracts` (close long)

    Net effect per active day: short the mid-day gap, long the night session.

    Note: TAIFEX labels the 夜盤 session by its END date, so 夜盤[t] is the
    session that started on the evening of day t-1. All P&L lines are
    attributed to index t (the date on which the position is finally closed).

    Active days are those with non-NaN previous-day close, night_open, and
    night_close. Pre-2017 days (no after-hours session) contribute 0 P&L
    and 0 cost.

    Parameters
    ----------
    start_date, end_date :
        Restrict the simulation to [start_date, end_date] inclusive. Either
        can be None. Useful for train/test splits — e.g. fit hypotheses on
        2017-01-01 → 2022-12-31 then check on 2023-01-01 → today.

    Returns
    -------
    dict with: pnl_gross, pnl_net, cum_gross, cum_net, cum_buyhold, costs
    """
    # ── 1. Load asset & slice to requested date range ───────────────────────
    if isinstance(asset, BaseAsset):
        asset_obj = asset
    else:
        asset_obj = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj)

    full_idx = asset_obj.index
    start, end = _resolve_dates(start_date, end_date)
    idx = full_idx
    if start is not None:
        idx = idx[idx >= start]
    if end is not None:
        idx = idx[idx <= end]
    if len(idx) == 0:
        raise ValueError(f"No data in date range [{start}, {end}]")

    # Compute on the FULL index first so close.shift(1) at the start of the
    # eval window uses the actual previous day's close, then slice to `idx`.
    close_full       = asset_obj.close
    night_open_full  = asset_obj.night_open
    night_close_full = asset_obj.night_close
    spread_full      = (asset_obj.ask - asset_obj.bid).fillna(0.0)

    # Use the rollover-aware previous close — on roll days this is yesterday's
    # back-month close (= today's front), so the mid-day-short PnL doesn't
    # bake in the calendar-spread artifact across the contract change.
    prev_close_full = asset_obj.continuous_prev_close

    close       = close_full.reindex(idx)
    night_open  = night_open_full.reindex(idx)
    night_close = night_close_full.reindex(idx)
    prev_close  = prev_close_full.reindex(idx)
    spread      = spread_full.reindex(idx)
    prev_spread = spread_full.shift(1).reindex(idx).fillna(0.0)

    # Days where all required prices exist
    active = prev_close.notna() & night_open.notna() & night_close.notna()

    # ── 2. Gross P&L (NTD per day) ──────────────────────────────────────────
    # Short at close[t-1], cover at night_open[t]:
    midday_pts = prev_close - night_open            # +ve if night-open prints below day-close
    # Long position from night_open[t] to night_close[t]:
    night_pts  = night_close - night_open
    pnl_gross  = (contracts * point_value * (midday_pts + night_pts)).where(active, np.nan)

    # ── 3. Transaction costs ────────────────────────────────────────────────
    # 4 transactions per active day (1 sell-open at close[t-1],
    # 2 buy at night_open[t], 1 sell-close at night_close[t]).
    fixed_per_day  = pd.Series(4.0 * contracts * fixed_per_side, index=close.index)

    # Tax: per side, on each transaction's notional.
    tax_per_day = (
        prev_close + 2.0 * night_open + night_close
    ) * point_value * fee_rate * contracts

    # Spread cost: use yesterday's bid-ask for the close[t-1] trade, today's
    # bid-ask as a proxy for the night-session trades (we don't have
    # bid/ask data for 夜盤). 4 trades total.
    spread_per_day = (
        prev_spread + 3.0 * spread
    ) * point_value * spread_fraction * contracts

    cost_total = (fixed_per_day + tax_per_day + spread_per_day).where(active, 0.0).fillna(0.0)

    # ── 4. Net P&L and cumulative ───────────────────────────────────────────
    pnl_net   = (pnl_gross - cost_total).where(active, np.nan)
    cum_gross = pnl_gross.fillna(0).cumsum()
    cum_net   = pnl_net.fillna(0).cumsum()

    cum_fixed  = fixed_per_day.where(active, 0).fillna(0).cumsum()
    cum_tax    = tax_per_day.where(active, 0).fillna(0).cumsum()
    cum_spread = spread_per_day.where(active, 0).fillna(0).cumsum()
    cum_cost   = cost_total.cumsum()

    # ── 5. "Long night session only" comparison ─────────────────────────────
    # Strategy: BUY at night_open[t], SELL at night_close[t]. 2 trades/day.
    night_only_gross   = (contracts * point_value * night_pts).where(active, np.nan)
    night_only_fixed   = pd.Series(2.0 * contracts * fixed_per_side, index=close.index)
    night_only_tax     = (night_open + night_close) * point_value * fee_rate * contracts
    night_only_spread  = 2.0 * spread * point_value * spread_fraction * contracts
    night_only_cost    = (night_only_fixed + night_only_tax + night_only_spread).where(active, 0.0).fillna(0.0)
    night_only_net     = (night_only_gross - night_only_cost).where(active, np.nan)
    cum_night_only_gross = night_only_gross.fillna(0).cumsum()
    cum_night_only_net   = night_only_net.fillna(0).cumsum()

    # ── 6. Buy-and-hold benchmark ───────────────────────────────────────────
    bh_daily    = (contracts * point_value * close.diff()).fillna(0.0)
    cum_buyhold = bh_daily.cumsum()

    # ── 7. Plot ─────────────────────────────────────────────────────────────
    _plot_midday_short_night_long(
        cum_gross, cum_net, pnl_gross, pnl_net,
        cum_fixed, cum_tax, cum_spread, cum_cost,
        cum_buyhold, bh_daily,
        cum_night_only_gross, cum_night_only_net,
        night_only_gross, night_only_net,
        contracts, active, idx,
    )

    return {
        "active":           active,
        "pnl_gross":        pnl_gross,
        "pnl_net":          pnl_net,
        "cum_gross":        cum_gross,
        "cum_net":          cum_net,
        "cum_buyhold":      cum_buyhold,
        "pnl_buyhold":      bh_daily,
        "cum_night_gross":  cum_night_only_gross,
        "cum_night_net":    cum_night_only_net,
        "pnl_night_gross":  night_only_gross,
        "pnl_night_net":    night_only_net,
        "costs": {
            "fixed":  cum_fixed,
            "tax":    cum_tax,
            "spread": cum_spread,
            "total":  cum_cost,
        },
    }


def _plot_midday_short_night_long(
    cum_gross: pd.Series,
    cum_net:   pd.Series,
    pnl_gross: pd.Series,
    pnl_net:   pd.Series,
    cum_fixed:  pd.Series,
    cum_tax:    pd.Series,
    cum_spread: pd.Series,
    cum_cost:   pd.Series,
    cum_buyhold: pd.Series,
    pnl_buyhold: pd.Series,
    cum_night_only_gross: pd.Series,
    cum_night_only_net:   pd.Series,
    pnl_night_only_gross: pd.Series,
    pnl_night_only_net:   pd.Series,
    contracts:  int,
    active:     pd.Series,
    idx:        pd.DatetimeIndex,
) -> None:
    sr_gross    = _sharpe(pnl_gross)
    sr_net      = _sharpe(pnl_net)
    sr_buyhold  = _sharpe(pnl_buyhold.replace(0, np.nan))
    sr_n_gross  = _sharpe(pnl_night_only_gross)
    sr_n_net    = _sharpe(pnl_night_only_net)
    n_active    = int(active.sum())

    date_range = f"{idx[0].date()} → {idx[-1].date()}"

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.30})

    # ── Top: cumulative PnL — combined vs night-only vs buy-and-hold ────────
    ax = axes[0]
    ax.plot(cum_gross.index, cum_gross.values / 1e3, color="#1565c0", linewidth=1.4,
            label=f"Mid-day short + Night long  gross  "
                  f"(SR {sr_gross:+.2f}, end {cum_gross.iloc[-1]/1e3:+,.1f}K)")
    ax.plot(cum_net.index,   cum_net.values   / 1e3, color="#c62828", linewidth=1.4,
            label=f"Mid-day short + Night long  net  "
                  f"(SR {sr_net:+.2f}, end {cum_net.iloc[-1]/1e3:+,.1f}K)")
    ax.plot(cum_night_only_gross.index, cum_night_only_gross.values / 1e3,
            color="#6a1b9a", linewidth=1.3, alpha=0.9,
            label=f"Night long only  gross  "
                  f"(SR {sr_n_gross:+.2f}, end {cum_night_only_gross.iloc[-1]/1e3:+,.1f}K)")
    ax.plot(cum_night_only_net.index, cum_night_only_net.values / 1e3,
            color="#00897b", linewidth=1.3, alpha=0.9,
            label=f"Night long only  net  "
                  f"(SR {sr_n_net:+.2f}, end {cum_night_only_net.iloc[-1]/1e3:+,.1f}K)")
    ax.plot(cum_buyhold.index, cum_buyhold.values / 1e3, color="#424242", linewidth=1.2,
            linestyle="--", alpha=0.85,
            label=f"Buy & hold  (SR {sr_buyhold:+.2f}, end {cum_buyhold.iloc[-1]/1e3:+,.1f}K)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Intraday session strategies   ({contracts} contract"
        f"{'s' if contracts != 1 else ''}, {n_active:,} active days,  {date_range})",
        fontsize=14,
    )
    ax.set_ylabel("Cumulative PnL (K NTD)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Bottom: stacked cumulative cost breakdown ───────────────────────────
    ax = axes[1]
    x = cum_fixed.index
    layer_1 = cum_fixed.values / 1e3
    layer_2 = (cum_fixed + cum_tax).values / 1e3
    layer_3 = (cum_fixed + cum_tax + cum_spread).values / 1e3

    ax.fill_between(x, 0, layer_1, color="#1565c0", alpha=0.7,
                    label=f"Fixed (20 NTD/side × 4)  total {cum_fixed.iloc[-1]/1e3:.1f}K")
    ax.fill_between(x, layer_1, layer_2, color="#00897b", alpha=0.7,
                    label=f"Tax (0.002% per trade)  total {cum_tax.iloc[-1]/1e3:.1f}K")
    ax.fill_between(x, layer_2, layer_3, color="#e65100", alpha=0.7,
                    label=f"Spread (half × 4)  total {cum_spread.iloc[-1]/1e3:.1f}K")
    ax.set_title("Cumulative transaction cost", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative cost (K NTD)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.show()


__all__ = ["simulate_midday_short_night_long"]
