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
from .simulate import _load_asset, _sharpe


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

        t = 日盤 close     →  SHORT  `contracts` contracts
        t = 夜盤 open      →  BUY  2 × `contracts`  (close short + open long)
        t = 夜盤 close     →  SELL  `contracts`     (close long)

    Net effect per active day: short the mid-day gap, long the night session.

    Active days are those with non-NaN `night_open` and `night_close` (TAIFEX
    introduced after-hours sessions in mid-2017). Pre-2017 days contribute 0
    P&L and 0 cost.

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

    idx = asset_obj.index
    if start_date is not None:
        idx = idx[idx >= pd.Timestamp(start_date)]
    if end_date is not None:
        idx = idx[idx <= pd.Timestamp(end_date)]
    if len(idx) == 0:
        raise ValueError(f"No data in date range [{start_date}, {end_date}]")

    close       = asset_obj.close.reindex(idx)
    night_open  = asset_obj.night_open.reindex(idx)
    night_close = asset_obj.night_close.reindex(idx)
    spread      = (asset_obj.ask - asset_obj.bid).reindex(idx).fillna(0.0)

    # Days where both 夜盤 prints exist
    active = close.notna() & night_open.notna() & night_close.notna()

    # ── 2. Gross P&L (NTD per day) ──────────────────────────────────────────
    # Short position from day-close to night-open:
    midday_pts = -(night_open - close)             # NEG of forward move
    # Long position from night-open to night-close:
    night_pts  = night_close - night_open
    pnl_gross  = (contracts * point_value * (midday_pts + night_pts)).where(active, np.nan)

    # ── 3. Transaction costs ────────────────────────────────────────────────
    # 4 transactions per active day (1 sell-open, 2 buy, 1 sell-close).
    fixed_per_day  = pd.Series(4.0 * contracts * fixed_per_side, index=close.index)

    # Tax: per side, on each transaction's notional.
    # short at close (1×), buy at night_open (2×), sell at night_close (1×).
    tax_per_day = (
        close + 2.0 * night_open + night_close
    ) * point_value * fee_rate * contracts

    # Spread cost: 4 round-trips through (a fraction of) the bid-ask.
    spread_per_day = 4.0 * spread * point_value * spread_fraction * contracts

    cost_total = (fixed_per_day + tax_per_day + spread_per_day).where(active, 0.0).fillna(0.0)

    # ── 4. Net P&L and cumulative ───────────────────────────────────────────
    pnl_net   = (pnl_gross - cost_total).where(active, np.nan)
    cum_gross = pnl_gross.fillna(0).cumsum()
    cum_net   = pnl_net.fillna(0).cumsum()

    cum_fixed  = fixed_per_day.where(active, 0).fillna(0).cumsum()
    cum_tax    = tax_per_day.where(active, 0).fillna(0).cumsum()
    cum_spread = spread_per_day.where(active, 0).fillna(0).cumsum()
    cum_cost   = cost_total.cumsum()

    # ── 5. Buy-and-hold benchmark (long `contracts` from start to end) ──────
    # Daily mark-to-market on close-to-close moves. The first day contributes
    # zero (no prior close to compare to within the window).
    bh_daily   = (contracts * point_value * close.diff()).fillna(0.0)
    cum_buyhold = bh_daily.cumsum()

    # ── 6. Plot ─────────────────────────────────────────────────────────────
    _plot_midday_short_night_long(
        cum_gross, cum_net, pnl_gross, pnl_net,
        cum_fixed, cum_tax, cum_spread, cum_cost,
        cum_buyhold, bh_daily,
        contracts, active, idx,
    )

    return {
        "active":      active,
        "pnl_gross":   pnl_gross,
        "pnl_net":     pnl_net,
        "cum_gross":   cum_gross,
        "cum_net":     cum_net,
        "cum_buyhold": cum_buyhold,
        "pnl_buyhold": bh_daily,
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
    contracts:  int,
    active:     pd.Series,
    idx:        pd.DatetimeIndex,
) -> None:
    sr_gross   = _sharpe(pnl_gross)
    sr_net     = _sharpe(pnl_net)
    sr_buyhold = _sharpe(pnl_buyhold.replace(0, np.nan))
    n_active   = int(active.sum())

    date_range = f"{idx[0].date()} → {idx[-1].date()}"

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2], "hspace": 0.30})

    # ── Top: cumulative PnL — gross vs net vs buy-and-hold ──────────────────
    ax = axes[0]
    ax.plot(cum_gross.index, cum_gross.values / 1e3, color="#1565c0", linewidth=1.4,
            label=f"Strategy gross  (SR {sr_gross:+.2f}, end {cum_gross.iloc[-1]/1e3:+,.1f}K)")
    ax.plot(cum_net.index,   cum_net.values   / 1e3, color="#c62828", linewidth=1.4,
            label=f"Strategy net  (SR {sr_net:+.2f}, end {cum_net.iloc[-1]/1e3:+,.1f}K)")
    ax.plot(cum_buyhold.index, cum_buyhold.values / 1e3, color="#424242", linewidth=1.2,
            linestyle="--", alpha=0.85,
            label=f"Buy & hold  (SR {sr_buyhold:+.2f}, end {cum_buyhold.iloc[-1]/1e3:+,.1f}K)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(
        f"Mid-day short  +  Night long   ({contracts} contract"
        f"{'s' if contracts != 1 else ''}, {n_active:,} active days,  {date_range})",
        fontsize=14,
    )
    ax.set_ylabel("Cumulative PnL (K NTD)")
    ax.legend(fontsize=11, loc="upper left")
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
