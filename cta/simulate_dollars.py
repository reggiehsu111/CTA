"""
simulate_dollars.py — dollar-PnL simulation for a CTA signal on MTX-style futures.

Unlike `simulate()` which returns fractional weights and percentage PnL,
this models a real-world account: take `dollar_amount` NTD as collateral,
round positions to integer contracts, apply per-trade fixed cost, tax,
and bid-ask spread; include rollover cost at monthly expiry.

Entry point
-----------
    simulate_by_dollars(signal, dollar_amount=1_000_000)
"""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .asset import BaseAsset
from . import operators as _ops
from .simulate import _load_asset, _sharpe, normalize_signal


# ─────────────────────────────────────────────────────────────────────────────
# Cost helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trade_costs(
    contracts: pd.Series,
    close:     pd.Series,
    spread:    pd.Series,
    point_value:     float,
    fixed_per_side:  float,
    fee_rate:        float,
    spread_fraction: float,
) -> dict:
    """Per-day NTD trading costs for a position series in integer contracts."""
    turnover  = contracts.diff().abs().fillna(contracts.abs())
    notional  = close * point_value

    fixed_c   = turnover * fixed_per_side
    tax_c     = turnover * notional * fee_rate
    spread_c  = turnover * spread.fillna(0) * point_value * spread_fraction

    return {
        "fixed":    fixed_c.fillna(0),
        "tax":      tax_c.fillna(0),
        "spread":   spread_c.fillna(0),
        "total":    (fixed_c + tax_c + spread_c).fillna(0),
        "turnover": turnover.fillna(0),
    }


def _rollover_costs(
    contracts:    pd.Series,
    close:        pd.Series,
    spread:       pd.Series,
    expiry_mask:  pd.Series,
    point_value:     float,
    fixed_per_side:  float,
    fee_rate:        float,
    spread_fraction: float,
) -> pd.Series:
    """
    Cost of rolling the current position on each expiry day.
    A roll = close front + reopen back, so 2× the unit cost of |current position|.
    """
    units    = contracts.abs().where(expiry_mask, 0).fillna(0)
    notional = close * point_value

    fixed_c  = units * fixed_per_side * 2
    tax_c    = units * notional * fee_rate * 2
    spread_c = units * spread.fillna(0) * point_value * spread_fraction * 2

    return (fixed_c + tax_c + spread_c).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    signal_normalized: pd.Series,
    asset_obj:         BaseAsset,
    dollar_amount:     float,
    point_value:       float,
    fixed_per_side:    float,
    fee_rate:          float,
    spread_fraction:   float,
    include_rollover:  bool,
) -> dict:
    """One dollar-PnL simulation; returns time-series of PnL components and costs."""
    close  = asset_obj.close
    bid    = asset_obj.bid
    ask    = asset_obj.ask
    spread = (ask - bid).reindex(close.index).fillna(0)

    exec_sig = signal_normalized.shift(2)

    # Position sizing: contracts active during the [t-1, t] return interval,
    # sized off close[t-1] (the last observable price before the interval).
    sizing_price        = close.shift(1).replace(0, np.nan)
    fractional_contracts = exec_sig * dollar_amount / (sizing_price * point_value)
    integer_contracts    = fractional_contracts.round().fillna(0)

    # Daily price change in NTD per contract
    price_change = (close - close.shift(1)) * point_value

    paper_pnl        = (fractional_contracts * price_change).fillna(0)
    actual_pnl_gross = (integer_contracts    * price_change).fillna(0)

    costs = _trade_costs(integer_contracts, close, spread, point_value,
                         fixed_per_side, fee_rate, spread_fraction)

    if include_rollover:
        try:
            ev          = _ops.Event("optexp")
            expiry_mask = (ev == 0)
        except Exception:
            expiry_mask = pd.Series(False, index=close.index)
        rollover = _rollover_costs(integer_contracts, close, spread, expiry_mask,
                                   point_value, fixed_per_side, fee_rate, spread_fraction)
    else:
        rollover = pd.Series(0.0, index=close.index)

    total_cost      = (costs["total"] + rollover).fillna(0)
    actual_pnl_net  = actual_pnl_gross - total_cost

    return {
        "paper_pnl":            paper_pnl,
        "actual_pnl_gross":     actual_pnl_gross,
        "actual_pnl_net":       actual_pnl_net,
        "fractional_contracts": fractional_contracts,
        "integer_contracts":    integer_contracts,
        "fixed_cost":           costs["fixed"],
        "tax_cost":             costs["tax"],
        "spread_cost":          costs["spread"],
        "rollover_cost":        rollover,
        "total_cost":           total_cost,
        "turnover":             costs["turnover"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def simulate_by_dollars(
    signal:         pd.Series,
    dollar_amount:  float = 1_000_000,
    *,
    asset:           "str | BaseAsset" = "mtx",
    time_granularity: str = "1d",
    contract:         str = "front",
    normalize:        "bool | str" = True,
    norm_window:      int = 252,
    norm_sensitivity: float = 1.0,
    point_value:      float = 50.0,
    fixed_per_side:   float = 20.0,
    fee_rate:         float = 0.00002,
    spread_fraction:  float = 0.5,
    include_rollover: bool = True,
    compare_amounts:  list | None = None,
    compare_spreads:  list | None = None,
) -> dict:
    """
    Simulate the *actual dollar* P&L of `signal` on MTX (or any TAIFEX index future).

    Models real-world trading: integer contract sizing, fixed-per-side fee,
    transaction tax, bid-ask spread, and monthly rollover at expiry.

    Parameters
    ----------
    signal           : raw signal, normalized internally if `normalize` is set.
    dollar_amount    : account size in NTD. Default 1,000,000.
    asset            : 'mtx' or another TAIFEX code, or a `BaseAsset`.
    contract         : 'front' (近月) or 'back' (次月). Only used if `asset` is a string.
    normalize        : True → tanh-z-score; False → use signal as-is (must be in [-1, 1]).
    point_value      : NTD per index point per contract (50 for MTX, 200 for TX).
    fixed_per_side   : NTD per contract per buy/sell. Default 20.
    fee_rate         : tax rate per side as fraction of notional. Default 0.00002 (0.002%).
    spread_fraction  : fraction of bid-ask spread paid per trade. 0 = mid-fill,
                       0.5 = half-spread, 1.0 = full-cross.
    include_rollover : if True, charge full round-trip costs on expiry days
                       proportional to the current position size.
    compare_amounts  : dollar amounts for the comparison subplot
                       (default: [10K, 100K, 1M, 10M] NTD).
    compare_spreads  : spread fractions for the comparison subplot
                       (default: [0, 0.25, 0.5, 1.0]).

    Returns
    -------
    dict with keys: signal_normalized, asset_obj, main, amount_runs,
                    spread_runs, rollover_off.
    """
    # ── Load asset & set context ────────────────────────────────────────────
    if isinstance(asset, BaseAsset):
        asset_obj = asset
    else:
        asset_obj = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj)

    # ── Normalize signal ────────────────────────────────────────────────────
    sig = signal.reindex(asset_obj.index).astype(float)
    if normalize is True:
        signal_norm = normalize_signal(sig, method="tanh",
                                       window=norm_window, sensitivity=norm_sensitivity)
    elif isinstance(normalize, str):
        signal_norm = normalize_signal(sig, method=normalize,
                                       window=norm_window, sensitivity=norm_sensitivity)
    else:
        signal_norm = sig

    # ── Main run + comparison sweeps ────────────────────────────────────────
    if compare_amounts is None:
        compare_amounts = [10_000, 100_000, 1_000_000, 10_000_000]
    if compare_spreads is None:
        compare_spreads = [0.0, 0.25, 0.5, 1.0]

    main = _run_one(signal_norm, asset_obj, dollar_amount,
                    point_value, fixed_per_side, fee_rate,
                    spread_fraction, include_rollover)

    amount_runs = {
        amt: _run_one(signal_norm, asset_obj, amt,
                      point_value, fixed_per_side, fee_rate,
                      spread_fraction, include_rollover)
        for amt in compare_amounts
    }

    spread_runs = {
        spr: _run_one(signal_norm, asset_obj, dollar_amount,
                      point_value, fixed_per_side, fee_rate,
                      spr, include_rollover)
        for spr in compare_spreads
    }

    rollover_off = _run_one(signal_norm, asset_obj, dollar_amount,
                            point_value, fixed_per_side, fee_rate,
                            spread_fraction, False)

    _plot_dollars(main, amount_runs, spread_runs, rollover_off, dollar_amount)

    return {
        "signal_normalized": signal_norm,
        "asset_obj":         asset_obj,
        "main":              main,
        "amount_runs":       amount_runs,
        "spread_runs":       spread_runs,
        "rollover_off":      rollover_off,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_dollars(main, amount_runs, spread_runs, rollover_off, dollar_amount):
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.22)

    # (0,0) Paper vs actual gross vs actual net
    ax = fig.add_subplot(gs[0, 0])
    cum_paper = main["paper_pnl"].cumsum()
    cum_gross = main["actual_pnl_gross"].cumsum()
    cum_net   = main["actual_pnl_net"].cumsum()
    ax.plot(cum_paper.index, cum_paper.values / 1e3, color="#1565c0", linewidth=1.3,
            label=f"Paper PnL (fractional)  end {cum_paper.iloc[-1]/1e3:,.1f}K")
    ax.plot(cum_gross.index, cum_gross.values / 1e3, color="#00897b", linewidth=1.3,
            label=f"Actual PnL gross (integer)  end {cum_gross.iloc[-1]/1e3:,.1f}K")
    ax.plot(cum_net.index,   cum_net.values   / 1e3, color="#c62828", linewidth=1.3,
            label=f"Actual PnL net (after costs)  end {cum_net.iloc[-1]/1e3:,.1f}K")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title(f"Cumulative PnL — paper vs actual  (D = {dollar_amount:,.0f} NTD)", fontsize=14)
    ax.set_ylabel("Cumulative PnL (K NTD)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (0,1) Dollar-amount comparison — show PnL in % of initial dollar
    ax = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("viridis")
    n = max(len(amount_runs) - 1, 1)
    for i, (amt, run) in enumerate(amount_runs.items()):
        cum   = run["actual_pnl_net"].cumsum()
        pct   = cum / amt * 100
        sr    = _sharpe(run["actual_pnl_net"])
        label = (f"{amt/1_000_000:.0f}M NTD" if amt >= 1_000_000
                 else f"{amt/1000:.0f}K NTD")
        ax.plot(pct.index, pct.values, color=cmap(i / n), linewidth=1.2,
                label=f"{label}  (SR {sr:+.2f}, end {pct.iloc[-1]:+.1f}%)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Cumulative net PnL — % of dollar amount", fontsize=14)
    ax.set_ylabel("PnL (% of D)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (1,0) Bid-ask spread sensitivity
    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.get_cmap("Reds")
    n    = max(len(spread_runs) - 1, 1)
    for i, (spr, run) in enumerate(spread_runs.items()):
        cum_net = run["actual_pnl_net"].cumsum()
        sr      = _sharpe(run["actual_pnl_net"])
        ax.plot(cum_net.index, cum_net.values / 1e3,
                color=cmap(0.3 + 0.6 * i / n), linewidth=1.2,
                label=f"{spr*100:.0f}% of spread  (SR {sr:+.2f}, "
                      f"end {cum_net.iloc[-1]/1e3:+.1f}K)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Cumulative net PnL — bid-ask spread sensitivity", fontsize=14)
    ax.set_ylabel("Cumulative PnL (K NTD)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (1,1) With vs without rollover
    ax = fig.add_subplot(gs[1, 1])
    cum_with    = main["actual_pnl_net"].cumsum()
    cum_without = rollover_off["actual_pnl_net"].cumsum()
    sr_with     = _sharpe(main["actual_pnl_net"])
    sr_without  = _sharpe(rollover_off["actual_pnl_net"])
    ax.plot(cum_with.index,    cum_with.values    / 1e3, color="#c62828", linewidth=1.3,
            label=f"with rollover  (SR {sr_with:+.2f}, end {cum_with.iloc[-1]/1e3:+.1f}K)")
    ax.plot(cum_without.index, cum_without.values / 1e3, color="#1565c0", linewidth=1.3,
            label=f"without rollover  (SR {sr_without:+.2f}, end {cum_without.iloc[-1]/1e3:+.1f}K)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_title("Cumulative net PnL — monthly rollover impact", fontsize=14)
    ax.set_ylabel("Cumulative PnL (K NTD)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (2,0) Position in contracts over time
    ax = fig.add_subplot(gs[2, 0])
    contracts = main["integer_contracts"]
    colors    = ["#00897b" if c >= 0 else "#c62828" for c in contracts.fillna(0)]
    ax.bar(contracts.index, contracts.values, color=colors, alpha=0.6, width=1.5)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
    avg_abs = contracts.abs().mean()
    max_abs = contracts.abs().max()
    ax.set_title(f"Position (integer contracts)  avg |c| = {avg_abs:.2f}  max |c| = {int(max_abs)}",
                 fontsize=14)
    ax.set_ylabel("Contracts")
    ax.grid(True, alpha=0.3)

    # (2,1) Cumulative cost breakdown — stacked area
    ax = fig.add_subplot(gs[2, 1])
    cum_fixed    = main["fixed_cost"].cumsum()
    cum_tax      = main["tax_cost"].cumsum()
    cum_spread   = main["spread_cost"].cumsum()
    cum_rollover = main["rollover_cost"].cumsum()

    layer_1 = cum_fixed.values / 1e3
    layer_2 = (cum_fixed + cum_tax).values / 1e3
    layer_3 = (cum_fixed + cum_tax + cum_spread).values / 1e3
    layer_4 = (cum_fixed + cum_tax + cum_spread + cum_rollover).values / 1e3

    ax.fill_between(cum_fixed.index, 0, layer_1, color="#1565c0", alpha=0.7,
                    label=f"Fixed (20 NTD/side)  total {cum_fixed.iloc[-1]/1e3:.1f}K")
    ax.fill_between(cum_fixed.index, layer_1, layer_2, color="#00897b", alpha=0.7,
                    label=f"Tax (0.002%)  total {cum_tax.iloc[-1]/1e3:.1f}K")
    ax.fill_between(cum_fixed.index, layer_2, layer_3, color="#e65100", alpha=0.7,
                    label=f"Spread  total {cum_spread.iloc[-1]/1e3:.1f}K")
    ax.fill_between(cum_fixed.index, layer_3, layer_4, color="#c62828", alpha=0.7,
                    label=f"Rollover  total {cum_rollover.iloc[-1]/1e3:.1f}K")
    ax.set_title("Cumulative cost breakdown", fontsize=14)
    ax.set_ylabel("Cumulative cost (K NTD)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.show()


__all__ = ["simulate_by_dollars"]
