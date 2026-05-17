"""
decomposition.py — regime decomposition of a CTA signal's PnL.

`RegimeDecomposition(decompose_num, signal, factor_dict)`

For each named factor:
  1. Normalize the factor to (-1, 1) (rolling tanh z-score).
  2. Split the [-1, 1] range into `decompose_num` equal-width buckets.
     For decompose_num=5 the edges are [-1, -0.6, -0.2, 0.2, 0.6, 1].
  3. Compute the signal's daily PnL = Lag(2, normalize(signal)) * Returns(-1).
  4. For each bucket, plot the cumulative PnL restricted to days when the
     factor (lagged 2 to match the signal lag) sits in that bucket.

Equivalent to f.ReturnsDecomposition but along the time axis: regimes are
date sets, not stock fractiles.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .asset import BaseAsset
from . import operators as _ops
from .simulate import _load_asset, _sharpe, normalize_signal


def RegimeDecomposition(
    decompose_num: int,
    signal: pd.Series,
    factor_dict: dict[str, pd.Series],
    *,
    asset: "str | BaseAsset" = "mtx",
    time_granularity: str = "1d",
    contract: str = "front",
    normalize: "bool | str" = True,
    norm_window: int = 252,
    norm_sensitivity: float = 1.0,
    factor_norm_window: "int | None" = None,
) -> dict:
    """
    Plot `decompose_num` cumulative-PnL curves per factor, one curve per regime.

    Parameters
    ----------
    decompose_num : int
        Number of equal-width regime buckets across [-1, 1].
    signal : pd.Series
        Raw signal, date-indexed. Normalized internally if `normalize` is set.
    factor_dict : dict[str, pd.Series]
        {name: factor_series}. Each factor is normalized to (-1, 1) internally
        and lagged 2 bars to align with the signal's execution lag.
    asset : str | BaseAsset
        Contract code (e.g. 'mtx', 'TX') or an existing BaseAsset.
    contract : 'front' | 'back'
        Used only when `asset` is a string.
    normalize, norm_window, norm_sensitivity
        Applied to the *signal* — same as `cta.simulate`.
    factor_norm_window : int | None
        Rolling tanh-z-score window used to normalize each **factor** to (-1, 1).
        Defaults to `norm_window` if not specified. Use a smaller value
        (e.g. 63) to make regimes more responsive to recent factor changes.

    Returns
    -------
    dict with keys:
        pnl, cum_pnl           : the signal's gross daily PnL series
        normalized_factors     : {name: normalized factor in (-1, 1)}
        bucket_pnls            : {name: {k: daily PnL Series masked to bucket k}}
    """
    if factor_norm_window is None:
        factor_norm_window = norm_window
    if decompose_num < 2:
        raise ValueError("decompose_num must be >= 2.")
    if not factor_dict:
        raise ValueError("factor_dict cannot be empty.")

    # ── 1. Asset & active context ─────────────────────────────────────────────
    if isinstance(asset, BaseAsset):
        asset_obj = asset
    else:
        asset_obj = _load_asset(asset, time_granularity, contract)
    _ops.set_active_asset(asset_obj)

    # ── 2. Signal → PnL (same model as simulate) ──────────────────────────────
    sig = signal.reindex(asset_obj.index).astype(float)
    if normalize is True:
        sig_n = normalize_signal(sig, method="tanh",
                                 window=norm_window, sensitivity=norm_sensitivity)
    elif isinstance(normalize, str):
        sig_n = normalize_signal(sig, method=normalize,
                                 window=norm_window, sensitivity=norm_sensitivity)
    else:
        sig_n = sig

    exec_sig = sig_n.shift(2)
    ret      = asset_obj.returns
    pnl      = (exec_sig * ret).rename("pnl")
    cum_pnl  = pnl.cumsum().rename("cum_pnl")

    # ── 3. Normalize factors to (-1, 1) and lag 2 ─────────────────────────────
    normalized_factors = {}
    lagged_factors     = {}
    for name, factor in factor_dict.items():
        f = factor.reindex(asset_obj.index).astype(float)
        f_n = normalize_signal(f, method="tanh",
                               window=factor_norm_window,
                               sensitivity=norm_sensitivity)
        normalized_factors[name] = f_n
        lagged_factors[name]     = f_n.shift(2)   # align with signal execution lag

    # ── 4. Bucket edges across [-1, 1] ────────────────────────────────────────
    edges = np.linspace(-1.0, 1.0, decompose_num + 1)

    # ── 5. Plot — one subplot per factor ──────────────────────────────────────
    n_fac  = len(factor_dict)
    n_cols = min(2, n_fac)
    n_rows = math.ceil(n_fac / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(13 * n_cols, 6 * n_rows),
        constrained_layout=True,
    )
    axs = np.atleast_2d(axs)
    if n_cols == 1:
        axs = axs.reshape(-1, 1)

    # Build a dark-saturated diverging palette (avoid washed-out middle band).
    if decompose_num == 3:
        _palette = ["#1565c0", "#424242", "#c62828"]
    elif decompose_num == 5:
        _palette = ["#0d47a1", "#42a5f5", "#424242", "#ef5350", "#b71c1c"]
    elif decompose_num == 7:
        _palette = ["#0d47a1", "#1976d2", "#64b5f6",
                    "#424242",
                    "#ef9a9a", "#e53935", "#b71c1c"]
    else:
        _palette = [plt.get_cmap("coolwarm")(k / (decompose_num - 1))
                    for k in range(decompose_num)]

    def _bucket_color(k: int):
        return _palette[k]

    bucket_pnls: dict[str, dict[int, pd.Series]] = {}

    for i, (name, fac_lag) in enumerate(lagged_factors.items()):
        r, c = divmod(i, n_cols)
        ax = axs[r, c]

        bucket_pnls[name] = {}
        valid_days = fac_lag.notna() & pnl.notna()
        n_valid    = int(valid_days.sum())

        for k in range(decompose_num):
            lo, hi = edges[k], edges[k + 1]
            # last bucket is inclusive on the right
            if k == decompose_num - 1:
                mask = (fac_lag >= lo) & (fac_lag <= hi)
            else:
                mask = (fac_lag >= lo) & (fac_lag < hi)

            pnl_k = pnl.where(mask, 0.0)
            cum_k = pnl_k.cumsum()

            n_days = int((mask & valid_days).sum())
            pct    = (n_days / n_valid * 100) if n_valid else 0.0
            sr     = _sharpe(pnl[mask].dropna()) if n_days > 1 else float("nan")
            sr_str = f"{sr:+.2f}" if not np.isnan(sr) else "n/a"

            label = f"{k + 1}  SR {sr_str}  ({pct:.0f}%, {n_days}d)"
            ax.plot(cum_k.index, cum_k.values,
                    linewidth=1.4, color=_bucket_color(k), label=label, alpha=0.95)

            bucket_pnls[name][k] = pnl_k

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title(f"{name}  —  signal PnL decomposed by regime", fontsize=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return (contribution from bucket)")
        ax.legend(fontsize=10, loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_fac, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axs[r, c].axis("off")

    plt.show()

    return {
        "pnl":                pnl,
        "cum_pnl":            cum_pnl,
        "normalized_factors": normalized_factors,
        "bucket_pnls":        bucket_pnls,
    }


__all__ = ["RegimeDecomposition"]
