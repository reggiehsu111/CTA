"""
signal_stats.py — comprehensive per-signal performance metrics.

Two entry points:

* `signal_stats(sig, ...)` — dict of ~30 metrics for one signal
* `batch_signal_stats(signals, ...)` — DataFrame with one row per signal

Metrics grouped by concern:

**Returns**
  cum_ret_gross_pct, cum_ret_net_pct, ann_ret_gross_pct, ann_ret_net_pct
  SR_gross, SR_net, sortino, calmar
  max_dd_pct, max_dd_days

**Alpha vs buy-and-hold**
  beta, alpha_ann_pct, corr, info_ratio

**Hit rate / distribution**
  win_rate, profit_factor, win_loss_ratio, best_day_pct, worst_day_pct
  ret_skew, ret_kurt, downside_dev

**Stability across time**
  yearly_sr_min, yearly_sr_max, yearly_sr_mean, yearly_sr_std
  SR_of_SR  (mean/std of yearly SR — how consistent the edge is)
  positive_years_pct  (fraction of years with positive SR)

**Trading intensity**
  held_pct, turnover_avg, turnover_ann, n_trades, tcost_pct_of_gross

**Metadata**
  sign (auto-flipped), n_bars, start_date, end_date
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Default TAIFEX MXF cost params — match cta.Simulate
_POINT_VALUE    = 50.0
_FIXED_PER_SIDE = 20.0
_FEE_RATE       = 0.00002


def _drawdown(cum: pd.Series) -> tuple[float, int]:
    """Max drawdown (as a % of peak) and its duration in bars, on a cumulative-return series."""
    if cum.empty:
        return float("nan"), 0
    running_max = cum.cummax()
    dd = cum - running_max                    # additive drawdown (cum PnL is already in units of return)
    max_dd = float(dd.min()) * 100
    # Duration: longest span from prior peak until recovery (or end of series)
    dur = 0; cur = 0
    prev_peak = -np.inf
    for i, v in enumerate(cum.values):
        if v > prev_peak:
            prev_peak = v
            cur = 0
        else:
            cur += 1
            dur = max(dur, cur)
    return max_dd, dur


def signal_stats(
    sig:            pd.Series,
    asset,
    start:          str | pd.Timestamp | None = None,
    end:            str | pd.Timestamp | None = None,
    exec_lag:       int   = 2,
    point_value:    float = _POINT_VALUE,
    fixed_per_side: float = _FIXED_PER_SIDE,
    fee_rate:       float = _FEE_RATE,
    auto_flip:      bool  = True,
    roll_adjusted:  bool  = False,
) -> dict:
    """Return a rich stats dict for one signal.

    Parameters
    ----------
    sig : pd.Series, date-indexed. Bounded in [-1, +1] is typical but not required.
    asset : cta.BaseAsset (must have `close`, `periods_per_year`).
    start, end : slice the eval window. Defaults to signal's own valid range.
    exec_lag : bars of shift (2 mirrors cta.Simulate).
    auto_flip : if True, flip sign so gross SR ≥ 0 and report the chosen sign.
    roll_adjusted : which return series to score against.
        False (default) = `close.pct_change()`, which on the 305 rollover days
        books the front/back calendar spread as P&L (mean 57 bps, max 352 bps).
        True = `asset.returns`, which prices the roll against yesterday's
        BACK-month close — what `cta.Simulate` has always used.
        Measured 2026-09-01 over the 11 registered signals the two disagree by
        up to 0.217 SR (opt_put_mo_oi_selftanh_w60: 0.308 raw vs 0.091 adjusted),
        mean |Δ| 0.063; buy-and-hold is 0.493 raw vs 0.700 adjusted. The default
        is left at False ONLY so that already-published gate numbers do not
        silently change underneath them — it is not the defensible choice.
        Pass True for new work and say which you used. See QNT-13.

    Returns
    -------
    dict of scalar metrics. Missing/undefined values are float('nan') so a
    DataFrame built from many of these has homogeneous dtypes.
    """
    close = asset["close"]
    ret   = asset.returns if roll_adjusted else close.pct_change()
    ppy   = int(asset.periods_per_year)

    sig = sig.reindex(asset.index).astype(float)
    exec_sig = sig.shift(exec_lag)
    pnl_g    = exec_sig * ret
    turnover = exec_sig.fillna(0).diff().abs()
    cost_pct = fixed_per_side / (close * point_value) + fee_rate
    tcost    = turnover * cost_pct

    if start is not None:
        pnl_g    = pnl_g.loc[start:]
        turnover = turnover.loc[start:]
        tcost    = tcost.loc[start:]
        exec_sig = exec_sig.loc[start:]
        ret_slice = ret.loc[start:]
    else:
        ret_slice = ret

    if end is not None:
        pnl_g    = pnl_g.loc[:end]
        turnover = turnover.loc[:end]
        tcost    = tcost.loc[:end]
        exec_sig = exec_sig.loc[:end]
        ret_slice = ret_slice.loc[:end]

    g = pnl_g.dropna()
    if len(g) < 10 or g.std() == 0 or not np.isfinite(g.std()):
        return {"sign": 0, "n_bars": int(len(g)), "note": "insufficient_data"}

    sr_g_raw = float(np.sqrt(ppy) * g.mean() / g.std())
    sign = -1 if (auto_flip and sr_g_raw < 0) else +1
    g_s = g * sign
    tc  = tcost.reindex(g.index).fillna(0)
    n_s = g_s - tc

    # ── Returns ────────────────────────────────────────────────────────────
    cum_gross = g_s.fillna(0).cumsum()
    cum_net   = n_s.fillna(0).cumsum()
    ann_ret_g = float(g_s.mean() * ppy) * 100
    ann_ret_n = float(n_s.mean() * ppy) * 100
    sr_g = float(np.sqrt(ppy) * g_s.mean() / g_s.std())
    sr_n = float(np.sqrt(ppy) * n_s.mean() / n_s.std()) if n_s.std() > 0 else float("nan")

    # Sortino — only downside variance
    downside = n_s[n_s < 0]
    dd_std   = downside.std() if len(downside) > 1 else np.nan
    sortino  = float(np.sqrt(ppy) * n_s.mean() / dd_std) if dd_std and dd_std > 0 else float("nan")

    # Max drawdown + duration
    max_dd_pct, max_dd_days = _drawdown(cum_net)

    # Calmar — annualized return / |max_dd|
    calmar = ann_ret_n / abs(max_dd_pct) if abs(max_dd_pct) > 1e-9 else float("nan")

    # ── Alpha / beta vs buy-and-hold ───────────────────────────────────────
    bh = ret_slice.reindex(g.index)
    j = pd.concat([n_s.rename("y"), bh.rename("x")], axis=1).dropna()
    if len(j) >= 30 and j["x"].var() > 0:
        cov  = np.cov(j["y"].values, j["x"].values, ddof=0)
        beta = float(cov[0, 1] / j["x"].var(ddof=0))
        alpha_daily = float(j["y"].mean() - beta * j["x"].mean())
        alpha_ann_pct = alpha_daily * ppy * 100
        corr = float(j["y"].corr(j["x"]))
        # Information ratio = alpha_ann / tracking_error_ann
        te = float(j["y"].std() - beta * j["x"].std())         # simple residual-vol approx
        te_ann = te * np.sqrt(ppy)
        info_ratio = (alpha_ann_pct / 100) / te_ann if te_ann and abs(te_ann) > 1e-9 else float("nan")
    else:
        beta = alpha_ann_pct = corr = info_ratio = float("nan")

    # ── Hit rate / distribution ────────────────────────────────────────────
    # "Held" bar = |exec_sig| > 0.01. Win rate is over held bars only.
    exec_s = exec_sig.reindex(g.index).fillna(0)
    active = exec_s.abs() > 0.01
    if active.any():
        win_rate = float((n_s[active] > 0).mean())
        wins  = n_s[active][n_s[active] > 0]
        losses= n_s[active][n_s[active] < 0]
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else float("nan")
        win_loss_ratio = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else float("nan")
    else:
        win_rate = profit_factor = win_loss_ratio = float("nan")

    ret_skew = float(n_s.skew())
    ret_kurt = float(n_s.kurt())
    best_day_pct  = float(n_s.max() * 100)
    worst_day_pct = float(n_s.min() * 100)
    downside_dev  = float(dd_std * np.sqrt(ppy) * 100) if dd_std and np.isfinite(dd_std) else float("nan")

    # ── Stability across calendar years ────────────────────────────────────
    yearly = n_s.groupby(n_s.index.year)
    def _yr_sr(x):
        return float(np.sqrt(ppy) * x.mean() / x.std()) if len(x) > 20 and x.std() > 0 else np.nan
    yearly_sr = yearly.apply(_yr_sr).dropna()
    if len(yearly_sr) >= 2:
        yr_sr_min  = float(yearly_sr.min())
        yr_sr_max  = float(yearly_sr.max())
        yr_sr_mean = float(yearly_sr.mean())
        yr_sr_std  = float(yearly_sr.std())
        sr_of_sr   = yr_sr_mean / yr_sr_std if yr_sr_std > 0 else float("nan")
        pos_years  = float((yearly_sr > 0).mean())
    else:
        yr_sr_min = yr_sr_max = yr_sr_mean = yr_sr_std = sr_of_sr = pos_years = float("nan")

    # ── Trading intensity ──────────────────────────────────────────────────
    held_pct     = float(active.mean()) * 100
    turnover_avg = float(turnover.reindex(g.index).fillna(0).mean())
    turnover_ann = turnover_avg * ppy
    n_trades     = int((turnover > 1e-9).sum())
    total_cost   = float(tc.sum())
    total_gross  = float(g_s.fillna(0).sum())
    tcost_pct_of_gross = (100 * total_cost / total_gross) if abs(total_gross) > 1e-9 else float("nan")

    return {
        # metadata
        "sign":            int(sign),
        "n_bars":          int(len(g)),
        "start_date":      str(g.index.min().date()),
        "end_date":        str(g.index.max().date()),
        # returns
        "cum_gross_pct":   round(float(cum_gross.iloc[-1]) * 100, 3),
        "cum_net_pct":     round(float(cum_net.iloc[-1])   * 100, 3),
        "ann_ret_gross_pct": round(ann_ret_g, 3),
        "ann_ret_net_pct":   round(ann_ret_n, 3),
        "SR_gross":        round(sr_g, 3),
        "SR_net":          round(sr_n, 3),
        "sortino":         round(sortino, 3) if np.isfinite(sortino) else float("nan"),
        "calmar":          round(calmar, 3) if np.isfinite(calmar) else float("nan"),
        "max_dd_pct":      round(max_dd_pct, 2),
        "max_dd_days":     int(max_dd_days),
        # alpha vs buy-hold
        "beta":            round(beta, 3) if np.isfinite(beta) else float("nan"),
        "alpha_ann_pct":   round(alpha_ann_pct, 3) if np.isfinite(alpha_ann_pct) else float("nan"),
        "corr_bh":         round(corr, 3) if np.isfinite(corr) else float("nan"),
        "info_ratio":      round(info_ratio, 3) if np.isfinite(info_ratio) else float("nan"),
        # hit rate / distribution
        "win_rate":        round(win_rate, 3) if np.isfinite(win_rate) else float("nan"),
        "profit_factor":   round(profit_factor, 3) if np.isfinite(profit_factor) else float("nan"),
        "win_loss_ratio":  round(win_loss_ratio, 3) if np.isfinite(win_loss_ratio) else float("nan"),
        "best_day_pct":    round(best_day_pct, 3),
        "worst_day_pct":   round(worst_day_pct, 3),
        "ret_skew":        round(ret_skew, 3),
        "ret_kurt":        round(ret_kurt, 3),
        "downside_dev_pct":round(downside_dev, 3) if np.isfinite(downside_dev) else float("nan"),
        # stability
        "yr_sr_min":       round(yr_sr_min, 3) if np.isfinite(yr_sr_min) else float("nan"),
        "yr_sr_max":       round(yr_sr_max, 3) if np.isfinite(yr_sr_max) else float("nan"),
        "yr_sr_mean":      round(yr_sr_mean, 3) if np.isfinite(yr_sr_mean) else float("nan"),
        "yr_sr_std":       round(yr_sr_std, 3) if np.isfinite(yr_sr_std) else float("nan"),
        "SR_of_SR":        round(sr_of_sr, 3) if np.isfinite(sr_of_sr) else float("nan"),
        "positive_years":  round(pos_years, 3) if np.isfinite(pos_years) else float("nan"),
        "n_years":         int(len(yearly_sr)),
        # trading intensity
        "held_pct":        round(held_pct, 2),
        "turnover_avg":    round(turnover_avg, 5),
        "turnover_ann":    round(turnover_ann, 2),
        "n_trades":        n_trades,
        "tcost_pct_of_gross": round(tcost_pct_of_gross, 2) if np.isfinite(tcost_pct_of_gross) else float("nan"),
    }


def batch_signal_stats(
    signals: dict[str, pd.Series],
    asset,
    start: str | pd.Timestamp | None = None,
    end:   str | pd.Timestamp | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run `signal_stats` on every signal in `signals`; return one-row-per-signal DataFrame.

    Sorted by `SR_net` descending. Signals with insufficient data are dropped.
    """
    rows = []
    for name, sig in signals.items():
        try:
            st = signal_stats(sig, asset, start=start, end=end, **kwargs)
            if st.get("note") == "insufficient_data":
                continue
            rows.append({"signal": name, **st})
        except Exception as e:
            rows.append({"signal": name, "note": f"error: {type(e).__name__}: {e}"})

    df = pd.DataFrame(rows)
    if "signal" in df.columns:
        df = df.set_index("signal")
    if "SR_net" in df.columns:
        df = df.sort_values("SR_net", ascending=False)
    return df


def composite_score(
    combined: pd.DataFrame,
    weights: dict[str, float] | None = None,
    clip_ratio: tuple[float, float] = (-2.0, 3.0),
) -> pd.DataFrame:
    """Blend multiple signal metrics into a single 0-1 composite score.

    Every component is percentile-ranked within the input population (so scales
    are comparable), then weighted and summed. Higher = better. Adds five
    `rank_*` columns showing per-dimension percentiles and a `composite_score`
    column that's the weighted mean.

    Inputs must include (from `batch_signal_stats` on IS + OOS join):
      * `SR_net`, `SR_net_oos`     — magnitude leg
      * `oos_is_ratio`             — consistency leg
      * `SR_of_SR`                 — stability leg
      * `turnover_ann`             — implementation-feasibility leg
      * `alpha_ann_pct`            — alpha-over-B&H leg

    Parameters
    ----------
    combined : joined IS+OOS stats DataFrame (from tbl_is.join(tbl_oos))
    weights : dict of {dimension: weight} — non-normalised. Default:
              {"magnitude": 0.30, "consistency": 0.15, "stability": 0.20,
               "turnover":  0.15, "alpha":       0.20}
              Any keys omitted are dropped; keys are normalised to sum to 1.
    clip_ratio : clip `oos_is_ratio` to this range before ranking so wild
                 outliers (near-zero IS) don't dominate.

    Returns
    -------
    combined augmented with:
      `min_abs_sr`, `oos_is_ratio_clipped`,
      `rank_magnitude`, `rank_consistency`, `rank_stability`,
      `rank_turnover`, `rank_alpha`,
      `composite_score`  (0 to 1; higher = better)
    """
    default_weights = {
        "magnitude":   0.30,
        "consistency": 0.15,
        "stability":   0.20,
        "turnover":    0.15,
        "alpha":       0.20,
    }
    w = dict(default_weights)
    if weights is not None:
        w.update(weights)
    # Normalise
    total = sum(w.values())
    if total <= 0:
        raise ValueError("weights sum to zero")
    w = {k: v / total for k, v in w.items()}

    out = combined.copy()

    # --- Magnitude: min(|SR_IS|, |SR_OOS|) ---
    out["min_abs_sr"] = np.minimum(out["SR_net"].abs(), out["SR_net_oos"].abs())

    # --- Consistency: OOS/IS ratio, clipped ---
    lo, hi = clip_ratio
    out["oos_is_ratio_clipped"] = out["oos_is_ratio"].clip(lower=lo, upper=hi)

    # --- Percentile ranks per dimension (0 = worst, 1 = best) ---
    #   higher raw value = better for magnitude / consistency / stability / alpha
    #   higher raw value = worse for turnover (so we negate before ranking)
    def _pct(series: pd.Series) -> pd.Series:
        # rank(method="average") then / N — pandas rank naturally NaN-preserving
        return series.rank(method="average", pct=True)

    out["rank_magnitude"]   = _pct(out["min_abs_sr"])
    out["rank_consistency"] = _pct(out["oos_is_ratio_clipped"])
    out["rank_stability"]   = _pct(out["SR_of_SR"])
    out["rank_turnover"]    = _pct(-out["turnover_ann"])
    out["rank_alpha"]       = _pct(out["alpha_ann_pct"])

    # --- Composite: weighted mean of percentiles ---
    dim_map = {
        "magnitude":   "rank_magnitude",
        "consistency": "rank_consistency",
        "stability":   "rank_stability",
        "turnover":    "rank_turnover",
        "alpha":       "rank_alpha",
    }
    composite = pd.Series(0.0, index=out.index)
    for dim, weight in w.items():
        col = dim_map.get(dim)
        if col is None:
            raise ValueError(f"unknown weight key {dim!r}. Choose from {list(dim_map)}")
        composite = composite + weight * out[col].fillna(0)
    out["composite_score"] = composite.round(4)

    return out.sort_values("composite_score", ascending=False)


__all__ = ["signal_stats", "batch_signal_stats", "composite_score"]
