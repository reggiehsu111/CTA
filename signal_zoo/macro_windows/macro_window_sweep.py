"""QNT-14: the 11 DAILY macro series at their true execution window.

QNT-12 scored every one of its 522 candidates at exec_lag=2 (the `c2c`
convention). Correct for the monthlies; too conservative for the daily US-close
series, which land overnight TPE and can legitimately trade the next Taipei
session at shift(1). This re-runs the 11 daily series on `ongap`, `day` and
`o2o` alongside `c2c` so the value of the tighter window is isolated.

Discipline (台指期 standing brief):
  * inputs PIT-aligned via ctx.macro (load_macro_tw), never load_macro
  * return legs re-derived from cta/signals/_base.py's variant definitions,
    NOT close.pct_change() - see _RET below for the night-labelling note
  * roll-adjusted: c2c uses asset.returns; o2o gets an explicit calendar-spread
    adjustment on the 305 roll days; day/ongap are intra-contract by
    construction so carry no roll exposure
  * realistic costs (fixed_per_side 70, fee_rate 4e-5), priced off each
    variant's OWN entry price
  * sign frozen on the in-sample half and carried into OOS unchanged
  * FULL grid written out, with mean(exec_w) and beta beside every Sharpe
  * nothing here selects a sign or a recommended variant - that is Reggie's call
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context
from cta.signals._base import _o2o_ret, _prev_open_cc

OUT = os.environ.get("MTX_SWEEP_OUT", "/home/ubuntu/mtx/signal_zoo/macro_windows")
PV, FIXED, FEE = 50.0, 70.0, 0.00004        # REAL costs, per side
PPY = 252

# ── The 11 daily US-close / 24h-market series named in QNT-14 ──────────────
DAILY = ["us_dxy_broad", "us_real_10y", "us_breakeven_10y", "us_breakeven_5y5y",
         "us_dgs5", "us_dgs30", "us_term_premium_10y", "twd_usd", "krw_usd",
         "cny_usd", "wti"]

TRANSFORMS = {
    "selfz":   lambda x, w: ops.selfz(x, w),
    "robustz": lambda x, w: ops.robust_z(x, w),
    "bdtanh":  lambda x, w: ops.bd_selftanh(x, w),
    "rankc":   lambda x, w: ops.rank_c(x, w),
    "signth":  lambda x, w: ops.sign_thresh(x, w),
    "dev":     lambda x, w: ops.dev(x, w),
}
WINDOWS = (60, 120, 252)

ctx = build_context()
A = ctx.asset

# ── Return + cost legs, re-derived from _base.py ───────────────────────────
# TAIFEX night labelling (cta/signals/_base.py, verified to the tick):
#     night_open[t]  = 15:00 of t-1        night_close[t] = 05:00 of t
# `ongap` is 05:00 t -> 08:45 t, so it is open[t] / night_close[t] SAME INDEX -
# no shift. night_close.shift(1) would reach 05:00 of t-1 and span the whole
# day session of t-1: that is the sign flip that took ongap +1.97 -> -0.04.
_o  = A["open"].astype(float)
_c  = A["close"].astype(float)
_nc = A["night_close"].astype(float)
_bc = A["back_close"].astype(float)
_roll = A.is_rollover

# o2o roll adjustment (QNT-52). This used to be a local close-measured
# approximation - yesterday's open scaled by back_close/close - written when the
# asset carried no back_OPEN column. QNT-21 added `back_open` to the loader and
# measured the approximation on all 305 roll days: mean |exact - approx| 31.7
# bps against mean |raw - exact| 47.0 bps, i.e. it removed only 32.4% of the
# roll error. It is gone. The sweeps now call _base.py's own o2o definition, so
# the sweep and the production runner share ONE roll convention by construction.
#
# MTX_O2O_APPROX=1 reproduces the retired behaviour for the before/after grid
# only, by hiding back_open so _prev_open_cc falls back to its close-measured
# branch (algebraically identical to the deleted line). Never set it for a
# result you intend to report.
_RET_ASSET = A
if os.environ.get("MTX_O2O_APPROX") == "1" and "back_open" in A.columns:
    _RET_ASSET = A.drop(columns=["back_open"])
    print("!! MTX_O2O_APPROX=1 - o2o uses the RETIRED 32%-effective approximation")
_prev_o_adj = _prev_open_cc(_RET_ASSET)

_RET = {
    "c2c":   A.returns,                        # close[t]/continuous_prev_close[t]-1
    "o2o":   _o2o_ret(_RET_ASSET),             # _base.py, exact via back_open
    "day":   _c / _o - 1,                      # 08:45 -> 13:45, intra-contract
    "ongap": _o / _nc - 1,                     # 05:00 -> 08:45, intra-contract
}
# entry price of each window, for the per-contract fixed cost
_ENTRY = {"c2c": _c, "o2o": _prev_o_adj, "day": _o, "ongap": _nc}
_COST  = {k: FIXED / (p * PV) + FEE for k, p in _ENTRY.items()}

# Shift lags. `pub_lag_days = 0` on these series means load_macro_tw puts the
# US observation of date D on TW index D - so sig[t] is not known until the US
# print lands, ~05:00-07:00 TPE of t+1.  See the PIT note in the write-up:
# this is ONE DAY MORE AGGRESSIVE than load_us_index_tw's pit_lag_days=1, so
# the us_* signals' shift_override={"o2o":1} must NOT be copied here.
_SHIFT = {"c2c": 2, "o2o": 2, "day": 1, "ongap": 1}


def wstats(sig, variant, start=None, end=None, sign=None):
    """signal_stats-equivalent metrics for one (signal, variant) pair.

    Mirrors cta/signal_stats.py's definitions exactly (SR_of_SR = mean/std of
    calendar-year SR, positive_years = fraction of years with SR>0, beta vs the
    buy-and-hold of the SAME window) so numbers are comparable to the gate table.
    """
    ret, cost, lag = _RET[variant], _COST[variant], _SHIFT[variant]
    s = sig.reindex(A.index).astype(float)
    pos = s.shift(lag)
    g_all = pos * ret
    to    = pos.fillna(0).diff().abs()
    tc_all = to * cost

    sl = slice(start, end)
    g = (g_all.loc[sl]).dropna()
    if len(g) < 60 or not np.isfinite(g.std()) or g.std() == 0:
        return None
    if sign is None:
        sign = -1 if float(np.sqrt(PPY) * g.mean() / g.std()) < 0 else 1
    g_s = g * sign
    tc  = tc_all.reindex(g.index).fillna(0)
    n_s = g_s - tc
    exec_w = (pos * sign).reindex(g.index)

    sr_g = float(np.sqrt(PPY) * g_s.mean() / g_s.std())
    sr_n = float(np.sqrt(PPY) * n_s.mean() / n_s.std()) if n_s.std() > 0 else np.nan

    cum = n_s.fillna(0).cumsum()
    dd = cum - cum.cummax()
    dur = cur = 0; peak = -np.inf
    for v in cum.values:
        if v > peak: peak, cur = v, 0
        else: cur += 1; dur = max(dur, cur)

    bh = ret.reindex(g.index)
    j = pd.concat([n_s.rename("y"), bh.rename("x")], axis=1).dropna()
    if len(j) >= 30 and j["x"].var() > 0:
        beta = float(np.cov(j["y"], j["x"], ddof=0)[0, 1] / j["x"].var(ddof=0))
    else:
        beta = np.nan

    yr = n_s.groupby(n_s.index.year).apply(
        lambda x: float(np.sqrt(PPY) * x.mean() / x.std()) if len(x) > 20 and x.std() > 0 else np.nan
    ).dropna()
    if len(yr) >= 2 and yr.std() > 0:
        sr_of_sr, pos_yr, yr_min = float(yr.mean() / yr.std()), float((yr > 0).mean()), float(yr.min())
    else:
        sr_of_sr = pos_yr = yr_min = np.nan

    return dict(sign=int(sign), n_bars=len(g), SR_gross=sr_g, SR_net=sr_n,
                SR_of_SR=sr_of_sr, positive_years=pos_yr, yr_sr_min=yr_min,
                n_years=int(len(yr)), beta=beta,
                mean_exec_w=float(exec_w.mean()), abs_exec_w=float(exec_w.abs().mean()),
                max_dd_pct=float(dd.min()) * 100, max_dd_days=int(dur),
                turnover_ann=float(to.reindex(g.index).fillna(0).mean()) * PPY,
                held_pct=float((exec_w.abs() > 0.01).mean()) * 100,
                start_date=str(g.index.min().date()), end_date=str(g.index.max().date()))


# ── Build the signals once ─────────────────────────────────────────────────
raw = {}
for sid in DAILY:
    raw[sid] = ctx.macro(sid).astype(float)

SIGS = {}
for sid, x in raw.items():
    for tn, tf in TRANSFORMS.items():
        for w in WINDOWS:
            s = tf(x, w).replace([np.inf, -np.inf], np.nan)
            s = cta.normalize_signal(s, method="tanh", window=252)
            if s.dropna().empty:
                continue
            SIGS[f"{sid}|{tn}|w{w}"] = s
print(f"signals built: {len(SIGS)}")

# Two sample regimes. `ongap` needs night_close, which only exists from
# 2017-05-16, so it CANNOT be compared to a 2001-start c2c number. The
# night-era block re-scores all four variants on the identical window.
NIGHT_START = str(A["night_close"].first_valid_index().date())
REGIMES = [
    ("full",  None,        "2018-12-31", "2019-01-01", ("c2c", "o2o", "day")),
    ("night", NIGHT_START, "2021-12-31", "2022-01-01", ("c2c", "o2o", "day", "ongap")),
]

rows = []
for reg, r_start, is_end, oos_start, variants in REGIMES:
    for cand, s in SIGS.items():
        sid, tn, w = cand.split("|")
        for v in variants:
            try:
                is_st = wstats(s, v, start=r_start, end=is_end)
                if is_st is None:
                    continue
                sign = is_st["sign"]
                oos = wstats(s, v, start=oos_start, sign=sign)
                full = wstats(s, v, start=r_start, sign=sign)
                if full is None:
                    continue
                rows.append(dict(
                    regime=reg, cand=cand, series=sid, transform=tn, window=int(w[1:]),
                    variant=v, shift=_SHIFT[v], sign_IS=sign,
                    SR_IS=is_st["SR_net"], SR_OOS=(oos or {}).get("SR_net", np.nan),
                    **{k: full[k] for k in
                       ("SR_net", "SR_gross", "SR_of_SR", "positive_years", "yr_sr_min",
                        "n_years", "beta", "mean_exec_w", "abs_exec_w", "max_dd_pct",
                        "max_dd_days", "turnover_ann", "held_pct", "n_bars",
                        "start_date", "end_date")}))
            except Exception as e:
                rows.append(dict(regime=reg, cand=cand, series=sid, transform=tn,
                                 window=int(w[1:]), variant=v,
                                 note=f"{type(e).__name__}: {e}"[:90]))

df = pd.DataFrame(rows)
df["gate_srsr"]  = df["SR_of_SR"] > 0.6
df["gate_posyr"] = df["positive_years"] >= 0.65
df["gate_beta"]  = df["beta"].abs() < 0.15
df["gate_nyr"]   = df["n_years"] >= 5
df["n_gates"]    = df[["gate_srsr", "gate_posyr", "gate_beta", "gate_nyr"]].sum(axis=1)
df.to_csv(f"{OUT}/window_sweep_full.csv", index=False)
print(f"cells: {len(df)}  -> window_sweep_full.csv")

# ── Buy-and-hold reference per window, same costs, same regimes ────────────
print("\n=== buy-and-hold reference (roll-adjusted, gross) ===")
for reg, r_start, _, _, variants in REGIMES:
    for v in variants:
        r = _RET[v].loc[r_start:].dropna()
        print(f"  {reg:5s} {v:5s} SR {np.sqrt(PPY)*r.mean()/r.std():+.3f}  "
              f"ann.vol {r.std()*np.sqrt(PPY):.3f}  n={len(r)}")


# ── Decomposition: does the gain come from the LAG or from the WINDOW? ─────
# `day` differs from `c2c` on two axes at once: the 5h intraday window AND
# shift(1) instead of shift(2). Re-scoring every variant at BOTH lags splits
# the two. shift(2) on `day` keeps the window but throws the freshness away.
rows2 = []
for reg, r_start, is_end, oos_start, variants in REGIMES:
    for cand, s in SIGS.items():
        for v in variants:
            for lag in (1, 2):
                _SHIFT[v], keep = lag, _SHIFT[v]
                try:
                    is_st = wstats(s, v, start=r_start, end=is_end)
                    if is_st is None:
                        continue
                    full = wstats(s, v, start=r_start, sign=is_st["sign"])
                    oos = wstats(s, v, start=oos_start, sign=is_st["sign"])
                    if full:
                        rows2.append(dict(regime=reg, cand=cand, variant=v, lag=lag,
                                          sign_IS=is_st["sign"], SR_IS=is_st["SR_net"],
                                          SR_OOS=(oos or {}).get("SR_net", np.nan),
                                          SR_net=full["SR_net"], SR_of_SR=full["SR_of_SR"],
                                          positive_years=full["positive_years"],
                                          n_years=full["n_years"], beta=full["beta"],
                                          mean_exec_w=full["mean_exec_w"]))
                finally:
                    _SHIFT[v] = keep
d2 = pd.DataFrame(rows2)
d2.to_csv(f"{OUT}/lag_decomposition.csv", index=False)
print(f"\ndecomposition cells: {len(d2)} -> lag_decomposition.csv")


# ── QNT-32 / QNT-25 reporting line ────────────────────────────────────────
# QNT-14's headline (+0.071 median dSR, 126/198 cells, p<0.001) counted each of
# 198 transform-window cells as a test. QNT-25 measured ICC(series)=0.50 on this
# very grid: n_eff ~ 21, and a PAIRED dSR carries its own, larger SE
# (SE(d) ~ SE(SR)*sqrt(2(1-rho)) ~ 0.22 at rho 0.42). Collapse to one number per
# source series and print the noise floor beside the effect.
print(f"\n{'='*100}\n=== QNT-25 REPORTING LINE (quote this, not the cell-level p) ===")
for _reg, _rs, _, _, _variants in REGIMES:
    _d = df[df.regime == _reg]
    _p = _d.pivot_table(index=["series", "transform", "window"],
                        columns="variant", values="SR_net").reset_index()
    _ny = float(_d.n_years.median())
    cta.sweep_headline(_d[_d.variant == "c2c"], "SR_net", n_years=_ny,
                       label=f"[{_reg}] daily-macro window sweep, SR_net on c2c").print()
    for _v in [c for c in _p.columns if c in _variants and c != "c2c"]:
        _q = _p.dropna(subset=[_v, "c2c"])
        cta.paired_headline(_q[_v], _q["c2c"], series=_q["series"], n_years=_ny,
                            label=f"[{_reg}] daily-macro window sweep",
                            a_name=_v, b_name="c2c").print()
