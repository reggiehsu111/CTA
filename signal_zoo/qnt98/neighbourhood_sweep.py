"""QNT-98 (Reggie's clarification 2026-09-02): robustness = sweep the ARGUMENTS
around each working signal, on BOTH axes —

  * the window axis   : w = 20 ... 504   (not just the 3 the grids carried)
  * the OPERATOR axis : swap InstMean for InstStdev / InstSkew / ... exactly as
                        the ticket comment asks.

Design
------
For every candidate cell that passes (or nearly passes) the house gates in the
current-era grids, its NEIGHBOURHOOD is {all 15 operators} x {all 9 windows} on
the SAME source series and the SAME execution variant, scored under that
candidate's own published sample/IS convention so the claimed cell reproduces
bit-for-bit as a check.

Two things make this a real test rather than a victory lap:

 1. The palette is split into LOCATION operators (selfz/robustz/bdtanh/rankc/
    signth/dev — all monotone functions of x, so they agree BY CONSTRUCTION and
    prove nothing) and SWAP operators (dispersion/shape/change statistics, which
    are not monotone in x). Reggie's InstStdev/InstSkew ask is the second group.
    Coherence is reported separately for each.
 2. A CONTROL set of series that passed nothing gets the identical treatment.
    If a winner's neighbourhood is no more coherent than a control's, the winner
    is the top of a noise distribution, not a signal.

Discipline (台指期 standing brief): PIT inputs via ctx.macro (load_macro_tw),
roll-adjusted legs re-derived from _base.py, realistic costs, sign FROZEN on the
IS half. Nothing here selects a sign, a variant or writes any config.

Reuses macro_window_sweep.py's setup block verbatim (_RET/_COST/_SHIFT/wstats)
so every number is comparable to the QNT-14/18/19 evidence. Writes only to
signal_zoo/qnt98/.
"""
import sys, os, io, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")

SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
_marker = "# ── Build the signals once"
assert _marker in _src, "macro_window_sweep.py changed shape - re-check the split point"
exec(compile(_src.split(_marker)[0], SWEEP, "exec"))          # noqa: S102
# -> ctx, A, wstats, _RET, _COST, _SHIFT, PV, FIXED, FEE, PPY

import numpy as np, pandas as pd
import cta
from cta.signals import _operators as ops

OUT = "/home/ubuntu/mtx/signal_zoo/qnt98"
NIGHT_START = str(A["night_close"].first_valid_index().date())

# ── Cost ladders. wstats reads _COST[variant] at call time, so we swap the
#    dict entry around a call the same way the harness swaps _SHIFT. ────────
_ENTRY_P = {"c2c": A["close"].astype(float), "o2o": _prev_o_adj,
            "day": A["open"].astype(float), "ongap": A["night_close"].astype(float)}
COST_LADDER = {
    "gross":  {v: p * 0.0            for v, p in _ENTRY_P.items()},
    "stub":   {v: 20.0 / (p * PV) + 0.00002 for v, p in _ENTRY_P.items()},
    "real":   {v: 70.0 / (p * PV) + 0.00004 for v, p in _ENTRY_P.items()},
    "real3x": {v: 210.0 / (p * PV) + 0.00012 for v, p in _ENTRY_P.items()},
}

def wstats_at(sig, variant, cost="real", lag=None, **kw):
    """wstats with a swappable cost ladder and an explicit execution lag.

    `lag` matters: slow_window_sweep.py holds shift(2) on EVERY variant for the
    non-daily inputs, because a monthly with pub_lag 16-90d does not publish
    overnight TPE and so cannot legitimately trade at shift(1). Only the daily
    US-close series earn day/ongap shift(1). Passing lag=None keeps the
    harness default (_SHIFT), which is the daily convention.
    """
    keep_c = _COST[variant]
    keep_s = _SHIFT[variant]
    _COST[variant] = COST_LADDER[cost][variant]
    if lag is not None:
        _SHIFT[variant] = lag
    try:
        return wstats(sig, variant, **kw)
    finally:
        _COST[variant] = keep_c
        _SHIFT[variant] = keep_s

# ── The operator palette ──────────────────────────────────────────────────
# LOCATION: monotone in x. These are the six the published grids swept.
# SWAP:     Reggie's ask - the window statistic itself is replaced.
def _d(x):  return x.diff()
def _mp(w): return max(3, w // 2)

LOCATION = {
    "selfz":   lambda x, w: ops.selfz(x, w),                       # (x-InstMean)/InstStdev
    "robustz": lambda x, w: ops.robust_z(x, w),
    "bdtanh":  lambda x, w: ops.bd_selftanh(x, w),
    "rankc":   lambda x, w: ops.rank_c(x, w),
    "signth":  lambda x, w: ops.sign_thresh(x, w),
    "dev":     lambda x, w: ops.dev(x, w),                         # x - InstMean
}
SWAP = {
    # --- InstMean -> InstStdev / InstSkew / kurtosis, applied to the level ---
    "instStdev": lambda x, w: cta.InstStdev(w, x),
    "instSkew":  lambda x, w: cta.InstSkew(w, x),
    "instKurt":  lambda x, w: x.rolling(w, min_periods=_mp(w)).kurt(),
    "minmax":    lambda x, w: ((x - x.rolling(w, min_periods=_mp(w)).min()) /
                               (x.rolling(w, min_periods=_mp(w)).max()
                                - x.rolling(w, min_periods=_mp(w)).min()).replace(0, np.nan)),
    # --- the same statistics on the CHANGE of the series -------------------
    "chg":       lambda x, w: x - x.shift(w),
    "slope":     lambda x, w: _slope(x, w),
    "dStdev":    lambda x, w: cta.InstStdev(w, _d(x)),
    "dSkew":     lambda x, w: cta.InstSkew(w, _d(x)),
    "ac1":       lambda x, w: _d(x).rolling(w, min_periods=_mp(w)).corr(_d(x).shift(1)),
    "pctpos":    lambda x, w: (_d(x) > 0).astype(float).rolling(w, min_periods=_mp(w)).mean(),
}

def _slope(x, w):
    """OLS slope of x on t over a rolling window, closed form (no .apply)."""
    n = w
    t = pd.Series(np.arange(len(x), dtype=float), index=x.index)
    mp = _mp(w)
    sx = t.rolling(n, min_periods=mp).mean()
    sy = x.rolling(n, min_periods=mp).mean()
    cxy = (t * x).rolling(n, min_periods=mp).mean() - sx * sy
    vxx = (t * t).rolling(n, min_periods=mp).mean() - sx * sx
    return cxy / vxx.replace(0, np.nan)

OPS = {**LOCATION, **SWAP}
FAMILY = {**{k: "location" for k in LOCATION}, **{k: "swap" for k in SWAP}}
WINDOWS = (20, 40, 60, 90, 120, 180, 252, 378, 504)

# ── Candidates: every gate-passing family head in the current-era grids, plus
#    the G1 near-miss. `conv` fixes the sample + IS/OOS split to the one the
#    published grid used for that cell, so the claimed number reproduces. ───
NIGHT = dict(start=NIGHT_START, is_end="2021-12-31", oos_start="2022-01-01")
FULL  = dict(start=None,        is_end="2018-12-31", oos_start="2019-01-01")

CANDIDATES = [
    # series,           kind,   variant, claim_op,  claim_w, grid,       conv
    ("cny_usd",         "level","ongap", "dev",     252, "G2 night", NIGHT),
    ("us_dxy_broad",    "level","o2o",   "signth",  252, "G2 night", NIGHT),
    ("us_dgs30",        "level","day",   "dev",     120, "G2 night", NIGHT),
    ("twd_usd",         "level","day",   "dev",     252, "G2 night", NIGHT),
    ("kr_kospi",        "yoy",  "ongap", "dev",     252, "G3 slow",  FULL),
    ("us_semi_ip",      "yoy",  "ongap", "selfz",   252, "G3 slow",  FULL),
    ("us_semi_ip_nsa",  "yoy",  "ongap", "rankc",   252, "G3 slow",  FULL),
    ("copper",          "yoy",  "ongap", "bdtanh",  252, "G3 slow",  FULL),
    ("epu_global",      "level","ongap", "robustz", 252, "G3 slow",  FULL),
    ("us_empire_state", "level","ongap", "robustz", 252, "G3 slow",  FULL),
    ("igrea",           "level","c2c",   "selfz",   252, "G1 near",  FULL),
]
# Control: series that passed NO gate anywhere, scored on the same variants.
CONTROLS = [
    ("krw_usd",           "level","ongap", NIGHT),
    ("wti",               "level","ongap", NIGHT),
    ("us_cfnai",          "level","ongap", FULL),
    ("cn_exports",        "yoy",  "ongap", FULL),
    ("us_recession_prob", "level","ongap", FULL),
    ("us_dgs5",           "level","day",   NIGHT),
    ("us_real_10y",       "level","o2o",   NIGHT),
    ("us_stlfsi",         "level","c2c",   FULL),
]

cat = cta.macro_catalog()
def raw_series(sid, kind):
    if kind == "yoy":
        per = {"M": 12, "Q": 4, "D": 252, "W": 52}.get(cat.loc[sid, "freq"], 12)
        return ctx.macro_yoy(sid, per).astype(float)
    return ctx.macro(sid).astype(float)

# ── Sweep ─────────────────────────────────────────────────────────────────
def sweep_one(sid, kind, variant, conv, role, claim_op=None, claim_w=None, grid="",
              end=None):
    x = raw_series(sid, kind)
    freq = str(cat.loc[sid, "freq"])
    lag = None if freq == "D" else 2          # see wstats_at docstring
    eff_lag = _SHIFT[variant] if lag is None else lag
    out, positions = [], {}
    for opn, fn in OPS.items():
        for w in WINDOWS:
            s = fn(x, w).replace([np.inf, -np.inf], np.nan)
            s = cta.normalize_signal(s, method="tanh", window=252)
            if s.dropna().empty:
                continue
            is_st = wstats_at(s, variant, "real", lag=lag,
                              start=conv["start"], end=conv["is_end"])
            if is_st is None:
                continue
            sign = is_st["sign"]
            oos  = wstats_at(s, variant, "real", lag=lag,
                              start=conv["oos_start"], end=end, sign=sign)
            full = wstats_at(s, variant, "real", lag=lag,
                             start=conv["start"], end=end, sign=sign)
            if full is None:
                continue
            row = dict(role=role, grid=grid, series=sid, kind=kind, variant=variant,
                       freq=freq, shift=eff_lag,
                       op=opn, op_family=FAMILY[opn], window=w, sign_IS=sign,
                       is_claim=int(opn == claim_op and w == claim_w),
                       SR_IS=is_st["SR_net"], SR_OOS=(oos or {}).get("SR_net", np.nan),
                       **{k: full[k] for k in
                          ("SR_net", "SR_gross", "SR_of_SR", "positive_years", "yr_sr_min",
                           "n_years", "beta", "mean_exec_w", "abs_exec_w", "max_dd_pct",
                           "max_dd_days", "turnover_ann", "held_pct", "n_bars",
                           "start_date", "end_date")})
            for cname in ("gross", "stub", "real3x"):
                st = wstats_at(s, variant, cname, lag=lag,
                               start=conv["start"], end=end, sign=sign)
                row[f"SR_{cname}"] = (st or {}).get("SR_net", np.nan)
            out.append(row)
            positions[f"{opn}|w{w}"] = (s.reindex(A.index).shift(eff_lag) * sign)
    return pd.DataFrame(out), positions

# ── Reproduction check: re-score the 11 claim cells capped at the grids' own
#    last bar (2026-08-31) and diff against the published numbers. ─────────
GRID_END = "2026-08-31"
_g2 = pd.read_csv("/home/ubuntu/mtx/signal_zoo/qnt19_postfloor/window_sweep_full.csv")
_g2 = _g2[_g2["regime"] == "night"]
_g3 = pd.read_csv("/home/ubuntu/mtx/signal_zoo/qnt19_postfloor/slow_window_sweep.csv")
_g1 = pd.read_csv("/home/ubuntu/mtx/signal_zoo/qnt19_postfloor/full_sweep.csv")

def published(sid, op, w, v, grid):
    if grid.startswith("G2"):
        d = _g2[(_g2["series"] == sid) & (_g2["transform"] == op)
                & (_g2["window"] == w) & (_g2["variant"] == v)]
    elif grid.startswith("G3"):
        d = _g3[(_g3["series"] == sid) & (_g3["transform"] == op)
                & (_g3["window"] == w) & (_g3["variant"] == v)]
    else:
        d = _g1[(_g1["series"] == sid) & (_g1["transform"] == op) & (_g1["window"] == w)]
        return float(d["SR_full"].iloc[0]) if len(d) else np.nan
    return float(d["SR_net"].iloc[0]) if len(d) else np.nan

repro = []
for sid, kind, v, cop, cw, grid, conv in CANDIDATES:
    x = raw_series(sid, kind)
    freq = str(cat.loc[sid, "freq"])
    lag = None if freq == "D" else 2
    s_ = cta.normalize_signal(OPS[cop](x, cw).replace([np.inf, -np.inf], np.nan),
                              method="tanh", window=252)
    is_ = wstats_at(s_, v, "real", lag=lag, start=conv["start"], end=conv["is_end"])
    fu_ = wstats_at(s_, v, "real", lag=lag, start=conv["start"], end=GRID_END,
                    sign=is_["sign"])
    repro.append(dict(series=sid, variant=v, op=cop, window=cw, grid=grid,
                      published=published(sid, cop, cw, v, grid), recomputed=fu_["SR_net"]))
RP = pd.DataFrame(repro)
RP["diff"] = RP.recomputed - RP.published
RP.to_csv(f"{OUT}/claim_reproduction.csv", index=False)
print("\n=== reproduction of the 11 claim cells, capped at the grids' last bar "
      f"({GRID_END}) ===")
print(RP.round(4).to_string(index=False))
print(f"max |diff| = {RP['diff'].abs().max():.4f}\n")

rows, POS = [], {}
for sid, kind, v, cop, cw, grid, conv in CANDIDATES:
    df, pos = sweep_one(sid, kind, v, conv, "candidate", cop, cw, grid)
    rows.append(df); POS[sid] = pos
    print(f"  {sid:18s} {v:6s} cells={len(df):3d}  claim SR_net="
          f"{df.loc[df.is_claim == 1, 'SR_net'].round(3).tolist()}", flush=True)
for sid, kind, v, conv in CONTROLS:
    df, pos = sweep_one(sid, kind, v, conv, "control")
    rows.append(df); POS[sid] = pos
    print(f"  [ctl] {sid:13s} {v:6s} cells={len(df):3d}", flush=True)

G = pd.concat(rows, ignore_index=True)
for g, c in [("gate_srsr", G.SR_of_SR > 0.6), ("gate_posyr", G.positive_years >= 0.65),
             ("gate_beta", G.beta.abs() < 0.15), ("gate_nyr", G.n_years >= 5)]:
    G[g] = c
G["n_gates"] = G[["gate_srsr", "gate_posyr", "gate_beta", "gate_nyr"]].sum(axis=1)
G.to_csv(f"{OUT}/neighbourhood_sweep.csv", index=False)
print(f"\ncells: {len(G)}  -> neighbourhood_sweep.csv")

# ── Neighbourhood redundancy: how many INDEPENDENT tests is a neighbourhood? ─
red = []
for sid, pos in POS.items():
    P = pd.DataFrame(pos).dropna(how="all")
    C = P.corr().values
    off = C[~np.eye(len(C), dtype=bool)]
    ev = np.linalg.eigvalsh(np.nan_to_num(C, nan=0.0))
    ev = ev[ev > 0]
    neff = float(ev.sum() ** 2 / (ev ** 2).sum())
    fam = pd.Series({k: FAMILY[k.split("|")[0]] for k in P.columns})
    loc, sw = P.loc[:, (fam == "location").values], P.loc[:, (fam == "swap").values]
    cl = loc.corrwith(loc.mean(axis=1)).mean()
    red.append(dict(series=sid, n_cells=P.shape[1], mean_abs_corr=float(np.abs(off).mean()),
                    n_eff_positions=neff,
                    loc_vs_swap_corr=float(loc.mean(axis=1).corr(sw.mean(axis=1)))))
R = pd.DataFrame(red)
R.to_csv(f"{OUT}/neighbourhood_redundancy.csv", index=False)
print(R.round(3).to_string(index=False))
