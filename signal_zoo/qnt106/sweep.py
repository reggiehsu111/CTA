"""QNT-106 step 3 — diagnostics + gate sweep on the 台股 market-internals axis.

PRE-RUN POWER TARGET (QNT-78 rule 2, QNT-94 n_eff correction) — stated before
the run, not after:
    target d = 0.12 SR, sd = 0.13
    S_required = 2.33 * (2.80 * 0.13 / 0.12)^2 = 22 source series
    S available = 29 NEW source series (6 breadth, 3 xsec, 6 liquidity,
                  7 institutional flow, 7 leverage)
    -> d_min = 4.26 * 0.13 / sqrt(29) = 0.103.
Adequate for d >= 0.12 and nothing smaller. Raw S is optimistic (series inside
a family correlate); the realised n_eff is MEASURED from the per-feature PnLs
and reported, never assumed.

PIT. Every series is a Taiwan EOD print: TWSE spot/margin ~15:00-21:00 TPE,
三大法人 ~15:00-16:00. So
    c2c / o2o  shift(2)   day / ongap  shift(1)     -- all safe
    night      EXCLUDED   night_open[t] = 15:00 of t-1, i.e. the same afternoon
                          the print lands. Not safely after publication.

OUT-OF-SAMPLE. Sign frozen on IS, never re-chosen.
  full  (c2c/o2o/day)          IS 2009..2017 | OOS1 2018-2021 | OOS2 2022-2026
  night (+ongap, 2017-05-16..) IS 2017-05..2022 | OOS 2023-2026
"""
import os, sys, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))  # noqa: S102

import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops
from scipy import stats
sys.path.insert(0, "/home/ubuntu/mtx/signal_zoo/qnt106")
from features import load as load_features, FAMILY

OUT = "/home/ubuntu/mtx/signal_zoo/qnt106"
TI  = A.index
F   = load_features().reindex(TI)
FEATS = list(F.columns)

TRANSFORMS = {"selfz": ops.selfz, "robustz": ops.robust_z, "bdtanh": ops.bd_selftanh,
              "rankc": ops.rank_c, "signth": ops.sign_thresh, "dev": ops.dev}
WINDOWS = (20, 60, 120, 252)
NIGHT_START = str(A["night_close"].first_valid_index().date())
REGIMES = [
    ("full",  None,        "2017-12-31", ("2018-01-01","2021-12-31"), ("2022-01-01", None),
     ("c2c","o2o","day")),
    ("night", NIGHT_START, "2022-12-31", ("2023-01-01", None),        (None, None),
     ("c2c","o2o","day","ongap")),
]

# ── diagnostics: price mirror? era-stable? ────────────────────────────────
r_c2c = _RET["c2c"].reindex(TI).astype(float)
fwd   = r_c2c.shift(-2)
ls    = np.log(A["close"].astype(float).reindex(TI))
rows = []
for f in FEATS:
    x = F[f].astype(float)
    blocks = {}
    for lo, hi in [(2009,2013),(2014,2018),(2019,2022),(2023,2026)]:
        m_ = (x.index.year >= lo) & (x.index.year <= hi)
        j = pd.concat([x[m_], fwd[m_]], axis=1).dropna()
        blocks[f"IC_{lo}_{hi}"] = (stats.spearmanr(j.iloc[:,0], j.iloc[:,1])[0]
                                   if len(j) > 200 else np.nan)
    ics = [v for v in blocks.values() if v == v]
    h1 = (x.index.year <= 2017); h2 = ~h1
    rows.append(dict(family=FAMILY[f], feature=f, n=int(x.notna().sum()),
                     corr_same_day=x.corr(r_c2c), **blocks,
                     IC_sign_agree=(float(np.mean(np.sign(ics) == np.sign(ics[0])))
                                    if ics else np.nan),
                     corr_spot_h1=x[h1].corr(ls[h1]), corr_spot_h2=x[h2].corr(ls[h2])))
qa = pd.DataFrame(rows); qa.to_csv(f"{OUT}/diagnostics.csv", index=False)
print("=== QNT-106 diagnostics ===")
print(qa.round(3).to_string(index=False), flush=True)
MIRROR = sorted(qa.loc[qa.corr_same_day.abs() > 0.35, "feature"])
print(f"\nprice mirrors (|corr_same_day| > 0.35): {MIRROR or 'none'}", flush=True)

# ── the gate sweep ────────────────────────────────────────────────────────
rows, pnl = [], {}
for reg, r_start, is_end, (o1s, o1e), (o2s, o2e), variants in REGIMES:
    for f in FEATS:
        x = F[f].astype(float)
        if (x.loc[r_start:] if r_start else x).notna().sum() < 600:
            continue
        for tn, tf in TRANSFORMS.items():
            for w in WINDOWS:
                sig = pd.Series(tf(x, w), index=TI).replace([np.inf, -np.inf], np.nan)
                sig = cta.normalize_signal(sig, method="tanh", window=252)
                if (sig.loc[r_start:] if r_start else sig).notna().sum() < 400:
                    continue
                for v in variants:
                    is_ = wstats(sig, v, start=r_start, end=is_end)
                    if is_ is None: continue
                    sg = is_["sign"]
                    o1 = wstats(sig, v, start=o1s, end=o1e, sign=sg) if o1s else None
                    o2 = wstats(sig, v, start=o2s, end=o2e, sign=sg) if o2s else None
                    full = wstats(sig, v, start=r_start, sign=sg)
                    if full is None: continue
                    cell = f"{reg}|{f}|{tn}|w{w}|{v}"
                    rec = dict(cell=cell, regime=reg, series=f, family=FAMILY[f],
                               transform=tn, window=w, variant=v, sign=sg,
                               SR_IS=is_["SR_net"],
                               SR_OOS1=(o1 or {}).get("SR_net", np.nan),
                               SR_OOS2=(o2 or {}).get("SR_net", np.nan),
                               is_mirror=f in MIRROR)
                    for k, val in full.items():
                        rec[k if k not in rec else f"full_{k}"] = val
                    rec["beta_per_w"] = (full["beta"] / full["abs_exec_w"]
                                         if full["abs_exec_w"] > 1e-6 else np.nan)
                    rows.append(rec)
                    pos = sig.shift(_SHIFT[v]) * sg
                    g = (pos * _RET[v]).reindex(TI)
                    tc = pos.fillna(0).diff().abs() * _COST[v]
                    pnl[cell] = (g - tc.reindex(TI).fillna(0))
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/sweep_full.csv", index=False)
pd.DataFrame(pnl).to_pickle(f"{OUT}/sweep_pnl.pkl")
print(f"\ncells: {len(res)}   series: {res.series.nunique()}", flush=True)

GATE = ((res.SR_of_SR > 0.6) & (res.positive_years >= 0.65)
        & (res.beta.abs() < 0.15) & (res.n_years >= 5))
res.loc[GATE].sort_values("SR_of_SR", ascending=False).to_csv(f"{OUT}/sweep_gated.csv", index=False)
print(f"four-gate passers: {int(GATE.sum())} of {len(res)}")
print(res.groupby("regime").apply(lambda d: pd.Series(
    {"cells": len(d), "gate_pass": int(((d.SR_of_SR>0.6)&(d.positive_years>=0.65)
      &(d.beta.abs()<0.15)&(d.n_years>=5)).sum()),
     "median_SR_net": d.SR_net.median()})).to_string())
