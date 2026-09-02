"""QNT-104 step 5 — the "combine both sides" leg of the ticket, selection-free.

Individual cells are ranked by noise (step 4). A combination is the one thing a
grid this redundant can still test honestly, PROVIDED the membership is not
chosen from the results. So the baskets here are defined by a structural rule
only:

  callEW  — equal weight of every CALL-side feature that is not a price mirror
  combEW  — ... every COMBINED (call+put) feature that is not a price mirror
  putEW   — ... every PUT-side feature that is not a price mirror (QNT-99 set)
  allEW   — all three pooled

"not a price mirror" = |corr(feature_t, return_t)| <= 0.35, which needs no
forward return and no gate outcome. Each member is z-scored (robust_z, w60,
tanh-squashed to [-1,1]) so the basket is a mean of comparable positions, and
each member's SIGN IS FROZEN ON THE IS HALF ONLY, exactly as in the cell sweep.
Baskets are then scored on IS / OOS1 / OOS2 with no further choices.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops

OUT = "/home/ubuntu/mtx/signal_zoo/qnt104"
TI  = A.index
IS_END, O1, O2 = "2016-12-31", ("2017-01-01", "2021-12-31"), ("2022-01-01", None)
W, MIRROR_T = 60, 0.35

qa   = pd.read_csv(f"{OUT}/oi_diagnostics.csv")
rows = []
for ef in ("monthly", "all"):
    P = (pd.read_csv(f"{OUT}/call_features_{ef}.csv", index_col=0, parse_dates=True)
           .join(pd.read_csv(f"{OUT}/comb_features_{ef}.csv", index_col=0, parse_dates=True))
           .join(pd.read_csv(f"{OUT}/put_features_{ef}.csv", index_col=0, parse_dates=True))
           .reindex(TI))
    q = qa[qa.panel == ef].set_index("feature")
    FAM = {"call": [c for c in P.columns if c.startswith("call_")],
           "comb": [c for c in P.columns if not c.startswith(("call_", "put_"))],
           "put":  [c for c in P.columns if c.startswith("put_")]}
    # put-side mirror flags are recomputed here (QNT-99 measured the same thing)
    r_c2c = _RET["c2c"].reindex(TI).astype(float)
    mir = {c: abs(float(P[c].corr(r_c2c))) > MIRROR_T for c in P.columns}

    legs, members = {}, {}
    for fam, cols in FAM.items():
        keep = [c for c in cols if not mir.get(c, True) and P[c].notna().sum() > 1500]
        members[fam] = keep
        for c in keep:
            s = pd.Series(ops.robust_z(P[c].astype(float), W), index=TI)
            legs[c] = np.tanh(s.replace([np.inf, -np.inf], np.nan) / 2.0)

    for v in ("c2c", "o2o", "day"):
        signs = {}
        for c, s in legs.items():
            st = wstats(s, v, end=IS_END)
            signs[c] = st["sign"] if st else 0
        for name, cols in [("callEW", members["call"]), ("combEW", members["comb"]),
                           ("putEW", members["put"]),
                           ("allEW", members["call"] + members["comb"] + members["put"])]:
            cols = [c for c in cols if signs.get(c)]
            if len(cols) < 3: continue
            b = pd.concat([legs[c] * signs[c] for c in cols], axis=1).mean(axis=1)
            b = b.clip(-1, 1)
            is_ = wstats(b, v, end=IS_END, sign=1)
            o1  = wstats(b, v, start=O1[0], end=O1[1], sign=1)
            o2  = wstats(b, v, start=O2[0], sign=1)
            fu  = wstats(b, v, sign=1)
            if fu is None: continue
            rows.append(dict(panel=ef, basket=name, n_members=len(cols), variant=v,
                             SR_IS=is_["SR_net"], SR_OOS1=(o1 or {}).get("SR_net", np.nan),
                             SR_OOS2=(o2 or {}).get("SR_net", np.nan),
                             **{f"full_{k}": fu[k] for k in
                                ("SR_net", "SR_of_SR", "positive_years", "n_years", "beta",
                                 "abs_exec_w", "held_pct", "turnover_ann")}))
            print(f"  {ef:7s} {name:6s} n={len(cols):2d} {v:5s} members={cols if name!='allEW' else '...'}"
                  [:150])

cb = pd.DataFrame(rows)
cb = cta.house_gates(cb, prefix="full_", beta_mode="both")
cb.to_csv(f"{OUT}/oi_combos.csv", index=False)
print("\n=== QNT-104 step 5: equal-weight OI baskets, sign frozen IS, no member selection ===")
print(cb[["panel", "basket", "n_members", "variant", "SR_IS", "SR_OOS1", "SR_OOS2",
          "full_SR_net", "full_SR_of_SR", "full_positive_years", "full_beta", "beta_per_w",
          "full_abs_exec_w", "full_n_years", "n_gates", "passes"]].round(3).to_string(index=False))
print(f"\nbaskets passing all four house gates: {int(cb.passes.sum())} / {len(cb)}")
