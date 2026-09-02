"""QNT-104 step 7 — circular-shift null for the equal-weight BASKETS.

Step 5's `combEW` basket passes the four gates on `o2o` in both panels. Two of
24 baskets passing is only a result if 2/24 is more than a no-information grid
produces — QNT-99 measured the gates passing 10-15% of the time on random masks.
Same construction as step 5, but every feature is circularly shifted by a random
offset before the basket is built, so the basket keeps its exposure, turnover,
autocorrelation and cross-member correlation and loses only the alignment with
returns.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops

OUT, NREP = "/home/ubuntu/mtx/signal_zoo/qnt104", 200
TI = A.index
IS_END, O1, O2 = "2016-12-31", ("2017-01-01", "2021-12-31"), ("2022-01-01", None)
W, MIRROR_T = 60, 0.35
r_c2c = _RET["c2c"].reindex(TI).astype(float)

LEGS = {}
for ef in ("monthly", "all"):
    P = (pd.read_csv(f"{OUT}/call_features_{ef}.csv", index_col=0, parse_dates=True)
           .join(pd.read_csv(f"{OUT}/comb_features_{ef}.csv", index_col=0, parse_dates=True))
           .join(pd.read_csv(f"{OUT}/put_features_{ef}.csv", index_col=0, parse_dates=True))
           .reindex(TI))
    fam = lambda c: "call" if c.startswith("call_") else ("put" if c.startswith("put_") else "comb")
    keep = [c for c in P.columns
            if abs(float(P[c].corr(r_c2c))) <= MIRROR_T and P[c].notna().sum() > 1500]
    LEGS[ef] = (P[keep], [fam(c) for c in keep])

rng = np.random.default_rng(1041)
rows = []
for rep in range(NREP + 1):                      # rep 0 = the REAL basket, unshifted
    for ef, (P, fams) in LEGS.items():
        X = P.values.astype(float)
        if rep:
            X = np.column_stack([np.roll(X[:, j], int(rng.integers(250, len(TI) - 250)))
                                 for j in range(X.shape[1])])
        Z = np.tanh(np.column_stack([ops.robust_z(pd.Series(X[:, j], index=TI), W).values
                                     for j in range(X.shape[1])]) / 2.0)
        legs = {c: pd.Series(Z[:, j], index=TI) for j, c in enumerate(P.columns)}
        for v in ("c2c", "o2o", "day"):
            sg = {c: (wstats(s, v, end=IS_END) or {}).get("sign", 0) for c, s in legs.items()}
            for name in ("callEW", "combEW", "putEW", "allEW"):
                cols = [c for c, f in zip(P.columns, fams)
                        if sg.get(c) and (name == "allEW" or f == name[:-2])]
                if len(cols) < 3: continue
                b = pd.concat([legs[c] * sg[c] for c in cols], axis=1).mean(axis=1).clip(-1, 1)
                fu = wstats(b, v, sign=1)
                if fu is None: continue
                o1 = wstats(b, v, start=O1[0], end=O1[1], sign=1)
                o2 = wstats(b, v, start=O2[0], sign=1)
                rows.append(dict(rep=rep, panel=ef, basket=name, variant=v,
                                 SR_IS=(wstats(b, v, end=IS_END, sign=1) or {}).get("SR_net", np.nan),
                                 SR_OOS1=(o1 or {}).get("SR_net", np.nan),
                                 SR_OOS2=(o2 or {}).get("SR_net", np.nan),
                                 **{f"full_{k}": fu[k] for k in
                                    ("SR_net", "SR_of_SR", "positive_years", "n_years",
                                     "beta", "abs_exec_w")}))
    if rep % 20 == 0:
        print(f"  rep {rep}/{NREP}", flush=True)

d = cta.house_gates(pd.DataFrame(rows), prefix="full_", beta_mode="both")
d.to_csv(f"{OUT}/combine_null.csv", index=False)
real, null = d[d.rep == 0], d[d.rep > 0]
obs = int(real.passes.sum())
per = null.groupby("rep").passes.sum()
print(f"\n=== basket circular-shift null, {NREP} reps x {len(real)} baskets ===")
print(f"observed four-gate passers: {obs} of {len(real)}  ({obs/len(real):.1%})")
print(f"null: mean {per.mean():.2f} passers/rep (sd {per.std():.2f}), "
      f"per-basket no-information pass rate {null.passes.mean():.1%}; "
      f"P(null rep >= {obs}) = {(per >= obs).mean():.3f}")
for name in ("callEW", "combEW", "putEW", "allEW"):
    n_, r_ = null[null.basket == name], real[real.basket == name]
    print(f"  {name:7s}: observed pass {int(r_.passes.sum())}/{len(r_)}, "
          f"null pass rate {n_.passes.mean():.1%}; observed best SR_net {r_.full_SR_net.max():+.3f}, "
          f"null P(SR_net >= obs best) = {(n_.full_SR_net >= r_.full_SR_net.max()).mean():.3f}")
# the specific claim: positive in IS and BOTH held-out blocks
q = lambda x: ((x.SR_IS > 0) & (x.SR_OOS1 > 0) & (x.SR_OOS2 > 0)).mean()
print(f"\npositive in IS and both held-out blocks: observed {q(real):.1%} of baskets, "
      f"null {q(null):.1%}")
