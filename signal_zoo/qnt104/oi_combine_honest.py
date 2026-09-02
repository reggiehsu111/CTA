"""QNT-104 step 8 — the honest error bar on the ONE non-null read.

Step 7's circular-shift null says the MEDIAN SR across the 24 equal-weight
baskets is above the no-information level out of sample (P = 0.000 on SR_OOS).
That p-value is optimistic and must not be quoted alone: the null shifts every
member INDEPENDENTLY, which decorrelates the basket's members and shrinks the
null basket's sampling error below the real one's. The correct denominator is
the real baskets' own redundancy.

So: rebuild the 24 real baskets, keep their PnLs, and test the aggregate against
`SE(SR | n_years) / sqrt(n_eff(baskets))` measured from the PnL correlation
matrix — the QNT-94 construction, applied to baskets instead of source series.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops

OUT = "/home/ubuntu/mtx/signal_zoo/qnt104"
TI, W, MIRROR_T = A.index, 60, 0.35
IS_END, O1S, O1E, O2S = "2016-12-31", "2017-01-01", "2021-12-31", "2022-01-01"
r_c2c = _RET["c2c"].reindex(TI).astype(float)

pnl, meta = {}, []
for ef in ("monthly", "all"):
    P = (pd.read_csv(f"{OUT}/call_features_{ef}.csv", index_col=0, parse_dates=True)
           .join(pd.read_csv(f"{OUT}/comb_features_{ef}.csv", index_col=0, parse_dates=True))
           .join(pd.read_csv(f"{OUT}/put_features_{ef}.csv", index_col=0, parse_dates=True))
           .reindex(TI))
    fam = lambda c: "call" if c.startswith("call_") else ("put" if c.startswith("put_") else "comb")
    keep = [c for c in P.columns
            if abs(float(P[c].corr(r_c2c))) <= MIRROR_T and P[c].notna().sum() > 1500]
    legs = {c: np.tanh(pd.Series(ops.robust_z(P[c].astype(float), W), index=TI) / 2.0) for c in keep}
    for v in ("c2c", "o2o", "day"):
        sg = {c: (wstats(s, v, end=IS_END) or {}).get("sign", 0) for c, s in legs.items()}
        for name in ("callEW", "combEW", "putEW", "allEW"):
            cols = [c for c in keep if sg.get(c) and (name == "allEW" or fam(c) == name[:-2])]
            if len(cols) < 3: continue
            b = pd.concat([legs[c] * sg[c] for c in cols], axis=1).mean(axis=1).clip(-1, 1)
            pos = b.shift(_SHIFT[v])
            g = pos * _RET[v] - pos.fillna(0).diff().abs() * _COST[v]
            k = f"{ef}|{name}|{v}"
            pnl[k] = g
            o = wstats(b, v, start=O1S, sign=1)
            meta.append(dict(key=k, panel=ef, basket=name, variant=v, n_members=len(cols)))
G = pd.DataFrame(pnl)
meta = pd.DataFrame(meta).set_index("key")

OOS = G.loc[O1S:]
sr = lambda x: np.sqrt(252) * x.mean() / x.std()
meta["SR_OOS_all"] = OOS.apply(lambda c: sr(c.dropna()))
meta["SR_OOS1"] = G.loc[O1S:O1E].apply(lambda c: sr(c.dropna()))
meta["SR_OOS2"] = G.loc[O2S:].apply(lambda c: sr(c.dropna()))
C = OOS.corr()
ev = np.clip(np.linalg.eigvalsh(C.values), 0, None)
neff = float(ev.sum() ** 2 / (ev ** 2).sum())
nyr = len(OOS.dropna(how="all")) / 252
se = np.sqrt((1 + 0.5) / (len(OOS.dropna(how="all"))) * 252) / np.sqrt(252)   # ~1/sqrt(years)
se = 1 / np.sqrt(nyr)
med = float(meta.SR_OOS_all.median())
iu = np.triu_indices_from(C.values, 1)
print("=== QNT-104 step 8: the aggregate OOS read, with its own error bar ===")
print(meta.round(3).to_string())
print(f"\nheld-out window {O1S} -> {OOS.dropna(how='all').index.max().date()} "
      f"({nyr:.1f} years), sign frozen {IS_END}")
print(f"24 baskets, mean pairwise OOS PnL corr {np.nanmean(C.values[iu]):+.3f}, "
      f"eigenvalue n_eff = {neff:.2f}")
print(f"SE(SR | {nyr:.1f}y) = {se:.3f}; SE of the median of {neff:.2f} effective baskets "
      f"= {se/np.sqrt(neff):.3f}")
print(f"observed median SR_OOS(net) = {med:+.3f}  ->  t = {med/(se/np.sqrt(neff)):.2f}, "
      f"two-sided p ~ {2*(1-__import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(med/(se/np.sqrt(neff))))):.3f}")
print(f"  (the circular-shift null gave P = 0.000 for the same statistic; it decorrelates "
      f"members, so THIS is the number to quote)")
print(f"\nper-block: median SR_OOS1 {meta.SR_OOS1.median():+.3f}, "
      f"median SR_OOS2 {meta.SR_OOS2.median():+.3f}, "
      f"{int((meta.SR_OOS_all > 0).sum())}/{len(meta)} baskets positive over the whole held-out window")
G.to_pickle(f"{OUT}/basket_pnl.pkl"); meta.to_csv(f"{OUT}/basket_oos.csv")

# ── step 8b: is the aggregate OOS positive just a scaled-down long-index bet? ──
# QNT-99 A3: MTX carries an unconditional long drift and `|beta| < 0.15` is
# scale-evadable, so a net-long book at 10% exposure looks neutral and earns the
# drift. The test that cannot be evaded: hedge each basket's OOS PnL against the
# SAME-variant buy-and-hold and re-score the RESIDUAL.
print("\n=== step 8b: beta-hedged OOS alpha (drift removed) ===")
rows = []
for k in G.columns:
    ef, name, v = k.split("|")
    y = G[k].loc[O1S:].dropna()
    x = _RET[v].reindex(y.index).astype(float)
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    b = float(np.cov(j.y, j.x, ddof=0)[0, 1] / j.x.var(ddof=0))
    res = j.y - b * j.x
    rows.append(dict(key=k, basket=name, variant=v, SR_OOS=sr(j.y), beta_OOS=b,
                     SR_alpha=sr(res),
                     SR_alpha_OOS1=sr((j.y - b * j.x).loc[:O1E]),
                     SR_alpha_OOS2=sr((j.y - b * j.x).loc[O2S:])))
H = pd.DataFrame(rows).set_index("key")
print(H.round(3).to_string())
Ares = pd.DataFrame({k: (G[k].loc[O1S:] - H.loc[k, "beta_OOS"] * _RET[k.split("|")[2]]
                         .reindex(G.index).loc[O1S:]) for k in G.columns})
C2 = Ares.corr(); ev2 = np.clip(np.linalg.eigvalsh(C2.values), 0, None)
ne2 = float(ev2.sum() ** 2 / (ev2 ** 2).sum())
med_a = float(H.SR_alpha.median())
sea = se / np.sqrt(ne2)
from scipy.stats import norm
print(f"\nhedged: n_eff {ne2:.2f}, median SR_alpha {med_a:+.3f}, SE {sea:.3f} -> "
      f"t = {med_a/sea:.2f}, p ~ {2*(1-norm.cdf(abs(med_a/sea))):.3f}  "
      f"[median-vs-mean penalty 1.25x -> t = {med_a/(1.25*sea):.2f}, p ~ "
      f"{2*(1-norm.cdf(abs(med_a/(1.25*sea)))):.3f}]")
print(f"unhedged for comparison: median {med:+.3f}, t {med/(se/np.sqrt(neff)):.2f} "
      f"[1.25x-penalised t {med/(1.25*se/np.sqrt(neff)):.2f}]")
print(f"{int((H.SR_alpha > 0).sum())}/{len(H)} baskets have positive hedged OOS alpha")
print(f"buy-and-hold over the held-out window: " + "  ".join(
    f"{v} SR {sr(_RET[v].loc[O1S:].dropna()):+.3f}" for v in ("c2c", "o2o", "day")))
H.to_csv(f"{OUT}/basket_alpha.csv")
