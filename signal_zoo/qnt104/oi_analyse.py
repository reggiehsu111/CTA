"""QNT-104 step 4 — read the grid: headline, realised n_eff, OOS decay, survivors.

Everything a POSITIVE claim needs under QNT-78 Rule 1 is computed here and
printed whether or not the answer is positive: per-feature n (never the cell
count), ICC / n_eff, sd(SR) across cells vs SE(SR | n_years). If the ratio is
below 1 no best cell may be quoted at all.

n_eff is measured, not assumed: each feature's cells are collapsed to one PnL,
and the eigenvalue n_eff of the per-feature correlation matrix is reported
alongside QNT-94's 0.43*S rule of thumb.
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd, cta
from scipy import stats

OUT = "/home/ubuntu/mtx/signal_zoo/qnt104"
sw  = pd.read_csv(f"{OUT}/oi_sweep_full.csv")
pnl = pd.read_pickle(f"{OUT}/oi_pnl.pkl")


def eig_neff(corr):
    ev = np.linalg.eigvalsh(np.nan_to_num(corr, nan=0.0))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


for reg in ("full", "night"):
    d = sw[sw.regime == reg].copy()
    print(f"\n{'='*78}\n=== REGIME {reg}: {len(d)} cells, {d.feature.nunique()} features "
          f"({d.groupby('family').feature.nunique().to_dict()}) ===")
    h = d.rename(columns={"full_SR_net": "SR_net", "full_n_years": "n_years"})
    cta.sweep_headline(h, value="SR_net", series_col="feature",
                       label=f"QNT-104 call+comb OI [{reg}]").print()

    # per-family aggregate test — the primary test under Rule 1
    for fam in ("call", "comb"):
        hf = h[h.family == fam]
        if not len(hf): continue
        cta.sweep_headline(hf, value="SR_net", series_col="feature",
                           label=f"QNT-104 [{reg}] family={fam}").print()

    # realised n_eff of the per-feature statistic (QNT-94 method)
    cells = d.set_index("cell")
    P = pnl[[c for c in cells.index if c in pnl.columns]].astype(float)
    fmap = cells.loc[P.columns, "feature"]
    perf = pd.DataFrame({f: P.loc[:, (fmap == f).values].mean(axis=1)
                         for f in fmap.unique()}).dropna(how="all")
    C = perf.corr()
    ne = eig_neff(C.values)
    iu = np.triu_indices_from(C.values, 1)
    print(f"per-feature PnL redundancy: S = {C.shape[0]}, eigenvalue n_eff = {ne:.2f} "
          f"({ne/C.shape[0]:.2f}*S; QNT-94 rule of thumb 0.43*S = {0.43*C.shape[0]:.2f}), "
          f"mean pairwise corr {np.nanmean(C.values[iu]):+.3f}")
    sd = float(h.SR_net.std())
    print(f"  -> d_min at this n_eff = 2.80*sqrt(2)*sd/sqrt(n_eff) = "
          f"{2.80*np.sqrt(2)*0.13/np.sqrt(ne):.3f} SR (sd=0.13 house value)")

    # OOS decay
    for a, b in (("SR_IS", "SR_OOS1"), ("SR_IS", "SR_OOS2"), ("SR_OOS1", "SR_OOS2")):
        j = d.dropna(subset=[a, b])
        if len(j) < 30: continue
        print(f"  {a} -> {b}: med {j[a].median():+.3f} -> {j[b].median():+.3f}  "
              f"frac>0 {(j[b] > 0).mean():.3f}  corr {j[a].corr(j[b]):+.3f}  n={len(j)}")

    # gates
    print(f"  four-gate passers (beta_per_w rule): {int(d.passes.sum())} / {len(d)} "
          f"= {d.passes.mean():.3%}   [raw-beta rule: "
          f"{int((d.gate_srsr & d.gate_posyr & d.gate_nyr & d.gate_beta_raw).sum())}]")
    g = d[d.passes]
    if len(g):
        print(g.sort_values("full_SR_net", ascending=False).head(20)[
            ["cell", "sign", "mirror", "SR_IS", "SR_OOS1", "SR_OOS2", "full_SR_net",
             "full_SR_of_SR", "full_positive_years", "full_beta", "beta_per_w",
             "full_abs_exec_w", "full_held_pct", "full_n_years"]].round(3).to_string(index=False))
        g.to_csv(f"{OUT}/survivors_{reg}.csv", index=False)

    print("\n  per-feature median SR_net / IS / OOS1 / OOS2:")
    pf = d.groupby(["family", "feature"])[["full_SR_net", "SR_IS", "SR_OOS1", "SR_OOS2"]].median()
    print(pf.round(3).to_string())

    # Wilcoxon on the per-feature OOS medians — the honest aggregate test
    for col in ("SR_OOS1", "SR_OOS2"):
        v = d.groupby("feature")[col].median().dropna()
        if len(v) >= 6:
            print(f"  per-feature {col}: median {v.median():+.3f}, {int((v>0).sum())}/{len(v)} "
                  f"positive, Wilcoxon p = {stats.wilcoxon(v)[1]:.3f}")

# ── cross-regime: does any feature survive BOTH held-out blocks? ───────────
d = sw[sw.regime == "full"].dropna(subset=["SR_OOS1", "SR_OOS2"])
both = d[(d.SR_IS > 0.3) & (d.SR_OOS1 > 0.3) & (d.SR_OOS2 > 0.3)]
print(f"\n{'='*78}\ncells with SR_net > 0.3 in IS *and* OOS1 *and* OOS2: {len(both)} / {len(d)} "
      f"({len(both)/max(len(d),1):.2%}); expected under independence if each block is a "
      f"coin flip at the observed marginal rates = "
      f"{(d.SR_IS>0.3).mean()*(d.SR_OOS1>0.3).mean()*(d.SR_OOS2>0.3).mean():.2%}")
if len(both):
    print(both.sort_values("full_SR_net", ascending=False).head(20)[
        ["cell", "sign", "mirror", "SR_IS", "SR_OOS1", "SR_OOS2", "full_SR_of_SR",
         "full_positive_years", "beta_per_w", "full_abs_exec_w", "n_gates"]]
        .round(3).to_string(index=False))
