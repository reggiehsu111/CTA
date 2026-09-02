"""QNT-106 step 4 — the QNT-78 reporting line, redundancy, OOS, baskets."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd, cta
from scipy import stats
OUT = "/home/ubuntu/mtx/signal_zoo/qnt106"
res = pd.read_csv(f"{OUT}/sweep_full.csv")
pnl = pd.read_pickle(f"{OUT}/sweep_pnl.pkl")

print("=== QNT-78 rule-1 reporting line ===")
for lab, d in [("all", res), ("full-regime", res[res.regime == "full"]),
               ("night-regime", res[res.regime == "night"]),
               ("non-mirror only", res[~res.is_mirror])]:
    print(cta.sweep_headline(d, value="SR_net", series_col="series",
                             label=f"QNT-106 internals [{lab}]"))

print("\n=== per-source-series aggregate test (full regime, sign frozen on IS) ===")
full = res[res.regime == "full"]
ps = full.groupby(["family", "series"]).agg(
    n_cells=("SR_net", "size"), med_SR=("SR_net", "median"),
    med_IS=("SR_IS", "median"), med_OOS1=("SR_OOS1", "median"),
    med_OOS2=("SR_OOS2", "median"), med_beta=("beta", "median"),
    med_beta_per_w=("beta_per_w", "median"), med_SRofSR=("SR_of_SR", "median")).reset_index()
ps.to_csv(f"{OUT}/per_series.csv", index=False)
print(ps.round(3).to_string(index=False))
w = stats.wilcoxon(ps.med_SR)
print(f"\nper-series median SR_net {ps.med_SR.median():+.3f}, "
      f"{int((ps.med_SR > 0).sum())}/{len(ps)} positive, Wilcoxon p = {w.pvalue:.3f}")
for c in ("med_IS", "med_OOS1", "med_OOS2"):
    print(f"  {c}: median {ps[c].median():+.3f}  p = {stats.wilcoxon(ps[c].dropna()).pvalue:.3f}")
ok = ps[["med_IS", "med_OOS2"]].dropna()
print(f"  corr(SR_IS, SR_OOS2) across series = {ok.med_IS.corr(ok.med_OOS2):+.3f}")

# ── redundancy: n_eff of the per-FEATURE PnLs (measured, not assumed) ─────
cells = [c for c in pnl.columns if c.startswith("full|")]
fp = {}
for f in full.series.unique():
    cs = [c for c in cells if c.split("|")[1] == f]
    fp[f] = pnl[cs].mean(axis=1)
FP = pd.DataFrame(fp).dropna(how="all")
C = FP.corr().dropna(how="all").dropna(axis=1, how="all")
lam = np.linalg.eigvalsh(C.values)
neff = lam.sum() ** 2 / (lam ** 2).sum()
print(f"\nn_eff of the {C.shape[0]} per-feature PnLs = {neff:.2f} "
      f"({neff/C.shape[0]:.2f}*S; QNT-94 rule of thumb 0.43*S), "
      f"mean |corr| {np.abs(C.values[np.triu_indices_from(C, 1)]).mean():.3f}")
print("realised d_min = 4.26*0.13/sqrt(n_eff) = "
      f"{4.26*0.13/np.sqrt(neff):.3f} SR  (pre-run claim on raw S=29 was 0.103)")

# ── selection-free equal-weight baskets, held out, BETA-HEDGED (QNT-104) ──
print("\n=== selection-free family baskets, sign frozen on IS, beta-hedged ===")
import io
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read()
             .split("# ── Build the signals once")[0], SWEEP, "exec"))
OOS = "2018-01-01"
rows = []
for (fam, v), d in full.groupby(["family", "variant"]):
    cs = [c for c in d.cell if c in pnl.columns]
    if len(cs) < 20: continue
    b = pnl[cs].mean(axis=1).loc[OOS:].dropna()
    if len(b) < 500: continue
    bh = _RET[v].reindex(b.index).astype(float)
    j = pd.concat([b.rename("y"), bh.rename("x")], axis=1).dropna()
    beta = np.cov(j.y, j.x, ddof=0)[0, 1] / j.x.var(ddof=0)
    alpha = j.y - beta * j.x
    sr = lambda s: float(np.sqrt(252) * s.mean() / s.std())
    rows.append(dict(family=fam, variant=v, n_cells=len(cs), n_days=len(j),
                     SR_OOS=sr(j.y), beta=beta, SR_alpha=sr(alpha),
                     t_alpha=float(alpha.mean() / alpha.std() * np.sqrt(len(alpha)))))
bk = pd.DataFrame(rows); bk.to_csv("/home/ubuntu/mtx/signal_zoo/qnt106/baskets.csv", index=False)
print(bk.round(3).to_string(index=False))
print(f"\nmedian basket SR_OOS {bk.SR_OOS.median():+.3f}  -> "
      f"beta-hedged alpha SR {bk.SR_alpha.median():+.3f}, median t {bk.t_alpha.median():+.2f}")
for v in full.variant.unique():
    bh = _RET[v].loc[OOS:].dropna()
    print(f"  buy-and-hold {v} SR over the same window: "
          f"{float(np.sqrt(252)*bh.mean()/bh.std()):+.2f}")
