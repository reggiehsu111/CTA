"""QNT-25: how many independent tests are the 18 igrea transform x window cells?

Builds the same 18 cells as macro_sweep/igrea_robustness.py (one macro series,
6 transforms x 3 windows) and measures REDUNDANCY:
  * correlation of the 18 net PnL streams and of the 18 positions
  * eigenvalue n_eff = (sum L)^2 / sum L^2  -- effective independent bets
  * PC1 share, and SR of PC1 vs SR of the residual (shared vs distinct edge)
No sweep is re-run; this is one series.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context

OUT = "/home/ubuntu/mtx/signal_zoo"
IS_END = "2018-12-31"
REAL = dict(fixed_per_side=70.0, fee_rate=0.00004)
ctx = build_context(); A = ctx.asset
TF = {"selfz": ops.selfz, "robustz": ops.robust_z, "bdtanh": ops.bd_selftanh,
      "rankc": ops.rank_c, "signth": ops.sign_thresh, "dev": ops.dev}
WS = (60, 120, 252)
ret, close = A.returns, A["close"]

def sr(v):
    v = pd.Series(v).dropna()
    return float(np.sqrt(252) * v.mean() / v.std()) if v.std() > 0 else np.nan

def net_pnl(e):
    tc = e.fillna(0).diff().abs() * (REAL["fixed_per_side"] / (close * 50.0) + REAL["fee_rate"])
    return (e * ret - tc).dropna()

def family(name):
    x = ctx.macro(name)
    pos, pnl, srs = {}, {}, {}
    for t in TF:
        for w in WS:
            s = cta.normalize_signal(TF[t](x, w).replace([np.inf, -np.inf], np.nan),
                                     method="tanh", window=252)
            g = int(cta.signal_stats(s, A, end=IS_END, auto_flip=True,
                                     roll_adjusted=True, **REAL)["sign"])
            e = (s * g).shift(2).reindex(A.index)
            k = f"{t}|w{w}"
            pos[k], pnl[k] = e, net_pnl(e)
            srs[k] = sr(pnl[k])
    return pd.DataFrame(pos), pd.DataFrame(pnl).dropna(how="all"), pd.Series(srs)

def redundancy(label, P, L, S):
    C = L.corr()
    off = C.values[np.triu_indices_from(C, 1)]
    Cp = P.corr(); offp = Cp.values[np.triu_indices_from(Cp, 1)]
    ev = np.linalg.eigvalsh(np.nan_to_num(C.values, nan=0.0))[::-1]
    neff = ev.sum() ** 2 / (ev ** 2).sum()
    ew = L.mean(axis=1)                       # the shared component, equal-weighted
    resid = L.sub(ew, axis=0)                 # what each cell adds beyond the family
    se = float(np.sqrt((1 + 0.5 * S.median() ** 2) / (len(L) / 252)))
    print(f"\n--- {label}: {L.shape[1]} cells, {len(L)} days, {len(L)/252:.1f} years ---")
    print(f"  SR per cell: min {S.min():+.3f}  median {S.median():+.3f}  max {S.max():+.3f}"
          f"  sd {S.std():.3f}   (all positive: {bool((S>0).all())})")
    print(f"  SE of ONE cell's SR over {len(L)/252:.0f}y = {se:.3f}"
          f"  -> the {S.max()-S.min():.3f} spread across 18 cells is {(S.max()-S.min())/se:.2f} SE wide")
    print(f"  pairwise PnL corr:      mean {off.mean():.3f}  median {np.median(off):.3f}"
          f"  min {off.min():.3f}  max {off.max():.3f}")
    print(f"  pairwise POSITION corr: mean {offp.mean():.3f}  min {offp.min():.3f}  max {offp.max():.3f}")
    print(f"  PC1 explains {ev[0]/ev.sum():.1%} of PnL variance; PC1+PC2 {ev[:2].sum()/ev.sum():.1%}")
    print(f"  eigenvalue n_eff = {neff:.2f} independent bets out of 18"
          f"   (18 cells are worth ~{neff:.1f} tests)")
    print(f"  SR of the EW-of-18 (the SHARED edge)          {sr(ew):+.3f}")
    print(f"  SR of each cell's RESIDUAL vs the family EW:  median {sr(resid.median(axis=1)):+.3f},"
          f" per-cell median {np.median([sr(resid[c]) for c in resid]):+.3f}"
          f"  -> the DISTINCT part carries no edge")
    print(f"  SE of the family median SR given n_eff={neff:.2f}: {se/np.sqrt(neff):.3f}"
          f"  -> median {S.median():+.3f} is {S.median()/(se/np.sqrt(neff)):.2f} SE from zero"
          f"  (t p~{2*(1-__import__('scipy.stats',fromlist=['norm']).norm.cdf(abs(S.median()/(se/np.sqrt(neff))))):.3f})")
    return dict(label=label, n_cells=L.shape[1], sr_med=S.median(), sr_sd=S.std(),
                corr_mean=off.mean(), pos_corr_mean=offp.mean(), pc1=ev[0]/ev.sum(),
                n_eff=neff, sr_ew=sr(ew), se_one=se)

rows = []
for nm in ["igrea", "kr_kospi", "epu_global", "us_dxy_broad"]:
    P, L, S = family(nm)
    rows.append(redundancy(nm, P, L, S))
    if nm == "igrea":
        L.corr().round(3).to_csv(f"{OUT}/qnt25_igrea_pnl_corr.csv")
        S.round(3).to_csv(f"{OUT}/qnt25_igrea_cell_sr.csv")

r = pd.DataFrame(rows).round(3)
r.to_csv(f"{OUT}/qnt25_family_redundancy.csv", index=False)
print("\n" + "=" * 96)
print(r.to_string(index=False))
print(f"\n  mean n_eff across families = {r.n_eff.mean():.2f}"
      f"  ->  an 18-cell family is worth ~{r.n_eff.mean():.0f} tests, not 18.")
print(f"  522 cells / 29 series at n_eff {r.n_eff.mean():.2f} per family"
      f" = ~{29*r.n_eff.mean():.0f} effective tests (if series were independent of each other).")
print(f"\nwrote {OUT}/qnt25_family_redundancy.csv, qnt25_igrea_pnl_corr.csv, qnt25_igrea_cell_sr.csv")
