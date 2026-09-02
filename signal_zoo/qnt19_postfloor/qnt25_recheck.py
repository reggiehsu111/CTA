"""QNT-19: does the +1 PIT floor change QNT-25's effective-n conclusions?

QNT-25 re-reported the macro sweeps at per-SERIES n and showed the cross-cell
SR dispersion sits below the Sharpe standard error. Grids 1 (QNT-12 522-cell)
and 2 (QNT-14 198-cell window grid) are built from `load_macro_tw`, so the floor
moves their inputs. Grids 3-6 read slow-input / combo CSVs the floor never
touches (verified: 0 of 1512 slow cells moved).

Reads only. Reuses QNT-25's own estimators verbatim so the two sides are
comparable, and prints pre-floor vs post-floor side by side.
"""
import numpy as np, pandas as pd
from scipy import stats

Z = "/home/ubuntu/mtx/signal_zoo"
POST = f"{Z}/qnt19_postfloor"
pd.set_option("display.width", 200)


def se_sr(sr, yrs):
    return np.sqrt((1.0 + 0.5 * np.asarray(sr, float) ** 2) / yrs)


def icc(df, value, group):
    g = df.groupby(group)[value]
    k = g.size().mean()
    grand = df[value].mean()
    msb = (g.size() * (g.mean() - grand) ** 2).sum() / (g.ngroups - 1)
    msw = ((df[value] - g.transform("mean")) ** 2).sum() / (len(df) - g.ngroups)
    r = float(np.clip((msb - msw) / (msb + (k - 1) * msw), 0.0, 1.0))
    return r, k, len(df) / (1 + (k - 1) * r)


def paired(x, label):
    x = pd.Series(x).dropna()
    n, pos = len(x), int((x > 0).sum())
    binom = stats.binomtest(pos, n, 0.5).pvalue
    try:
        w = stats.wilcoxon(x).pvalue
    except ValueError:
        w = np.nan
    print(f"  {label:40s} n={n:3d}  median {x.median():+.4f}  pos {pos}/{n} ({pos/n:.0%})"
          f"  binom p={binom:.3f}  wilcoxon p={w:.3f}")
    return x.median(), pos / n, binom


def noise(sr_cells, yrs, label):
    sd = float(np.std(sr_cells, ddof=1))
    se = float(np.mean(se_sr(sr_cells, yrs)))
    print(f"  {label:40s} sd(SR) {sd:.3f}  SE(SR) {se:.3f}  ratio {sd/se:.2f}"
          f"   -> {'BELOW noise floor' if sd/se < 1 else '~ noise floor' if sd/se < 1.5 else 'above'}")
    return sd / se


print("=" * 100)
print("GRID 1 — 522-cell standalone macro sweep")
print("=" * 100)
for tag, path in (("pre ", f"{Z}/macro_sweep/full_sweep.csv"),
                  ("post", f"{POST}/full_sweep.csv")):
    g1 = pd.read_csv(path)
    r, k, neff = icc(g1, "SR_full", "series")
    r2, _, neff2 = icc(g1, "SR_of_SR", "series")
    print(f"[{tag}] cells {len(g1)}  series {g1.series.nunique()}"
          f"  ICC(SR_full) {r:.3f} n_eff {neff:.1f}  ICC(SR_of_SR) {r2:.3f} n_eff {neff2:.1f}")
    noise(g1.SR_full, g1.n_years, f"[{tag}] SR_full, all cells")
    noise(g1.SR_of_SR, g1.n_years, f"[{tag}] SR_of_SR, all cells")
    ps = g1.groupby("series").agg(SR_med=("SR_full", "median"), SRSR_med=("SR_of_SR", "median"))
    paired(ps.SR_med, f"[{tag}] per-series median SR_full")
    psx = g1.groupby("series").agg(SR_IS=("SR_IS", "median"), SR_OOS=("SR_OOS", "median"))
    paired(psx.SR_IS, f"[{tag}] per-series median SR_IS")
    paired(psx.SR_OOS, f"[{tag}] per-series median SR_OOS")
    mixed = int(g1.groupby("series").sign_IS.nunique().gt(1).sum())
    print(f"[{tag}] best cell SR_of_SR {g1.SR_of_SR.max():.3f} ({g1.loc[g1.SR_of_SR.idxmax(),'cand']})"
          f"   series disagreeing with themselves on sign: {mixed}/{g1.series.nunique()}\n")

print("=" * 100)
print("GRID 2 — 198-cell daily-macro window grid (regime=full)")
print("=" * 100)
rows = []
for tag, path in (("pre ", f"{Z}/macro_windows/window_sweep_full.csv"),
                  ("post", f"{POST}/window_sweep_full.csv")):
    g2 = pd.read_csv(path)
    g2 = g2[g2.regime == "full"]
    piv = g2.pivot_table(index=["series", "transform", "window"],
                         columns="variant", values="SR_net").reset_index()
    piv["d_day"] = piv["day"] - piv["c2c"]
    piv["d_o2o"] = piv["o2o"] - piv["c2c"]
    r, k, neff = icc(piv, "d_day", "series")
    print(f"[{tag}] {len(piv)} cells, {piv.series.nunique()} series"
          f"  ICC(dSR day-c2c) {r:.3f}  n_eff {neff:.1f}")
    m_cell, f_cell, p_cell = paired(piv.d_day, f"[{tag}] dSR day-c2c, per CELL")
    m_ser, f_ser, p_ser = paired(piv.groupby("series").d_day.median(),
                                 f"[{tag}] dSR day-c2c, per SERIES")
    paired(piv.groupby("series").d_o2o.median(), f"[{tag}] dSR o2o-c2c, per SERIES")
    noise(piv.c2c, g2.n_years.median(), f"[{tag}] SR_c2c across cells")
    noise(piv.day, g2.n_years.median(), f"[{tag}] SR_day across cells")
    rho = piv.c2c.corr(piv.day)
    se_d = float(se_sr(0.1, g2.n_years.median())) * np.sqrt(2 * (1 - rho))
    psd = piv.groupby("series").d_day.median()
    print(f"[{tag}] corr(c2c,day) {rho:.3f}  SE(paired dSR) {se_d:.3f}"
          f"  per-series median = {psd.median()/se_d:+.2f} SE\n")
    rows.append(dict(floor=tag.strip(), n_eff=round(neff, 1), cell_median=round(m_cell, 4),
                     cell_winrate=round(f_cell, 3), cell_binom_p=round(p_cell, 4),
                     series_median=round(m_ser, 4), series_winrate=round(f_ser, 3),
                     series_binom_p=round(p_ser, 4), sigma_units=round(psd.median()/se_d, 2)))

pd.DataFrame(rows).to_csv(f"{POST}/qnt25_recheck_grid2.csv", index=False)
print(f"wrote {POST}/qnt25_recheck_grid2.csv")
