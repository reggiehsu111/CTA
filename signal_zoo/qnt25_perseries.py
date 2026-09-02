"""QNT-25: re-report the macro sweeps at per-SERIES n, and make the noise floor explicit.

No sweep is re-run. Everything below reads the CSVs already on disk from
QNT-12 (macro_sweep/) and QNT-14/16/18 (macro_windows/, macro_combo/).

Three questions per grid:
  1. how many effective tests are there?  (cells / series, and an ICC-based n_eff)
  2. what does the headline become when collapsed to one number per source series?
  3. is the cross-cell SR dispersion bigger than the standard error of a Sharpe?
"""
import numpy as np, pandas as pd
from scipy import stats

Z = "/home/ubuntu/mtx/signal_zoo"
pd.set_option("display.width", 200)

def se_sr(sr, yrs):
    """SE of an annualised Sharpe estimated over `yrs` years (Lo 2002, iid)."""
    return np.sqrt((1.0 + 0.5 * np.asarray(sr, float) ** 2) / yrs)

def icc(df, value, group):
    """One-way random-effects ICC + design-effect n_eff for a balanced-ish grid."""
    g = df.groupby(group)[value]
    k = g.size().mean()
    grand = df[value].mean()
    msb = (g.size() * (g.mean() - grand) ** 2).sum() / (g.ngroups - 1)
    msw = ((df[value] - g.transform("mean")) ** 2).sum() / (len(df) - g.ngroups)
    r = (msb - msw) / (msb + (k - 1) * msw)
    r = float(np.clip(r, 0.0, 1.0))
    deff = 1 + (k - 1) * r
    return r, k, len(df) / deff

def paired_report(x, label, alt="two-sided"):
    x = pd.Series(x).dropna()
    n, pos = len(x), int((x > 0).sum())
    binom = stats.binomtest(pos, n, 0.5, alternative=alt).pvalue
    try:
        w = stats.wilcoxon(x, alternative=alt).pvalue
    except ValueError:
        w = np.nan
    t = stats.ttest_1samp(x, 0.0).pvalue
    print(f"  {label:34s} n={n:3d}  median {x.median():+.4f}  mean {x.mean():+.4f}"
          f"  range [{x.min():+.3f},{x.max():+.3f}]  pos {pos}/{n} ({pos/n:.0%})"
          f"  binom p={binom:.3f}  wilcoxon p={w:.3f}  t p={t:.3f}")
    return dict(n=n, median=x.median(), mean=x.mean(), pos=pos, frac=pos/n,
                binom=binom, wilcoxon=w, ttest=t, lo=x.min(), hi=x.max())

def noise_floor(sr_cells, yrs, label):
    sd = float(np.std(sr_cells, ddof=1))
    se = float(np.mean(se_sr(sr_cells, yrs)))
    r = sd / se
    verdict = ("dispersion BELOW the noise floor - ranking cells ranks noise" if r < 1.0 else
               "dispersion ~ the noise floor - ranking is mostly noise" if r < 1.5 else
               "dispersion exceeds the noise floor")
    print(f"  {label:34s} sd(SR) across cells = {sd:.3f}   SE(SR|{np.mean(yrs):.0f}y) = {se:.3f}"
          f"   ratio {r:.2f}   -> {verdict}")
    return sd, se

out = {}
BAR = "=" * 100

# ══════════════════════════════════════════════════════════════════════════
print(BAR); print("GRID 1 — QNT-12 standalone macro sweep  (macro_sweep/full_sweep.csv)"); print(BAR)
g1 = pd.read_csv(f"{Z}/macro_sweep/full_sweep.csv")
print(f"  {len(g1)} cells = {g1.series.nunique()} series x {g1["transform"].nunique()} transforms"
      f" x {g1["window"].nunique()} windows   ({len(g1)/g1.series.nunique():.0f} cells per series)")
r, k, neff = icc(g1, "SR_full", "series")
print(f"  ICC(series) on SR_full = {r:.3f}  ->  design effect {1+(k-1)*r:.1f},  n_eff = {neff:.1f} independent tests"
      f"  (nominal {len(g1)})")
r2, _, neff2 = icc(g1, "SR_of_SR", "series")
print(f"  ICC(series) on SR_of_SR = {r2:.3f}  ->  n_eff = {neff2:.1f}")
noise_floor(g1.SR_full, g1.n_years, "SR_full, all cells")
noise_floor(g1.SR_of_SR, g1.n_years, "SR_of_SR, all cells")
print("\n  collapsed to one number per source series (n=29):")
ps = g1.groupby("series").agg(SR_med=("SR_full", "median"), SRSR_med=("SR_of_SR", "median"),
                              SR_max=("SR_full", "max"), SRSR_max=("SR_of_SR", "max"),
                              n_years=("n_years", "median"))
out["g1_sr"] = paired_report(ps.SR_med, "per-series median SR_full")
out["g1_srsr"] = paired_report(ps.SRSR_med - 0.6, "per-series median SR_of_SR - 0.6", alt="greater")
print(f"\n  best CELL SR_of_SR = {g1.SR_of_SR.max():.3f} ({g1.loc[g1.SR_of_SR.idxmax(),'cand']}), gate 0.6")
srsr_se = float(np.sqrt((1 + 0.5 * 0.577 ** 2) / g1.n_years.median()))
print(f"  SE of an SR_of_SR estimated on {g1.n_years.median():.0f} annual SRs ~ {srsr_se:.3f}"
      f"  ->  best cell is {(0.6-0.577)/srsr_se:.2f} SE below the gate, i.e. indistinguishable from it")
print(f"  max of {len(g1)} draws from N(0,{g1.SR_of_SR.std():.3f}) around the mean {g1.SR_of_SR.mean():.3f}:"
      f" expected max ~ {g1.SR_of_SR.mean() + g1.SR_of_SR.std()*stats.norm.ppf(1-1/(neff2+1)):.3f}"
      f" at n_eff={neff2:.0f};  ~{g1.SR_of_SR.mean() + g1.SR_of_SR.std()*stats.norm.ppf(1-1/(len(g1)+1)):.3f} at n={len(g1)}")
print("\n  the per-series SR_full positives are SIGN-SELECTION BIAS, not edge")
print("  (sign is fitted on IS<=2018, so IS and hence full-sample SR are mechanically positive):")
psx = g1.groupby("series").agg(SR_IS=("SR_IS", "median"), SR_full=("SR_full", "median"),
                               SR_OOS=("SR_OOS", "median"))
for c in ["SR_IS", "SR_full", "SR_OOS"]:
    out[f"g1_{c}"] = paired_report(psx[c], f"per-series median {c}")
print(f"  -> the whole apparent macro tilt lives in the fitted half. OOS is a coin flip.")
mixed = g1.groupby("series").sign_IS.nunique().gt(1).sum()
print(f"  {mixed}/29 series do not even agree with themselves on SIGN across their own 18 cells.")
ps.round(3).sort_values("SR_med", ascending=False).to_csv(f"{Z}/qnt25_g1_per_series.csv")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + BAR); print("GRID 2 — QNT-14 daily-macro window sweep  (macro_windows/window_sweep_full.csv)"); print(BAR)
g2 = pd.read_csv(f"{Z}/macro_windows/window_sweep_full.csv")
g2 = g2[g2.regime == "full"]
piv = g2.pivot_table(index=["series", "transform", "window"], columns="variant", values="SR_net").reset_index()
piv["d_day"] = piv["day"] - piv["c2c"]
piv["d_o2o"] = piv["o2o"] - piv["c2c"]
print(f"  {len(piv)} cells = {piv.series.nunique()} series x 18 transform-windows")
r, k, neff = icc(piv, "d_day", "series")
print(f"  ICC(series) on dSR(day-c2c) = {r:.3f}  ->  n_eff = {neff:.1f} independent tests (nominal 198)")
print("\n  AS REPORTED IN QNT-14 (cell level, treats 198 transforms as 198 tests):")
out["g2_cell"] = paired_report(piv.d_day, "dSR day@1 - c2c@2, per CELL")
print("\n  RE-REPORTED at per-series n:")
psd = piv.groupby("series").d_day.median()
out["g2_series"] = paired_report(psd, "dSR day@1 - c2c@2, per SERIES")
out["g2_o2o"] = paired_report(piv.groupby("series").d_o2o.median(), "dSR o2o@2 - c2c@2, per SERIES")
print()
noise_floor(piv.c2c, g2.n_years.median(), "SR_c2c across 198 cells")
noise_floor(piv.day, g2.n_years.median(), "SR_day across 198 cells")
rho_all = piv.c2c.corr(piv.day)
rho_in = piv.groupby("series").apply(lambda d: d.c2c.corr(d.day)).median()
se1 = float(se_sr(0.1, g2.n_years.median()))
se_d = se1 * np.sqrt(2 * (1 - rho_all))
print(f"  corr(SR_c2c, SR_day) across cells = {rho_all:.3f}  (within-series median {rho_in:.3f})")
print(f"  => SE of a PAIRED dSR ~ {se_d:.3f};  observed per-series median {psd.median():+.3f}"
      f" = {psd.median()/se_d:.2f} SE.  With n=11 series the SE of the median is ~{se_d/np.sqrt(11):.3f}.")
piv.groupby("series")[["c2c", "day", "o2o", "d_day"]].median().round(3).to_csv(f"{Z}/qnt25_g2_per_series.csv")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + BAR); print("GRID 3 — QNT-18 slow-macro window sweep  (macro_windows/slow_window_sweep.csv)"); print(BAR)
g3 = pd.read_csv(f"{Z}/macro_windows/slow_window_sweep.csv")
p3 = g3.pivot_table(index=["series", "transform", "window"], columns="variant", values="SR_net").reset_index()
p3["d_day"] = p3["day"] - p3["c2c"]
print(f"  {len(p3)} cells = {p3.series.nunique()} series x 18   (all variants at shift 2)")
r, k, neff = icc(p3, "d_day", "series")
print(f"  ICC(series) on dSR = {r:.3f}  ->  n_eff = {neff:.1f} (nominal {len(p3)})")
out["g3_cell"] = paired_report(p3.d_day, "dSR day@2 - c2c@2, per CELL")
out["g3_series"] = paired_report(p3.groupby("series").d_day.median(), "dSR day@2 - c2c@2, per SERIES")
noise_floor(p3.c2c, g3.n_years.median(), "SR_c2c across cells")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + BAR); print("GRID 4 — QNT-18 registered-signal window sweep  (registered_window_sweep.csv)"); print(BAR)
g4 = pd.read_csv(f"{Z}/macro_windows/registered_window_sweep.csv")
g4 = g4[g4.regime == "full"]
p4 = g4.pivot_table(index="signal", columns="variant", values="SR_net")
p4 = p4.dropna(subset=["c2c", "day"])
d4 = (p4["day"] - p4["c2c"])
print(f"  {len(p4)} signals — here a cell IS a series, so n was already honest")
out["g4"] = paired_report(d4, "dSR day - c2c, per SIGNAL")
noise_floor(p4.c2c.dropna(), g4.n_years.median(), "SR_c2c across signals")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + BAR); print("GRID 5 — QNT-18 de-tilt test  (macro_windows/detilt_matched.csv)"); print(BAR)
g5 = pd.read_csv(f"{Z}/macro_windows/detilt_matched.csv")
g5["series"] = g5.cand.str.split("|").str[0]
for grid in ["daily macro", "slow macro", "registered"]:
    s = g5[g5.grid == grid]
    p = s.pivot_table(index=["cand", "how"], columns="variant", values="SR").reset_index()
    p["series"] = p.cand.str.split("|").str[0]
    p["d"] = p["day"] - p["c2c"]
    for how in ["raw", "detilt"]:
        q = p[p.how == how]
        print(f"  [{grid} / {how}]  cells={len(q)} series={q.series.nunique()}")
        paired_report(q.d, "  per CELL")
        paired_report(q.groupby("series").d.median(), "  per SERIES")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + BAR); print("GRID 6 — QNT-16 macro combination  (macro_combo/)"); print(BAR)
cb = pd.read_csv(f"{Z}/macro_combo/combo_scoreboard.csv")
fam = pd.read_csv(f"{Z}/macro_combo/combo_family_sweep.csv")
ew3 = cb[cb.combo == "EW(igrea+epu_global+kr_kospi)"].iloc[0]
ny = float(ew3.n_years)
se_srsr = float(np.sqrt((1 + 0.5 * ew3.SR_of_SR ** 2) / ny))
print(f"  EW3 headline SR_of_SR = {ew3.SR_of_SR:.3f} on {ny:.0f} annual SRs, gate 0.60")
print(f"  SR_of_SR is itself a Sharpe of the yearly-SR sample: SE ~ sqrt((1+SRSR^2/2)/n_years) = {se_srsr:.3f}")
print(f"  -> the gate sits {(0.60-ew3.SR_of_SR)/se_srsr:.2f} SE away. 95% CI ~ [{ew3.SR_of_SR-1.96*se_srsr:.2f}, {ew3.SR_of_SR+1.96*se_srsr:.2f}]"
      f" — it contains both 0 and 0.6.")
print(f"  n of independent legs = 3 (igrea, epu_global, kr_kospi), pnl corr 0.16-0.40")
print(f"\n  the SAME EW3 basket across its 18 transform-window cells (combo_family_sweep.csv):")
print(f"    SR_of_SR  min {fam.SR_of_SR.min():.3f}  median {fam.SR_of_SR.median():.3f}  max {fam.SR_of_SR.max():.3f}"
      f"  sd {fam.SR_of_SR.std():.3f}")
print(f"    the headline 0.542 is not even the family's own median-cell value; the best cell"
      f" ({fam.loc[fam.SR_of_SR.idxmax(),'cell']}) is {fam.SR_of_SR.max():.3f}, still < 0.6.")
print(f"    sd across the 18 cells ({fam.SR_of_SR.std():.3f}) vs SE of one estimate ({se_srsr:.3f})"
      f" -> ratio {fam.SR_of_SR.std()/se_srsr:.2f}: picking the best cell of 18 is picking noise.")
print(f"    gate_srsr passes in {int(fam.gate_srsr.sum())}/18 cells.")

pd.DataFrame(out).T.to_csv(f"{Z}/qnt25_headline_tests.csv")
print(f"\nwrote {Z}/qnt25_headline_tests.csv, qnt25_g1_per_series.csv, qnt25_g2_per_series.csv")
