"""QNT-98 — breadth + power ledger for the macro->MTX program. No sweep is re-run."""
import sys; sys.path.insert(0, "/home/ubuntu/mtx")
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SZ = "/home/ubuntu/mtx/signal_zoo/"
OUT = SZ + "qnt98/"

G1 = pd.read_csv(SZ + "qnt19_postfloor/full_sweep.csv")           # standalone direction
G2 = pd.read_csv(SZ + "qnt19_postfloor/window_sweep_full.csv")    # daily macro x exec window
G2 = G2[G2.regime == "night"]
G3 = pd.read_csv(SZ + "qnt19_postfloor/slow_window_sweep.csv")    # slow macro x exec window
GRIDS = [("G1 standalone\n29 series", G1, "SR_full"),
         ("G2 night window\n11 daily series", G2, "SR_net"),
         ("G3 slow window\n21 series", G3, "SR_net")]

fig, ax = plt.subplots(2, 2, figsize=(15, 11))

# --- 1. sign frozen on IS: per-series IS vs OOS -----------------------------
a = ax[0, 0]
for (lab, d, _), c in zip(GRIDS, ["#1f77b4", "#d62728", "#2ca02c"]):
    ps = d.groupby("series")[["SR_IS", "SR_OOS"]].median()
    a.scatter(ps.SR_IS, ps.SR_OOS, s=42, alpha=.75, color=c,
              label=f"{lab.splitlines()[0]}  {int((ps.SR_OOS>0).sum())}/{len(ps)} OOS+")
lim = 1.0
a.plot([-lim, lim], [-lim, lim], "k--", lw=.8); a.axhline(0, color="k", lw=.8); a.axvline(0, color="k", lw=.8)
a.set_xlim(-.6, lim); a.set_ylim(-1.0, lim)
a.set_xlabel("per-series median SR, IS $\\leq$2018 (sign fitted here)")
a.set_ylabel("per-series median SR, OOS $>$2018")
a.set_title("1. Sign frozen on IS — the edge does not survive\n"
            "IS 61/61 series positive by construction; OOS 26/61", fontsize=11)
a.legend(fontsize=8, loc="lower right"); a.grid(alpha=.25)

# --- 2. gate funnel ---------------------------------------------------------
a = ax[0, 1]
names, vals = [], []
for lab, d, sr in GRIDS:
    g_sr = (d.SR_of_SR > 0.6); g_py = (d.positive_years >= 0.65)
    g_b = (d.beta.abs() < 0.15); g_n = (d.n_years >= 5)
    vals.append([100.0, 100*g_n.mean(), 100*g_b.mean(), 100*g_py.mean(), 100*g_sr.mean(),
                 100*(g_sr & g_py & g_b & g_n).mean()])
    names.append(lab)
labels = ["all cells", "n_years$\\geq$5", "|beta|<0.15", "$\\geq$65% pos yrs", "SR_of_SR>0.6", "all four"]
x = np.arange(len(labels)); w = .26
for i, (nm, v) in enumerate(zip(names, vals)):
    b = a.bar(x + (i-1)*w, v, w, label=nm.replace("\n", " "))
    a.bar_label(b, fmt="%.0f", fontsize=7, padding=1)
a.set_xticks(x); a.set_xticklabels(labels, fontsize=8, rotation=15)
a.set_ylabel("% of cells passing (marginally)")
a.set_title("2. House gates, marginal pass rate — 2,826 cells scored\n"
            "stability (SR_of_SR) is the binding gate, not beta or cost", fontsize=11)
a.legend(fontsize=8); a.grid(alpha=.25, axis="y")

# --- 3. transaction cost ladder --------------------------------------------
a = ax[1, 0]
ps = G1.groupby("series")[["SR_full_gross", "SR_full_stubcost", "SR_full"]].median().median()
combo = pd.read_csv(SZ + "macro_combo/combo_scoreboard.csv").set_index("combo")
ew3 = combo.loc["EW(igrea+epu_global+kr_kospi)"]
ig = combo.loc["igrea"]
steps = ["gross", "stub cost\n(20+2e-5)", "realistic\n(70+4e-5)", "3x realistic"]
a.plot(steps, [ig.SR_gross, ig.SR_stub, ig.SR_full, ig.SR_3xreal], "o-", label="igrea (best family)")
a.plot(steps, [ew3.SR_gross, ew3.SR_stub, ew3.SR_full, ew3.SR_3xreal], "s-", label="EW3 macro sleeve")
a.plot(steps[:3], [ps.SR_full_gross, ps.SR_full_stubcost, ps.SR_full], "^-",
       label="G1 per-series median (S=29)")
a.axhline(0.703, color="k", ls="--", lw=1, label="roll-adj buy & hold 0.703")
a.set_ylabel("Sharpe (roll-adjusted)")
a.set_title("3. Transaction costs are not the constraint\n"
            "monthly macro turns ~6/yr; gross$\\rightarrow$realistic costs 0.011 SR", fontsize=11)
a.legend(fontsize=8); a.grid(alpha=.25)

# --- 4. power ---------------------------------------------------------------
a = ax[1, 1]
sd = 0.13
S = np.arange(4, 80)
a.plot(S, 2.80*sd/np.sqrt(S), label="$d_{min}$, raw S (QNT-78)")
a.plot(S, 4.26*sd/np.sqrt(S), label="$d_{min}$, corrected for $n_{eff}$=0.43S (QNT-94)")
for s, c, lab in [(29, "#d62728", "S=29 today"), (43, "#2ca02c", "S=43 if all tidy series added")]:
    a.axvline(s, color=c, ls=":", lw=1.2)
    a.annotate(f"{lab}\n$d_{{min}}$={4.26*sd/np.sqrt(s):.3f}", (s, 4.26*sd/np.sqrt(s)),
               textcoords="offset points", xytext=(8, 18), fontsize=8, color=c)
obs = np.median([G1.groupby("series").SR_OOS.median().median(),
                 G2.groupby("series").SR_OOS.median().median(),
                 G3.groupby("series").SR_OOS.median().median()])
a.axhline(abs(obs), color="k", lw=1.2)
a.annotate(f"observed OOS effect |{obs:+.3f}|", (60, abs(obs)), textcoords="offset points",
           xytext=(-90, 6), fontsize=8)
a.set_xlabel("source series S"); a.set_ylabel("smallest resolvable effect $d$ (SR units)")
a.set_title("4. What this breadth can resolve\n"
            "at S=29 the floor is 0.103 SR; the OOS effect is ~0", fontsize=11)
a.legend(fontsize=8); a.grid(alpha=.25)

fig.suptitle("QNT-98 — macro $\\rightarrow$ MTX: no tradable signal, and the breadth behind that answer",
             fontsize=13, y=.995)
fig.tight_layout()
fig.savefig(OUT + "qnt98_breadth_ledger.png", dpi=115)
print("wrote", OUT + "qnt98_breadth_ledger.png")

# ledger table
rows = []
for lab, d, sr in GRIDS:
    ps = d.groupby("series")[["SR_IS", "SR_OOS"]].median()
    g = (d.SR_of_SR > 0.6) & (d.positive_years >= 0.65) & (d.beta.abs() < 0.15) & (d.n_years >= 5)
    rows.append(dict(grid=lab.replace("\n", " "), cells=len(d), S=d.series.nunique(),
                     four_gate_cells=int(g.sum()),
                     med_SR_IS=round(ps.SR_IS.median(), 3), med_SR_OOS=round(ps.SR_OOS.median(), 3),
                     OOS_pos=f"{int((ps.SR_OOS>0).sum())}/{len(ps)}",
                     binom_p=round(stats.binomtest(int((ps.SR_OOS > 0).sum()), len(ps)).pvalue, 3)))
led = pd.DataFrame(rows)
led.to_csv(OUT + "qnt98_ledger.csv", index=False)
print(led.to_string(index=False))
