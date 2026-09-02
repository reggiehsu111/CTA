import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
OUT = "/home/ubuntu/mtx/signal_zoo/qnt106"
sw = pd.read_csv(f"{OUT}/sweep_full.csv"); d = sw[sw.regime == "full"]
ps = pd.read_csv(f"{OUT}/per_series.csv"); nl = pd.read_csv(f"{OUT}/null_control.csv")
nd = pd.read_csv(f"{OUT}/null_draws.csv"); bk = pd.read_csv(f"{OUT}/baskets.csv")
COL = {"breadth": "tab:blue", "xsec": "tab:purple", "liquidity": "tab:green",
       "flow": "tab:orange", "leverage": "tab:red"}

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
a = ax[0, 0]
for lbl, col, c in [("IS 2009-17", "SR_IS", "0.3"), ("OOS1 2018-21", "SR_OOS1", "tab:blue"),
                    ("OOS2 2022-26", "SR_OOS2", "tab:red")]:
    a.hist(d[col].dropna(), bins=60, histtype="step", lw=1.6, color=c, label=lbl, density=True)
a.axvline(0, color="k", lw=.8); a.legend(fontsize=8)
a.set_title("SR_net by block, sign frozen on IS\n(2,088 full-regime cells, 29 internals series)", fontsize=9)
a.set_xlabel("SR_net")

a = ax[0, 1]
a.scatter(d.SR_IS, d.SR_OOS2, s=3, alpha=.25, color="tab:red")
a.axhline(0, color="k", lw=.6); a.axvline(0, color="k", lw=.6)
a.set_xlabel("SR_IS"); a.set_ylabel("SR_OOS2 (2022-26, held out)")
a.set_title(f"IS carries nothing to the 2nd held-out block\ncell corr = {d.SR_IS.corr(d.SR_OOS2):+.3f}", fontsize=9)

a = ax[0, 2]
p = ps.sort_values("med_SR"); y = np.arange(len(p))
a.barh(y, p.med_SR, color=[COL[f] for f in p.family])
a.axvline(0, color="k", lw=.8)
se = 1 / np.sqrt(18)
a.axvspan(-se, se, color="0.85", zorder=0, label=f"±SE(SR|18y) = {se:.2f}")
a.set_yticks(y); a.set_yticklabels(p.series, fontsize=6)
a.set_xlabel("per-series median SR_net"); a.legend(fontsize=7, loc="lower right")
a.set_title("Every series sits inside its own noise band\nWilcoxon p = 0.949", fontsize=9)

a = ax[1, 0]
a.bar(np.arange(3) - .2, [ps.med_IS.median(), ps.med_OOS1.median(), ps.med_OOS2.median()],
      .4, color=["0.5", "tab:blue", "tab:red"])
a.set_xticks(range(3)); a.set_xticklabels(["IS\n2009-17", "OOS1\n2018-21", "OOS2\n2022-26"], fontsize=8)
a.axhline(0, color="k", lw=.8); a.set_ylabel("median per-series SR_net")
a.set_title("The two held-out blocks disagree in SIGN\n(p=0.000 negative, then p=0.005 positive)", fontsize=9)

a = ax[1, 1]
a.hist(nd["night"], bins=15, color="0.7", label="circular-shift null (40 reps)")
a.axvline(nl.loc[nl.regime == "night", "observed"].iloc[0], color="tab:red", lw=2,
          label="observed = 63")
a.set_xlabel("four-gate passers, night regime"); a.legend(fontsize=8)
a.set_title(f"Gate passers are at the no-information rate\nnull mean "
            f"{nl.loc[nl.regime=='night','null_mean'].iloc[0]:.0f}, p = "
            f"{nl.loc[nl.regime=='night','p'].iloc[0]:.2f}", fontsize=9)

a = ax[1, 2]
x = np.arange(len(bk))
a.bar(x - .2, bk.SR_OOS, .4, color="0.6", label="basket SR_OOS")
a.bar(x + .2, bk.SR_alpha, .4, color="tab:red", label="beta-hedged alpha")
a.axhline(0, color="k", lw=.8)
a.set_xticks(x); a.set_xticklabels(bk.family + "\n" + bk.variant, fontsize=6, rotation=90)
a.legend(fontsize=8); a.set_ylabel("SR (2018-2026, held out)")
a.set_title("Selection-free family baskets: negative before\nand after hedging (median alpha t = -1.19)", fontsize=9)

fig.suptitle("QNT-106 — 台股 market internals as MTX signals: 29 new source series, null on every axis",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .96])
fig.savefig(f"{OUT}/qnt106_summary.png", dpi=110)
print("saved", f"{OUT}/qnt106_summary.png")
