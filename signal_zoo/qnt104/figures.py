import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
OUT = "/home/ubuntu/mtx/signal_zoo/qnt104"
sw = pd.read_csv(f"{OUT}/oi_sweep_full.csv"); d = sw[sw.regime == "full"]
nl = pd.read_csv(f"{OUT}/null_control.csv")
rv = pd.read_csv(f"{OUT}/null_real_vectorised.csv")
cb = pd.read_csv(f"{OUT}/oi_combos.csv")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
a = ax[0, 0]
for lbl, c in (("IS 2009-16", "0.3"), ("OOS1 2017-21", "tab:blue"), ("OOS2 2022-26", "tab:red")):
    col = {"IS 2009-16": "SR_IS", "OOS1 2017-21": "SR_OOS1", "OOS2 2022-26": "SR_OOS2"}[lbl]
    a.hist(d[col].dropna(), bins=60, histtype="step", lw=1.6, color=c, label=lbl, density=True)
a.axvline(0, color="k", lw=0.8); a.legend(fontsize=8)
a.set_title("SR_net by block, sign frozen on IS\n(3,744 cells, call + combined OI)", fontsize=9)
a.set_xlabel("SR_net")

a = ax[0, 1]
a.scatter(d.SR_IS, d.SR_OOS2, s=3, alpha=.25, color="tab:red")
a.axhline(0, color="k", lw=.6); a.axvline(0, color="k", lw=.6)
a.set_xlabel("SR_IS"); a.set_ylabel("SR_OOS2 (2022-26, held out)")
a.set_title(f"IS does not carry to the second held-out block\ncorr = {d.SR_IS.corr(d.SR_OOS2):+.3f}", fontsize=9)

a = ax[0, 2]
pf = d.groupby("feature")[["SR_IS", "SR_OOS1", "SR_OOS2"]].median().sort_values("SR_IS")
y = np.arange(len(pf))
a.barh(y - .25, pf.SR_IS, .25, label="IS", color="0.5")
a.barh(y, pf.SR_OOS1, .25, label="OOS1", color="tab:blue")
a.barh(y + .25, pf.SR_OOS2, .25, label="OOS2", color="tab:red")
a.set_yticks(y); a.set_yticklabels(pf.index, fontsize=6); a.axvline(0, color="k", lw=.6)
a.legend(fontsize=7); a.set_title("per-feature median SR_net (26 features)", fontsize=9)

a = ax[1, 0]
obs = int(rv.passes.sum())
a.hist(nl.passers, bins=range(int(nl.passers.max()) + 3), color="0.7", edgecolor="k", lw=.5)
a.axvline(obs, color="tab:red", lw=2, label=f"observed {obs}")
a.set_xlabel("four-gate passers per grid"); a.legend(fontsize=8)
a.set_title(f"circular-shift null ({len(nl)} reps)\nno-information grid, same features", fontsize=9)

a = ax[1, 1]
a.scatter(d.full_abs_exec_w, d.beta_per_w.abs(), s=3, alpha=.2, color="0.4")
g = d[d.passes]
a.scatter(g.full_abs_exec_w, g.beta_per_w.abs(), s=22, color="tab:red", label="4-gate passers")
a.axhline(0.15, color="tab:blue", ls="--", lw=1, label="beta_per_w gate")
a.axvline(0.31, color="tab:green", ls=":", lw=1, label="exposure measurability floor")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("mean |exec_w|")
a.set_ylabel("|beta_per_w|"); a.legend(fontsize=7)
a.set_title("QNT-100 gate: the front-share / max-pain\ncells are index bets at ~0.8-1.3", fontsize=9)

a = ax[1, 2]
H = pd.read_csv(f"{OUT}/basket_alpha.csv")
x = np.arange(len(H))
a.bar(x - .2, H.SR_OOS, .4, color="tab:green", label="SR_OOS (net)")
a.bar(x + .2, H.SR_alpha, .4, color="0.45", label="hedged alpha")
a.axhline(0, color="k", lw=.6)
a.axhline(H.SR_OOS.median(), color="tab:green", ls="--", lw=1)
a.axhline(H.SR_alpha.median(), color="0.45", ls="--", lw=1)
a.set_xticks([]); a.legend(fontsize=7); a.set_xlabel("24 equal-weight OI baskets")
a.set_title("held-out 2017-26: SR +0.35 (t 2.6) is the index drift\n"
            "hedged alpha median -0.07 (t -0.6)", fontsize=9)
fig.suptitle("QNT-104 — TXO call & call/put-combined open interest vs MTX "
             "(real costs, roll-adjusted, sign frozen on IS)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .96])
fig.savefig(f"{OUT}/qnt104_summary.png", dpi=110)
print("wrote qnt104_summary.png")
