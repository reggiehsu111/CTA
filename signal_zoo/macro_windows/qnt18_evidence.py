"""QNT-18 evidence figure: the `day` window advantage does not generalise, and
the reason is that day and c2c are only weakly correlated readings of the edge.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/ubuntu/mtx/signal_zoo/macro_windows"
d = pd.read_csv(f"{OUT}/detilt_matched.csv"); d = d[d.how == "raw"]
GR = ["daily macro", "slow macro", "registered"]
C  = {"daily macro": "#1f77b4", "slow macro": "#ef8a3c", "registered": "#c9333b"}
P  = {g: d[d.grid == g].pivot_table(index="cand", columns="variant", values="SR").dropna() for g in GR}
for g in GR: P[g]["dSR"] = P[g]["day"] - P[g]["c2c"]

fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

# (0,0) headline: paired median ΔSR by grid, window channel and headline channel
lag = pd.read_csv(f"{OUT}/lag_decomposition.csv").query("regime=='full'")
w = {"daily macro": (+0.0508, 0.611, 198), "slow macro": (-0.0322, 0.413, 378),
     "registered": (-0.1069, 0.182, 11)}
h = {"daily macro": (+0.0708, 0.636), "slow macro": (-0.0322, 0.413), "registered": (-0.0894, 0.364)}
x = np.arange(3)
ax[0,0].bar(x-0.19, [w[g][0] for g in GR], 0.36, color=[C[g] for g in GR], label="window channel  day@2 − c2c@2")
ax[0,0].bar(x+0.19, [h[g][0] for g in GR], 0.36, color=[C[g] for g in GR], alpha=0.45, hatch="//",
            label="headline  day@declared-lag − c2c@2")
for i, g in enumerate(GR):
    ax[0,0].text(i-0.19, w[g][0]+(0.006 if w[g][0]>0 else -0.016), f"{w[g][0]:+.3f}\nwin {w[g][1]:.0%}", ha="center", fontsize=8.5)
    ax[0,0].text(i+0.19, h[g][0]+(0.006 if h[g][0]>0 else -0.016), f"{h[g][0]:+.3f}\nwin {h[g][1]:.0%}", ha="center", fontsize=8.5)
ax[0,0].axhline(0, color="k", lw=1); ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels([f"{g}\nn={w[g][2]}" for g in GR])
ax[0,0].set_ylabel("paired median ΔSR_net")
ax[0,0].set_ylim(-0.135, 0.105)
ax[0,0].set_title("QNT-14's +0.071 does not survive outside the 11 daily macro series\n"
                  "per-SERIES test (the real n): macro 8/11 p=0.23 n.s.  ·  registered 2/11 p=0.014",
                  fontsize=10.5)
ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=0.25, axis="y")

# (0,1) the driver: ΔSR vs the strength of the c2c baseline
for g in GR:
    ax[0,1].scatter(P[g]["c2c"], P[g]["dSR"], s=(60 if g=="registered" else 11), c=C[g],
                    alpha=(0.95 if g=="registered" else 0.4), label=f"{g}  r={np.corrcoef(P[g].dSR,P[g].c2c)[0,1]:+.2f}",
                    edgecolors=("k" if g=="registered" else "none"), lw=0.5, zorder=3 if g=="registered" else 2)
A = pd.concat(P.values())
b = np.polyfit(A.c2c, A.dSR, 1); xs = np.linspace(A.c2c.min(), A.c2c.max(), 40)
ax[0,1].plot(xs, np.polyval(b, xs), "k--", lw=1.4, label=f"pooled r={np.corrcoef(A.dSR,A.c2c)[0,1]:+.2f}")
ax[0,1].axhline(0, color="grey", lw=0.8); ax[0,1].axvline(0, color="grey", lw=0.8)
ax[0,1].set_xlabel("SR_net of the c2c cell (matched lag 2)"); ax[0,1].set_ylabel("ΔSR_net (day − c2c)")
ax[0,1].set_title("`day` wins exactly where c2c happens to read LOW\n"
                  "(partly mechanical — see lower-left for the non-mechanical form)", fontsize=10.5)
ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=0.25)

# (1,0) day vs c2c scatter — the weak correlation itself
for g in GR:
    ax[1,0].scatter(P[g]["c2c"], P[g]["day"], s=(60 if g=="registered" else 11), c=C[g],
                    alpha=(0.95 if g=="registered" else 0.4),
                    label=f"{g}  corr={np.corrcoef(P[g].day,P[g].c2c)[0,1]:+.2f}",
                    edgecolors=("k" if g=="registered" else "none"), lw=0.5, zorder=3 if g=="registered" else 2)
lo, hi = -0.35, 0.85
ax[1,0].plot([lo,hi],[lo,hi], "k--", lw=1.2, label="y = x")
ax[1,0].set_xlim(lo,hi); ax[1,0].set_ylim(lo,hi)
ax[1,0].set_xlabel("SR_net  c2c@2"); ax[1,0].set_ylabel("SR_net  day@2")
ax[1,0].set_title("median SR_day ≈ +0.08…+0.16 in EVERY grid while median SR_c2c\n"
                  "ranges +0.06 → +0.40; corr(day,c2c) only +0.18 on the macro grid", fontsize=10.5)
ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=0.25)

# (1,1) the 11 registered signals, per signal
R = P["registered"].sort_values("c2c")
y = np.arange(len(R))
ax[1,1].barh(y-0.2, R["c2c"], 0.4, color="#4c72b0", label="c2c @2")
ax[1,1].barh(y+0.2, R["day"], 0.4, color="#c9333b", label="day @2")
ax[1,1].set_yticks(y); ax[1,1].set_yticklabels([i.replace("_"," ")[:34] for i in R.index], fontsize=7.6)
ax[1,1].axvline(0, color="k", lw=1)
ax[1,1].set_xlabel("SR_net (full history, net of real costs, declared frozen sign)")
ax[1,1].set_title("The 11 registered signals: `day` loses on 9 of 11\n"
                  "(matched lag 2; the two options signals are enabled=False)", fontsize=10.5)
ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=0.25, axis="x")

plt.tight_layout(); plt.savefig(f"{OUT}/qnt18_day_window_generalisation.png", dpi=110)
print("saved", f"{OUT}/qnt18_day_window_generalisation.png")
