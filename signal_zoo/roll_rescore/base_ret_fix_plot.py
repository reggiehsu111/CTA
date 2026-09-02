"""QNT-21 summary figure: what the _base.py roll fix changes."""
import sys, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW, FIX, INK, MUTED, GRID = "#4A6FE3", "#E07A2F", "#22252a", "#6b7280", "#e6e6e3"
OUT = "/home/ubuntu/mtx/signal_zoo/roll_rescore"
R = pd.read_csv(f"{OUT}/base_ret_fix.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor="#fcfcfb",
                               gridspec_kw={"width_ratios": [1.35, 1]})
for a in (ax1, ax2):
    a.set_facecolor("#fcfcfb")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=MUTED, labelsize=9)

# ── panel 1: the 9 enabled signals on their live variant, full history ────
L = R[(R.live) & (R.window == "full")].sort_values("SR_raw")
y = np.arange(len(L))
for yy, r, f in zip(y, L.SR_raw, L.SR_fix):
    ax1.plot([r, f], [yy, yy], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
ax1.scatter(L.SR_raw, y, s=95, color=RAW, zorder=3, edgecolor="#fcfcfb", linewidth=2, label="live page today (raw pct_change)")
ax1.scatter(L.SR_fix, y, s=95, color=FIX, zorder=3, edgecolor="#fcfcfb", linewidth=2, label="roll-adjusted (proposed)")
for yy, r, f, v in zip(y, L.SR_raw, L.SR_fix, L.variant):
    d = f - r
    ax1.text(max(r, f) + 0.035, yy, ("—" if abs(d) < 5e-4 else f"{d:+.3f}"),
             va="center", fontsize=9, color=INK if abs(d) > 5e-4 else MUTED)
ax1.set_yticks(y)
ax1.set_yticklabels([f"{n[:34]}  [{v}]" for n, v in zip(L.signal, L.variant)], fontsize=9, color=INK)
ax1.set_xlabel("Sharpe (net, live stub cost, runner's own positions)", fontsize=9.5, color=MUTED)
ax1.set_title("The 9 enabled signals — live SR vs roll-adjusted SR\n"
              "6 on c2c move; the 3 on day/ongap are intra-contract and unchanged",
              fontsize=11.5, color=INK, loc="left", pad=12)
ax1.grid(axis="x", color=GRID, lw=0.8); ax1.set_axisbelow(True)
ax1.set_xlim(-0.05, 1.02)
ax1.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower right")

# ── panel 2: buy-and-hold per leg — which legs carry the roll ─────────────
BH = pd.DataFrame({
    "variant":   ["c2c", "o2o", "noonpause", "day", "ongap", "night"],
    "raw":       [0.493, 0.483, -0.253, 0.230, 0.507, 1.149],
    "fix":       [0.700, 0.666,  0.611, 0.230, 0.507, 1.149]})
x = np.arange(len(BH)); w = 0.36
ax2.bar(x - w/2 - 0.01, BH.raw, w, color=RAW, label="raw", zorder=3)
ax2.bar(x + w/2 + 0.01, BH.fix, w, color=FIX, label="roll-adjusted", zorder=3)
for xx, r, f in zip(x, BH.raw, BH.fix):
    ax2.text(xx - w/2 - 0.01, r + (0.03 if r >= 0 else -0.09), f"{r:.2f}", ha="center", fontsize=8.5, color=INK)
    ax2.text(xx + w/2 + 0.01, f + 0.03, f"{f:.2f}", ha="center", fontsize=8.5, color=INK)
ax2.axhline(0, color=MUTED, lw=1)
ax2.set_xticks(x); ax2.set_xticklabels(BH.variant, fontsize=9.5, color=INK)
ax2.set_ylabel("buy-and-hold Sharpe, gross", fontsize=9.5, color=MUTED)
ax2.set_title("Buy-and-hold per return leg, 2001–2026\n"
              "c2c / o2o / noonpause cross a contract\nboundary; day / ongap / night do not",
              fontsize=11.5, color=INK, loc="left", pad=12)
ax2.grid(axis="y", color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper left")
ax2.set_ylim(-0.45, 1.42)

fig.suptitle("QNT-21 — the calendar spread in cta/signals/_base.py's return legs", 
             fontsize=13.5, color=INK, x=0.008, ha="left", y=0.985, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/base_ret_fix.png", dpi=135, facecolor="#fcfcfb")
print("wrote", f"{OUT}/base_ret_fix.png")
