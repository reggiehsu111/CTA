"""QNT-18 part 3: WHY does the `day` advantage reverse outside the 11 daily macro series?

Hypothesis. `day` buy-and-hold is SR 0.230 vs c2c 0.700 - almost all TAIEX drift
is overnight. So switching a position from c2c to day mechanically SUBTRACTS the
drift the position was earning through its net long/short tilt. That is a gain
only when the tilt was unwanted beta contaminating the Sharpe, and a pure loss
when the position is already beta-flat (nothing to strip) or genuinely wants the
tilt. Prediction: Delta SR(day - c2c) should rise with beta_c2c.

Tests it on all three grids at once:
  * QNT-14 daily macro     198 cells   window_sweep_full.csv     (regime=full)
  * QNT-18 slow macro      378 cells   slow_window_sweep.csv
  * QNT-18 registered       11 signals registered_window_sweep.csv (regime=full)
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/ubuntu/mtx/signal_zoo/macro_windows"

def paired(path, key, query=None, lag_note=""):
    d = pd.read_csv(f"{OUT}/{path}")
    if query:
        d = d.query(query)
    p = d.pivot_table(index=key, columns="variant", values=["SR_net", "beta", "mean_exec_w",
                                                            "SR_of_SR", "positive_years"])
    out = pd.DataFrame({
        "dSR":       p["SR_net"]["day"]   - p["SR_net"]["c2c"],
        "dSRofSR":   p["SR_of_SR"]["day"] - p["SR_of_SR"]["c2c"],
        "SR_c2c":    p["SR_net"]["c2c"],   "SR_day": p["SR_net"]["day"],
        "beta_c2c":  p["beta"]["c2c"],     "beta_day": p["beta"]["day"],
        "mw_c2c":    p["mean_exec_w"]["c2c"],
    }).dropna(subset=["dSR", "beta_c2c"])
    out["grid"] = lag_note
    return out

G = {
    "daily macro (QNT-14, day@1 vs c2c@2)":
        paired("window_sweep_full.csv", "cand", "regime=='full'", "daily macro"),
    "slow macro (QNT-18, both @2)":
        paired("slow_window_sweep.csv", "cand", None, "slow macro"),
    "registered signals (QNT-18, declared lags)":
        paired("registered_window_sweep.csv", "signal", "regime=='full'", "registered"),
}

pd.set_option("display.width", 200)
print("=== paired day - c2c, per grid ===")
print(f"{'grid':44s} {'n':>4s} {'medΔSR':>8s} {'win':>6s} {'medΔSRofSR':>11s} "
      f"{'med|β_c2c|':>10s} {'med|β_day|':>10s} {'corr(ΔSR,β_c2c)':>16s}")
for k, d in G.items():
    r = np.corrcoef(d.dSR, d.beta_c2c)[0, 1] if len(d) > 3 else np.nan
    print(f"{k:44s} {len(d):4d} {d.dSR.median():+8.3f} {(d.dSR>0).mean():6.1%} "
          f"{d.dSRofSR.median():+11.3f} {d.beta_c2c.abs().median():10.3f} "
          f"{d.beta_day.abs().median():10.3f} {r:+16.3f}")

ALL = pd.concat(G.values())
print(f"\npooled n={len(ALL)}  corr(ΔSR, beta_c2c) = "
      f"{np.corrcoef(ALL.dSR, ALL.beta_c2c)[0,1]:+.3f}   "
      f"corr(ΔSR, |beta_c2c|) = {np.corrcoef(ALL.dSR, ALL.beta_c2c.abs())[0,1]:+.3f}")

print("\n=== median ΔSR by beta_c2c decile (pooled) ===")
ALL["bdec"] = pd.qcut(ALL.beta_c2c, 10, duplicates="drop")
print(ALL.groupby("bdec")[["dSR", "beta_c2c", "SR_c2c", "SR_day"]]
        .agg(["size", "median"]).round(3).to_string())

print("\n=== the prediction, stated as a rule ===")
hi = ALL[ALL.beta_c2c > 0.15]; lo = ALL[ALL.beta_c2c.abs() <= 0.05]
print(f"  cells with beta_c2c > +0.15 (beta to strip):   n={len(hi):4d}  "
      f"median ΔSR {hi.dSR.median():+.3f}  win {(hi.dSR>0).mean():.1%}")
print(f"  cells with |beta_c2c| <= 0.05 (already flat):  n={len(lo):4d}  "
      f"median ΔSR {lo.dSR.median():+.3f}  win {(lo.dSR>0).mean():.1%}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
cols = {"daily macro": "#1f77b4", "slow macro": "#ff7f0e", "registered": "#d62728"}
for g, c in cols.items():
    d = ALL[ALL.grid == g]
    ax[0].scatter(d.beta_c2c, d.dSR, s=(46 if g == "registered" else 12),
                  alpha=(0.95 if g == "registered" else 0.45), c=c, label=f"{g} (n={len(d)})",
                  edgecolors=("k" if g == "registered" else "none"), linewidths=0.5, zorder=3 if g=="registered" else 2)
b = np.polyfit(ALL.beta_c2c, ALL.dSR, 1)
xs = np.linspace(ALL.beta_c2c.min(), ALL.beta_c2c.max(), 50)
ax[0].plot(xs, np.polyval(b, xs), "k--", lw=1.4,
           label=f"fit  ΔSR = {b[0]:+.2f}·β {b[1]:+.3f}")
ax[0].axhline(0, color="grey", lw=0.8); ax[0].axvline(0, color="grey", lw=0.8)
ax[0].set_xlabel("beta of the c2c cell vs MTX buy-and-hold")
ax[0].set_ylabel("ΔSR_net  (day − c2c)")
ax[0].set_title("`day` helps exactly where c2c carried index beta\n"
                "(day B&H SR 0.230 vs c2c 0.700 — the window strips the drift)", fontsize=10)
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

med = ALL.groupby("grid")["dSR"].median()
order = ["daily macro", "slow macro", "registered"]
ax[1].bar(range(3), [med[g] for g in order], color=[cols[g] for g in order])
for i, g in enumerate(order):
    d = ALL[ALL.grid == g]
    ax[1].text(i, med[g] + (0.006 if med[g] >= 0 else -0.018),
               f"{med[g]:+.3f}\nwin {(d.dSR>0).mean():.0%}\nn={len(d)}",
               ha="center", fontsize=9)
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(order)
ax[1].set_ylabel("median paired ΔSR_net (day − c2c)")
ax[1].set_title("QNT-14's +0.071 does NOT generalise", fontsize=10)
ax[1].grid(alpha=0.25, axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/day_effect_generalisation.png", dpi=115)
print(f"\nplot -> {OUT}/day_effect_generalisation.png")
