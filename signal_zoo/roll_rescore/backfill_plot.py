"""QNT-21 backfill figure: what the stored-history rewrite changes on the live board."""
import sys, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from db_utils import engine

RAW, FIX, INK, MUTED, GRID = "#4A6FE3", "#E07A2F", "#22252a", "#6b7280", "#e6e6e3"
OUT = "/home/ubuntu/mtx/signal_zoo/roll_rescore"

imp = pd.read_csv(f"{OUT}/live_board_impact_final.csv")
new = pd.read_csv(f"{OUT}/backfill_rows.csv", parse_dates=["date"]).set_index(
    ["date", "signal_name", "variant"])["new_pnl_1d"]
vals = pd.read_sql("SELECT date, signal_name, variant, pnl_1d FROM mtx_signal_values", engine)
vals["date"] = pd.to_datetime(vals["date"])
vals = vals.set_index(["date", "signal_name", "variant"])
vals["pnl_new"] = vals["pnl_1d"]
vals.loc[new.index, "pnl_new"] = new.values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor="#fcfcfb",
                               gridspec_kw={"width_ratios": [1.3, 1]})
for a in (ax1, ax2):
    a.set_facecolor("#fcfcfb")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=MUTED, labelsize=9)

# ── panel 1: enabled signals, SR on the stored history before vs after ────
L = imp[imp.enabled].sort_values("SR_old")
y = np.arange(len(L))
for yy, r, f in zip(y, L.SR_old, L.SR_new):
    ax1.plot([r, f], [yy, yy], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
ax1.scatter(L.SR_old, y, s=95, color=RAW, zorder=3, edgecolor="#fcfcfb", linewidth=2,
            label="stored history today (raw)")
ax1.scatter(L.SR_new, y, s=95, color=FIX, zorder=3, edgecolor="#fcfcfb", linewidth=2,
            label="after backfill (roll-adjusted)")
for yy, r, f in zip(y, L.SR_old, L.SR_new):
    d = f - r
    ax1.text(max(r, f) + 0.02, yy, ("—" if abs(d) < 5e-4 else f"{d:+.3f}"),
             va="center", fontsize=9, color=INK if abs(d) > 5e-4 else MUTED)
ax1.set_yticks(y)
ax1.set_yticklabels([f"{n[:34]}  [{v}]" for n, v in zip(L.signal, L.variant)],
                    fontsize=9, color=INK)
ax1.set_xlabel("Sharpe of stored pnl_1d, full history", fontsize=9.5, color=MUTED)
ax1.set_title("The 9 enabled signals — stored PnL before vs after the backfill\n"
              "6 on c2c move +0.009…+0.071; the 3 on day/ongap are untouched",
              fontsize=11.5, color=INK, loc="left", pad=12)
ax1.grid(axis="x", color=GRID, lw=0.8); ax1.set_axisbelow(True)
ax1.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower right")

# ── panel 2: equal-weight book of the 6 enabled c2c signals, cumulative ────
c2c = L[L.variant == "c2c"]["signal"].tolist()
sub = vals[vals.index.get_level_values("variant") == "c2c"]
sub = sub[sub.index.get_level_values("signal_name").isin(c2c)]
old_b = sub.groupby(level="date")["pnl_1d"].mean().sort_index().fillna(0).cumsum()
new_b = sub.groupby(level="date")["pnl_new"].mean().sort_index().fillna(0).cumsum()
ax2.plot(old_b.index, old_b.values, color=RAW, lw=1.7, label="stored (raw)")
ax2.plot(new_b.index, new_b.values, color=FIX, lw=1.7, label="after backfill")
ax2.set_title("Equal-weight book of the 6 enabled c2c signals\n"
              f"cumulative pnl_1d: {old_b.iloc[-1]:.2f} → {new_b.iloc[-1]:.2f} "
              f"over {len(old_b)} sessions", fontsize=11.5, color=INK, loc="left", pad=12)
ax2.set_ylabel("cumulative sum of pnl_1d", fontsize=9.5, color=MUTED)
ax2.grid(color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper left")

fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.text(0.006, 0.012, "QNT-21 — 6,277 of 382,926 stored rows rewritten (1.64%): "
         "c2c 2,643 · o2o 2,562 · noonpause 1,072. pnl_1d only; positions untouched.",
         fontsize=9.5, color=MUTED, ha="left")
fig.savefig(f"{OUT}/backfill_impact.png", dpi=130, facecolor="#fcfcfb")
print("wrote backfill_impact.png")
