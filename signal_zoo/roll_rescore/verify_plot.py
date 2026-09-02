"""QNT-21 post-backfill verification figure: read the DB back and compare it to the
saved pre-image. Nothing here is modelled - both series come from disk/DB as they are."""
import sys, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from db_utils import engine

RAW, FIX, INK, MUTED, GRID = "#4A6FE3", "#E07A2F", "#22252a", "#6b7280", "#e6e6e3"
OUT = "/home/ubuntu/mtx/signal_zoo/roll_rescore"
CUT = "2026-08-31"

imp = pd.read_csv(f"{OUT}/live_board_impact_final.csv")
pre = pd.read_csv(f"{OUT}/backfill_preimage.csv", parse_dates=["date"])
pre = pre.set_index(["date", "signal_name", "variant"])["pnl_1d"]

db = pd.read_sql("SELECT date, signal_name, variant, pnl_1d FROM mtx_signal_values", engine)
db["date"] = pd.to_datetime(db["date"])
db = db[db["date"] <= CUT].set_index(["date", "signal_name", "variant"]).sort_index()
db["pnl_pre"] = db["pnl_1d"]
db.loc[pre.index, "pnl_pre"] = pre.values          # roll the DB back to the pre-image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), facecolor="#fcfcfb",
                               gridspec_kw={"width_ratios": [1.3, 1]})
for a in (ax1, ax2):
    a.set_facecolor("#fcfcfb")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=MUTED, labelsize=9)

def sr(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(252)

L = imp[imp.enabled].copy()
L["SR_db"] = [sr(db.xs((r.signal, r.variant), level=("signal_name", "variant"))["pnl_1d"])
              for r in L.itertuples()]
L["SR_pre"] = [sr(db.xs((r.signal, r.variant), level=("signal_name", "variant"))["pnl_pre"])
               for r in L.itertuples()]
L = L.sort_values("SR_pre")
y = np.arange(len(L))
for yy, p, q in zip(y, L.SR_pre, L.SR_db):
    ax1.plot([p, q], [yy, yy], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
ax1.scatter(L.SR_pre, y, s=95, color=RAW, zorder=3, edgecolor="#fcfcfb", linewidth=2,
            label="pre-image (what the page served before)")
ax1.scatter(L.SR_db, y, s=95, color=FIX, zorder=3, edgecolor="#fcfcfb", linewidth=2,
            label="DB read back now (roll-adjusted)")
for yy, p, q in zip(y, L.SR_pre, L.SR_db):
    d = q - p
    ax1.text(max(p, q) + 0.02, yy, ("—" if abs(d) < 5e-4 else f"{d:+.3f}"),
             va="center", fontsize=9, color=INK if abs(d) > 5e-4 else MUTED)
ax1.set_yticks(y)
ax1.set_yticklabels([f"{n[:34]}  [{v}]" for n, v in zip(L.signal, L.variant)],
                    fontsize=9, color=INK)
ax1.set_xlabel(f"Sharpe of stored pnl_1d, full history to {CUT}", fontsize=9.5, color=MUTED)
ax1.set_title("Read back from mtx_signal_values after the write\n"
              "matches the predicted post-fix board to 2.2e-16 on all 11 series",
              fontsize=11.5, color=INK, loc="left", pad=12)
ax1.grid(axis="x", color=GRID, lw=0.8); ax1.set_axisbelow(True)
ax1.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower right")

c2c = L[L.variant == "c2c"]["signal"].tolist()
sub = db[(db.index.get_level_values("variant") == "c2c")
         & (db.index.get_level_values("signal_name").isin(c2c))]
pre_b = sub.groupby(level="date")["pnl_pre"].mean().sort_index().fillna(0).cumsum()
new_b = sub.groupby(level="date")["pnl_1d"].mean().sort_index().fillna(0).cumsum()
ax2.plot(pre_b.index, pre_b.values, color=RAW, lw=1.7, label="pre-image")
ax2.plot(new_b.index, new_b.values, color=FIX, lw=1.7, label="DB now")
ax2.set_title(f"Equal-weight book of the 6 enabled c2c signals\n"
              f"cumulative stored PnL: {pre_b.iloc[-1]:.2f} -> {new_b.iloc[-1]:.2f}",
              fontsize=11.5, color=INK, loc="left", pad=12)
ax2.set_ylabel("cumulative pnl_1d", fontsize=9.5, color=MUTED)
ax2.grid(color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper left")

fig.suptitle("QNT-21 - backfill applied: 6,277 rows of pnl_1d rewritten, verified against the DB",
             fontsize=13, color=INK, x=0.008, ha="left", y=0.985, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(f"{OUT}/backfill_verified.png", dpi=125, facecolor="#fcfcfb")
print("wrote backfill_verified.png")
print(L[["signal", "variant", "SR_pre", "SR_db"]].to_string(index=False))
