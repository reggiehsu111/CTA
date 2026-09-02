"""QNT-13 figures for the Linear write-up."""
import sys, pickle, warnings
sys.path.insert(0,"/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0,"/home/ubuntu/mtx")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import cta

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8985"
BLUE="#2a78d6"; ORANGE="#eb6834"; RED="#e34948"; GRID="#e6e5e1"
mpl.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "text.color":INK,"axes.labelcolor":INK2,"xtick.color":INK2,"ytick.color":INK2,
    "axes.edgecolor":GRID,"font.size":10})

def strip(ax, xgrid=True):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

T = pd.read_csv("signal_zoo/roll_rescore/rescore_full.csv")
SHORT = lambda s: s.replace("_selftanh","").replace("_signth","").replace("_pct_chg","")[:34]

# ── Figure 1: ΔSR_net when scored on roll-adjusted returns ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True)
order = T[T.window=="full"].sort_values("SR_net_d")["signal"].tolist()
for ax, win in zip(axes, ("full", "2019+")):
    t = T[T.window==win].set_index("signal").loc[order]
    y = np.arange(len(t)); d = t["SR_net_d"].values
    cols = [BLUE if v>=0 else RED for v in d]
    ax.barh(y, d, height=0.62, color=cols, edgecolor=SURF, linewidth=2, zorder=3)
    for i,(v,en) in enumerate(zip(d, t["enabled"])):
        ax.text(v + (0.006 if v>=0 else -0.006), i, f"{v:+.3f}", va="center",
                ha="left" if v>=0 else "right", fontsize=9, color=INK2, zorder=4)
    ax.axvline(0, color=MUTED, lw=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'● ' if e else '○ '}{SHORT(s)}" for s,e in zip(t.index, t["enabled"])],
                       fontsize=9)
    ax.set_xlim(-0.30, 0.14); ax.set_xlabel("ΔSharpe (net)")
    ax.set_title(f"{win}", fontsize=11, color=INK, loc="left", pad=8)
    strip(ax)
fig.suptitle("Scoring on roll-adjusted returns moves net Sharpe by −0.218 to +0.083\n"
             "MTX 1d, c2c shift(2), realistic cost (70 TWD/side + 4e-5).  ● = enabled   ○ = disabled",
             fontsize=12.5, color=INK, x=0.012, ha="left", y=0.99)
axes[1].tick_params(axis="y", length=0)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("signal_zoo/roll_rescore/delta_sr.png", dpi=140)
print("wrote delta_sr.png")

# ── Figure 2: cumulative return, raw vs roll-adjusted ──────────────────────
A = cta.load_asset("mtx","1d")
D = pickle.load(open("signal_zoo/roll_rescore/signals.pkl","rb"))["sigs"]
r_raw, r_adj = A["close"].pct_change(), A.returns
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
panels = [("Buy-and-hold MTX", pd.Series(1.0, index=A.index), 0.496, 0.703),
          ("opt_put_mo_oi_selftanh_w60  (worst affected)", D["opt_put_mo_oi_selftanh_w60"].reindex(A.index).shift(2), 0.254, 0.036)]
for ax,(title,pos,s0,s1) in zip(axes, panels):
    for r,c,lab,srv in ((r_raw,ORANGE,"raw  close.pct_change()",s0),(r_adj,BLUE,"roll-adjusted  asset.returns",s1)):
        cum = (pos*r).fillna(0).cumsum()
        ax.plot(cum.index, cum.values, color=c, lw=2, label=f"{lab}   SR {srv:.3f}", zorder=3)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_ylabel("cumulative gross return")
    ax.axhline(0, color=MUTED, lw=1, zorder=2); strip(ax, xgrid=False)
fig.suptitle("The calendar spread is a persistent drag on the raw series, and was most of the options signals' apparent edge",
             fontsize=12.5, color=INK, x=0.012, ha="left", y=0.99)
fig.tight_layout(rect=[0,0,1,0.90]); fig.savefig("signal_zoo/roll_rescore/cumret.png", dpi=140)
print("wrote cumret.png")
