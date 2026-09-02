"""QNT-12: robustness + TCA pack for the only family that came near the gates.

igrea = Kilian index of global real economic activity (dry-bulk freight rates),
monthly, pub lag 40d. Thesis: Taiwan is a globally-geared export cycle, so a
global real-activity read should have directional information on TAIEX.

Three tests, all of which a result must ship with:
  (1) FAMILY AGREEMENT  — does it hold across all 6 transforms x 3 windows,
      or only in the one cell that was reported?
  (2) IS/OOS with the sign FROZEN ON IS + a per-year SR table
  (3) COST CURVE — SR as a function of cost, from gross to 3x realistic
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context

OUT = "/home/ubuntu/mtx/signal_zoo/macro_sweep"
IS_END, OOS_START = "2018-12-31", "2019-01-01"
REAL = dict(fixed_per_side=70.0, fee_rate=0.00004)
ctx = build_context(); A = ctx.asset
x = ctx.macro("igrea")

TF = {"selfz": ops.selfz, "robustz": ops.robust_z, "bdtanh": ops.bd_selftanh,
      "rankc": ops.rank_c, "signth": ops.sign_thresh, "dev": ops.dev}
WS = (60, 120, 252)

def build(t, w):
    return cta.normalize_signal(TF[t](x, w).replace([np.inf,-np.inf], np.nan),
                                method="tanh", window=252)

# ── (1) family agreement ──────────────────────────────────────────────────
rows = []
for t in TF:
    for w in WS:
        s = build(t, w)
        sgn = int(cta.signal_stats(s, A, end=IS_END, auto_flip=True, roll_adjusted=True, **REAL)["sign"])
        f = cta.signal_stats(s*sgn, A, auto_flip=False, roll_adjusted=True, **REAL)
        i = cta.signal_stats(s*sgn, A, end=IS_END, auto_flip=False, roll_adjusted=True, **REAL)
        o = cta.signal_stats(s*sgn, A, start=OOS_START, auto_flip=False, roll_adjusted=True, **REAL)
        rows.append(dict(transform=t, window=w, sign=sgn, SR_IS=i["SR_net"], SR_OOS=o["SR_net"],
                         SR_full=f["SR_net"], SR_of_SR=f["SR_of_SR"],
                         positive_years=f["positive_years"], beta=f["beta"],
                         held_pct=f["held_pct"], turnover_ann=f["turnover_ann"],
                         max_dd_pct=f["max_dd_pct"], max_dd_days=f["max_dd_days"]))
fam = pd.DataFrame(rows)
fam.to_csv(f"{OUT}/igrea_family.csv", index=False)
pd.set_option("display.width", 220)
print("=== (1) FAMILY AGREEMENT — igrea, all 18 cells, IS-frozen sign, realistic cost ===")
print(fam.round(3).to_string(index=False))
print(f"\n  signs agree: {fam['sign'].nunique()==1} ({sorted(fam['sign'].unique())})")
print(f"  SR_full > 0 in {int((fam.SR_full>0).sum())}/18 cells; median {fam.SR_full.median():.3f}")
print(f"  SR_of_SR > 0.6 in {int((fam.SR_of_SR>0.6).sum())}/18 cells  <-- the gate")

# ── (2) per-year SR, sign frozen on IS ────────────────────────────────────
BEST_T, BEST_W = "selfz", 120
s = build(BEST_T, BEST_W)
sgn = int(cta.signal_stats(s, A, end=IS_END, auto_flip=True, roll_adjusted=True, **REAL)["sign"])
e = (s*sgn).shift(2).reindex(A.index)
ret = A.returns
pnl = (e*ret)
close = A["close"]
def net(fixed, fee):
    tc = e.fillna(0).diff().abs() * (fixed/(close*50.0) + fee)
    return (pnl - tc).dropna()
n = net(REAL["fixed_per_side"], REAL["fee_rate"])
yr = n.groupby(n.index.year).apply(lambda v: np.sqrt(252)*v.mean()/v.std() if len(v)>20 and v.std()>0 else np.nan).dropna()
print(f"\n=== (2) PER-YEAR SR — igrea|{BEST_T}|w{BEST_W}, sign {sgn:+d} frozen on IS<=2018 ===")
print(yr.round(2).to_string())
print(f"  positive years {(yr>0).mean():.1%} ({int((yr>0).sum())}/{len(yr)})   min {yr.min():.2f}  max {yr.max():.2f}")
print(f"  SR_of_SR = mean/std = {yr.mean()/yr.std():.3f}   (gate > 0.6)")
print(f"  IS SR {np.sqrt(252)*n[:IS_END].mean()/n[:IS_END].std():.3f}   OOS SR {np.sqrt(252)*n[OOS_START:].mean()/n[OOS_START:].std():.3f}")
bh = ret.dropna(); print(f"  buy&hold SR (roll-adj) {np.sqrt(252)*bh.mean()/bh.std():.3f}")

# ── (3) cost curve ────────────────────────────────────────────────────────
print(f"\n=== (3) COST CURVE — igrea|{BEST_T}|w{BEST_W} ===")
curve = []
for lbl, fx, fe in [("gross (no cost)", 0.0, 0.0),
                    ("framework stub 20+2e-5", 20.0, 0.00002),
                    ("+ 期交稅 (20+4e-5)", 20.0, 0.00004),
                    ("realistic 70+4e-5 (1pt slip)", 70.0, 0.00004),
                    ("2x realistic", 140.0, 0.00008),
                    ("3x realistic", 210.0, 0.00012)]:
    v = net(fx, fe); curve.append((lbl, float(np.sqrt(252)*v.mean()/v.std())))
    print(f"  {lbl:32s} SR {curve[-1][1]:+.3f}")
print(f"  turnover {e.fillna(0).diff().abs().mean()*252:.2f}/yr, held {float((e.abs()>1e-9).mean())*100:.1f}% of days,"
      f" mean|pos| {float(e.abs().mean()):.3f}")

# ── plots ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
for lbl, fx, fe in [("gross", 0.0, 0.0), ("stub", 20.0, 0.00002), ("realistic", 70.0, 0.00004)]:
    net(fx, fe).cumsum().plot(ax=ax[0][0], label=f"{lbl}")
ax[0][0].axvline(pd.Timestamp(OOS_START), color="k", ls="--", lw=1)
ax[0][0].set_title(f"igrea|{BEST_T}|w{BEST_W} 累積報酬 — 成本敏感度 (虛線右側 = OOS)"); ax[0][0].legend(); ax[0][0].grid(alpha=.3)
yr.plot(kind="bar", ax=ax[0][1], color=["tab:green" if v>0 else "tab:red" for v in yr])
ax[0][1].axhline(0, color="k", lw=.8); ax[0][1].set_title(f"逐年 Sharpe (net, 實際成本) — {(yr>0).mean():.0%} 正報酬年, 門檻 65%")
fam.pivot(index="transform", columns="window", values="SR_full").plot(kind="bar", ax=ax[1][0])
ax[1][0].axhline(0, color="k", lw=.8); ax[1][0].set_title("族群一致性: SR_full 跨 6 轉換 x 3 視窗"); ax[1][0].grid(alpha=.3)
ax[1][1].scatter(fam.SR_IS, fam.SR_OOS, s=60)
lim = [min(fam.SR_IS.min(), fam.SR_OOS.min())-.1, max(fam.SR_IS.max(), fam.SR_OOS.max())+.1]
ax[1][1].plot(lim, lim, "k--", lw=1); ax[1][1].axhline(0, color="gray", lw=.6); ax[1][1].axvline(0, color="gray", lw=.6)
ax[1][1].set_xlabel("SR 樣本內 (<=2018)"); ax[1][1].set_ylabel("SR 樣本外 (2019+)")
ax[1][1].set_title("IS vs OOS — 符號凍結於樣本內")
plt.tight_layout(); plt.savefig(f"{OUT}/igrea_robustness.png", dpi=110)
print(f"\nsaved {OUT}/igrea_robustness.png")
