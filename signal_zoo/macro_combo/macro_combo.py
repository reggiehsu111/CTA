"""QNT-16: does an equal-weight sleeve of igrea + epu_global + kr_kospi reach
SR_of_SR > 0.6, which none of the three reaches alone?

Discipline (台指期 standing brief):
  * legs are the EXACT QNT-12 best cells, rebuilt from the same PIT-aligned
    `ctx.macro*` inputs and the same transform/window
  * sign is fitted on IS (<=2018-12-31) with auto_flip and then FROZEN — it is
    never re-derived on the full sample and this script never picks one by hand
  * scored roll_adjusted=True at realistic MTX cost (70 TWD/side + 4e-5)
  * every subset of the three is reported, not the best one
"""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context

OUT = "/home/ubuntu/mtx/signal_zoo/macro_combo"
IS_END, OOS_START = "2018-12-31", "2019-01-01"
REAL = dict(fixed_per_side=70.0, fee_rate=0.00004)
STUB = dict(fixed_per_side=20.0, fee_rate=0.00002)
POINT_VALUE = 50.0

ctx   = build_context()
ASSET = ctx.asset
print(f"asset: {len(ASSET.index)} rows, {ASSET.index.min().date()} -> {ASSET.index.max().date()}")

# ── The three legs: QNT-12 best cell per family ────────────────────────────
LEGS = {
    "igrea":      dict(series="igrea",      kind="level", tf="selfz",   w=120),
    "epu_global": dict(series="epu_global", kind="level", tf="robustz", w=120),
    "kr_kospi":   dict(series="kr_kospi",   kind="yoy",   tf="signth",  w=120),
}
TF = {"selfz": ops.selfz, "robustz": ops.robust_z, "signth": ops.sign_thresh}

cat = cta.macro_catalog()
legs, signs = {}, {}
for name, spec in LEGS.items():
    if spec["kind"] == "yoy":
        per = {"M": 12, "Q": 4, "D": 252, "W": 52}.get(cat.loc[spec["series"], "freq"], 12)
        x = ctx.macro_yoy(spec["series"], per)
    else:
        x = ctx.macro(spec["series"])
    sig = TF[spec["tf"]](x.astype(float), spec["w"]).replace([np.inf, -np.inf], np.nan)
    sig = cta.normalize_signal(sig, method="tanh", window=252)
    # sign fitted on IS ONLY, then frozen
    s = int(cta.signal_stats(sig, ASSET, end=IS_END, auto_flip=True,
                             roll_adjusted=True, **REAL)["sign"])
    signs[name] = s
    legs[name] = (sig * s).rename(name)
    print(f"leg {name:11s} {spec['series']}_{spec['kind']}|{spec['tf']}|w{spec['w']}  "
          f"IS-frozen sign = {s:+d}   n={sig.dropna().shape[0]}")

# ── Net-PnL series, replicating signal_stats' arithmetic exactly ───────────
close, ret = ASSET["close"], ASSET.returns
def net_pnl(sig, **cost):
    e  = sig.reindex(ASSET.index).astype(float).shift(2)
    to = e.fillna(0).diff().abs()
    cp = cost["fixed_per_side"] / (close * POINT_VALUE) + cost["fee_rate"]
    return (e * ret - to * cp).rename(getattr(sig, "name", "sig"))

# ── STEP 2: PnL correlation across the three legs ─────────────────────────
pnls = pd.concat([net_pnl(v, **REAL) for v in legs.values()], axis=1).dropna()
corr = pnls.corr()
print(f"\n=== STEP 2: net-PnL correlation matrix (overlap {pnls.index.min().date()} "
      f"-> {pnls.index.max().date()}, n={len(pnls)}) ===")
print(corr.round(3).to_string())
corr.to_csv(f"{OUT}/pnl_correlation.csv")
sigcorr = pd.concat(legs.values(), axis=1).dropna().corr()
print("\n(signal-level correlation, for reference)")
print(sigcorr.round(3).to_string())

# ── STEP 3/4: score every subset, full / IS / OOS ─────────────────────────
subsets = {}
for k in (1, 2, 3):
    for c in combinations(legs, k):
        nm = "EW(" + "+".join(c) + ")" if k > 1 else c[0]
        subsets[nm] = pd.concat([legs[n] for n in c], axis=1).mean(axis=1, skipna=False)

def row(nm, sig):
    f = cta.signal_stats(sig, ASSET, auto_flip=False, roll_adjusted=True, **REAL)
    i = cta.signal_stats(sig, ASSET, end=IS_END,     auto_flip=False, roll_adjusted=True, **REAL)
    o = cta.signal_stats(sig, ASSET, start=OOS_START, auto_flip=False, roll_adjusted=True, **REAL)
    fs= cta.signal_stats(sig, ASSET, auto_flip=False, roll_adjusted=True, **STUB)
    fr= cta.signal_stats(sig, ASSET, auto_flip=False, roll_adjusted=True,
                         fixed_per_side=210.0, fee_rate=1.2e-4)   # 3x realistic
    return {"combo": nm, "n_legs": nm.count("+") + 1,
            "SR_IS": i["SR_net"], "SR_OOS": o["SR_net"], "SR_full": f["SR_net"],
            "SR_gross": f["SR_gross"], "SR_stub": fs["SR_net"], "SR_3xreal": fr["SR_net"],
            "SR_of_SR": f["SR_of_SR"], "positive_years": f["positive_years"],
            "yr_sr_min": f["yr_sr_min"], "yr_sr_mean": f["yr_sr_mean"],
            "yr_sr_std": f["yr_sr_std"], "n_years": f["n_years"], "beta": f["beta"],
            "mean_pos": float(sig.reindex(ASSET.index).shift(2).mean()),
            "max_dd_days": f["max_dd_days"], "max_dd_pct": f["max_dd_pct"],
            "turnover_ann": f["turnover_ann"], "held_pct": f["held_pct"],
            "SR_of_SR_IS": i["SR_of_SR"], "SR_of_SR_OOS": o["SR_of_SR"],
            "posyr_IS": i["positive_years"], "posyr_OOS": o["positive_years"]}

res = pd.DataFrame([row(nm, s) for nm, s in subsets.items()])
res["gate_srsr"]  = res.SR_of_SR > 0.6
res["gate_posyr"] = res.positive_years >= 0.65
res["gate_beta"]  = res.beta.abs() < 0.15
res["gate_nyr"]   = res.n_years >= 5
res["n_gates"]    = res[["gate_srsr","gate_posyr","gate_beta","gate_nyr"]].sum(axis=1)
res["OOS_minus_IS"] = res.SR_OOS - res.SR_IS
res = res.sort_values(["n_legs", "SR_of_SR"], ascending=[True, False])
res.to_csv(f"{OUT}/combo_scoreboard.csv", index=False)

pd.set_option("display.width", 260, "display.max_columns", 50)
print("\n=== STEP 3: all subsets, roll-adjusted, realistic cost, IS-frozen signs ===")
print(res[["combo","SR_full","SR_of_SR","positive_years","beta","n_years","max_dd_days",
           "turnover_ann","n_gates"]].round(3).to_string(index=False))
print("\n=== gates (SR_of_SR>0.6 | pos_yrs>=65% | |beta|<0.15 | n_years>=5) ===")
print(res[["combo","gate_srsr","gate_posyr","gate_beta","gate_nyr","n_gates"]].to_string(index=False))
print("\n=== STEP 4: IS vs OOS ===")
print(res[["combo","SR_IS","SR_OOS","OOS_minus_IS","SR_full","SR_of_SR_IS","SR_of_SR_OOS",
           "posyr_IS","posyr_OOS"]].round(3).to_string(index=False))
print("\n=== cost curve ===")
print(res[["combo","SR_gross","SR_stub","SR_full","SR_3xreal"]].round(3).to_string(index=False))

# ── benchmark ──────────────────────────────────────────────────────────────
bh = pd.Series(1.0, index=ASSET.index)
bstat = cta.signal_stats(bh, ASSET, auto_flip=False, roll_adjusted=True, **REAL)
print(f"\nbuy-and-hold (roll-adjusted): SR_net {bstat['SR_net']}  SR_of_SR {bstat['SR_of_SR']}  "
      f"pos_yrs {bstat['positive_years']}  n_years {bstat['n_years']}")

# ── per-year SR table ──────────────────────────────────────────────────────
def yearly_sr(sig):
    p = net_pnl(sig, **REAL).dropna()
    ppy = int(ASSET.periods_per_year)
    return p.groupby(p.index.year).apply(
        lambda x: np.sqrt(ppy)*x.mean()/x.std() if len(x) > 20 and x.std() > 0 else np.nan).dropna()

EW3 = "EW(" + "+".join(legs) + ")"
yr = pd.concat({n: yearly_sr(subsets[n]) for n in list(legs) + [EW3]}, axis=1)
yr.to_csv(f"{OUT}/yearly_sr.csv")
print("\n=== per-year SR (net, realistic, roll-adjusted) ===")
print(yr.round(2).to_string())

# ── plots ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for n in list(legs) + [EW3]:
    net_pnl(subsets[n], **REAL).dropna().cumsum().plot(ax=ax[0,0], lw=1.6 if n == EW3 else 1, label=n)
ax[0,0].axvline(pd.Timestamp(IS_END), color="k", ls="--", lw=0.8)
ax[0,0].set_title("累積淨損益 (roll-adjusted, realistic cost)"); ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)

im = ax[0,1].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0,1].set_xticks(range(len(corr))); ax[0,1].set_xticklabels(corr.columns, rotation=30, fontsize=8)
ax[0,1].set_yticks(range(len(corr))); ax[0,1].set_yticklabels(corr.index, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax[0,1].text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
ax[0,1].set_title("PnL 相關矩陣"); fig.colorbar(im, ax=ax[0,1], fraction=.046)

yr.plot(kind="bar", ax=ax[1,0], width=.8); ax[1,0].axhline(0, color="k", lw=.8)
ax[1,0].set_title("逐年 Sharpe"); ax[1,0].legend(fontsize=7); ax[1,0].tick_params(labelsize=7)

r = res.set_index("combo").loc[list(legs) + [EW3]]
ax[1,1].bar(range(len(r)), r.SR_of_SR, color=["#888"]*3 + ["#2b7"])
ax[1,1].axhline(0.6, color="r", ls="--", lw=1.2, label="gate 0.6")
ax[1,1].set_xticks(range(len(r))); ax[1,1].set_xticklabels(r.index, rotation=20, fontsize=8)
ax[1,1].set_title("SR_of_SR vs house gate"); ax[1,1].legend(fontsize=8); ax[1,1].grid(alpha=.3, axis="y")
fig.suptitle("QNT-16 igrea + epu_global + kr_kospi 等權組合", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/combo_dashboard.png", dpi=110)
print(f"\nwrote {OUT}/combo_dashboard.png")
