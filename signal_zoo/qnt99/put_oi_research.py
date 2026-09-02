"""QNT-99 Part B step 3 — ask the questions, then sweep.

Q3  Is the feature a MIRROR of price rather than a read on positioning?
    corr(feature_t, same-day return_t). A feature whose moneyness is measured
    against today's spot moves mechanically when spot moves; if that
    contemporaneous correlation is large the "signal" is just -1 x today's return.
Q4  Is the relation STABLE across eras? Spearman IC vs next-day c2c return, per
    5-year block. opt_put_mo_oi_selftanh_w60 died because its sign flipped
    between halves; any replacement must be checked the same way BEFORE it is
    scored.
Q5  Only then: the gate sweep, sign frozen on the IS half.
"""
import sys, os, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))  # noqa: S102

import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops
from scipy import stats

OUT = "/home/ubuntu/mtx/signal_zoo/qnt99"
IS_END, OOS_START = "2016-12-31", "2017-01-01"   # options history starts 2009 here
TI = A.index
_RET["night"] = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"] = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1

FEATS = ["put_oi_total", "put_cog", "put_disp", "put_otm_share", "put_far_share",
         "put_wall", "put_front_share", "put_churn", "put_oi_growth",
         "put_cog_chg", "put_far_chg"]
TRANSFORMS = {"selfz": lambda x, w: ops.selfz(x, w),
              "robustz": lambda x, w: ops.robust_z(x, w),
              "bdtanh": lambda x, w: ops.bd_selftanh(x, w),
              "rankc": lambda x, w: ops.rank_c(x, w),
              "signth": lambda x, w: ops.sign_thresh(x, w),
              "dev": lambda x, w: ops.dev(x, w)}
WINDOWS = (20, 60, 120, 252)
VARIANTS = ("c2c", "o2o", "day", "ongap", "night")

panels = {ef: pd.read_csv(f"{OUT}/put_oi_features_{ef}.csv", index_col=0, parse_dates=True)
          for ef in ("monthly", "all")}

# ── Q3 / Q4 ────────────────────────────────────────────────────────────────
r_c2c = _RET["c2c"].reindex(TI).astype(float)
rows = []
for ef, p in panels.items():
    p = p.reindex(TI)
    for f in FEATS:
        x = p[f].astype(float)
        same = x.corr(r_c2c)
        # feature is public after the 13:45 close of t -> earliest c2c leg is t+2
        fwd = r_c2c.shift(-2)
        blocks = {}
        for lo, hi in [(2009, 2013), (2014, 2018), (2019, 2022), (2023, 2026)]:
            m = (x.index.year >= lo) & (x.index.year <= hi)
            j = pd.concat([x[m], fwd[m]], axis=1).dropna()
            blocks[f"IC_{lo}_{hi}"] = (stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0]
                                       if len(j) > 200 else np.nan)
        ics = [v for v in blocks.values() if v == v]
        rows.append(dict(panel=ef, feature=f, n=int(x.notna().sum()),
                         corr_same_day=same, **blocks,
                         IC_sign_agree=(float(np.mean(np.sign(ics) == np.sign(ics[0])))
                                        if ics else np.nan)))
qa = pd.DataFrame(rows)
qa.to_csv(f"{OUT}/put_oi_diagnostics.csv", index=False)
print("=== Q3/Q4 put-OI feature diagnostics ===")
print("corr_same_day = mechanical mirror check; IC_* = Spearman vs c2c return at t+2")
print(qa.round(3).to_string(index=False))

# ── Q5 gate sweep ──────────────────────────────────────────────────────────
rows, pnl = [], {}
for ef, p in panels.items():
    p = p.reindex(TI)
    for f in FEATS:
        x = p[f].astype(float)
        if x.notna().sum() < 1000:
            continue
        for tn, tf in TRANSFORMS.items():
            for w in WINDOWS:
                sig = pd.Series(tf(x, w), index=TI).replace([np.inf, -np.inf], np.nan)
                if sig.notna().sum() < 500:
                    continue
                for v in VARIANTS:
                    is_ = wstats(sig, v, end=IS_END)
                    if is_ is None: continue
                    oos = wstats(sig, v, start=OOS_START, sign=is_["sign"])
                    full = wstats(sig, v, sign=is_["sign"])
                    if full is None: continue
                    cell = f"{ef}|{f}|{tn}|w{w}|{v}"
                    rows.append(dict(panel=ef, feature=f, transform=tn, window=w,
                                     variant=v, cell=cell, sign=is_["sign"],
                                     SR_IS=is_["SR_net"],
                                     SR_OOS=(oos or {}).get("SR_net", np.nan),
                                     **{f"full_{a}": full[a] for a in
                                        ("SR_net","SR_of_SR","positive_years","yr_sr_min",
                                         "n_years","beta","mean_exec_w","abs_exec_w",
                                         "turnover_ann","held_pct","max_dd_days","n_bars")}))
                    pos = sig.shift(_SHIFT[v]) * is_["sign"]
                    pnl[cell] = pos * _RET[v] - pos.fillna(0).diff().abs() * _COST[v]
sw = pd.DataFrame(rows)
sw.to_csv(f"{OUT}/put_oi_sweep_full.csv", index=False)
pd.DataFrame(pnl).to_pickle(f"{OUT}/put_oi_pnl.pkl")
g = sw[(sw.full_SR_of_SR > 0.6) & (sw.full_positive_years >= 0.65)
       & (sw.full_beta.abs() < 0.15) & (sw.full_n_years >= 5)]
g.to_csv(f"{OUT}/put_oi_sweep_gated.csv", index=False)
print(f"\n=== Q5 put-OI sweep: {len(sw)} cells ({sw.feature.nunique()} features x "
      f"{sw.panel.nunique()} panels), {len(g)} pass all 4 gates ===")
if len(g):
    print(g.sort_values("full_SR_net", ascending=False).head(20)[
        ["cell","sign","SR_IS","SR_OOS","full_SR_net","full_SR_of_SR","full_positive_years",
         "full_beta","full_n_years","full_abs_exec_w","full_turnover_ann"]].to_string(index=False))
h = sw.rename(columns={"full_SR_net":"SR_net","full_n_years":"n_years"})
cta.sweep_headline(h, value="SR_net", series_col="feature", label="QNT-99 put-OI").print()
cta.sweep_headline(h.dropna(subset=["SR_OOS"]), value="SR_OOS", series_col="feature",
                   label="QNT-99 put-OI OOS").print()
d = sw.dropna(subset=["SR_IS","SR_OOS"])
print(f"IS->OOS (split {IS_END}): med IS {d.SR_IS.median():+.3f}  med OOS {d.SR_OOS.median():+.3f}  "
      f"frac OOS>0 {(d.SR_OOS>0).mean():.3f}  corr {d.SR_IS.corr(d.SR_OOS):+.3f}")
print("\nper-feature median SR_net / IS / OOS:")
print(sw.groupby("feature")[["full_SR_net","SR_IS","SR_OOS"]].median().round(3).to_string())
