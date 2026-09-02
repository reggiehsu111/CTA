"""QNT-104 step 2+3 — diagnostics, then the gate sweep, on CALL and COMBINED OI.

Pre-run power target (QNT-78 Rule 2, QNT-94 correction), stated before the run:
  target d = 0.12 SR, sd = 0.13
  S_required = 2.33 * (2.80 * 0.13 / 0.12)^2 = 22 source series (corrected)
  S available = 26 NEW features (11 call-side + 15 combined)
  -> d_min = 4.26 * 0.13 / sqrt(26) = 0.109. Adequate for d >= 0.12, and for
     nothing smaller. Raw S is optimistic: call/put features of the same OI book
     correlate much harder than macro series do, so the realised n_eff is
     measured from the per-feature PnLs at the end and reported, not assumed.

OUT-OF-SAMPLE DESIGN (what the ticket asks for). Sign is frozen on IS and never
re-chosen. Two regimes, because `ongap`/`night` need `night_close`, which only
exists from 2017-05-16 and therefore has NO overlap with the day-regime IS half:

  full  regime (c2c/o2o/day)              IS 2009-01..2016-12 | OOS1 2017..2021 | OOS2 2022..2026
  night regime (+ ongap/night, 2017-05..) IS 2017-05..2022-12 | OOS  2023..2026

OOS2 in the full regime is a SECOND held-out block: it is scored once, at the
end, with the sign and the cell choice already fixed by IS. A cell that works on
IS and OOS1 but not OOS2 is not a signal.
"""
import os, sys, warnings, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))  # noqa: S102

import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops
from scipy import stats

OUT = "/home/ubuntu/mtx/signal_zoo/qnt104"
TI  = A.index
_RET["night"]   = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"]  = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1

CALL_F = ["call_oi_total", "call_cog", "call_disp", "call_otm_share", "call_far_share",
          "call_wall", "call_front_share", "call_churn", "call_oi_growth",
          "call_cog_chg", "call_far_chg"]
COMB_F = ["pcr_oi", "pcr_vol", "pcr_oi_chg", "pcr_far", "pcr_atm", "cog_gap", "cog_mid",
          "far_asym", "wall_gap", "wall_mid", "churn_ratio", "oi_growth_diff",
          "disp_ratio", "front_diff", "max_pain"]
FEATS  = CALL_F + COMB_F
FAMILY = {**{f: "call" for f in CALL_F}, **{f: "comb" for f in COMB_F}}

TRANSFORMS = {"selfz": ops.selfz, "robustz": ops.robust_z, "bdtanh": ops.bd_selftanh,
              "rankc": ops.rank_c, "signth": ops.sign_thresh, "dev": ops.dev}
WINDOWS = (20, 60, 120, 252)

REGIMES = [
    # name,   start,        IS end,       OOS1,                     OOS2,                     variants
    ("full",  None,        "2016-12-31", ("2017-01-01","2021-12-31"), ("2022-01-01", None), ("c2c","o2o","day")),
    ("night", "2017-05-16","2022-12-31", ("2023-01-01", None),        (None, None),
     ("c2c","o2o","day","ongap","night")),
]

panels = {}
for ef in ("monthly", "all"):
    c = pd.read_csv(f"{OUT}/call_features_{ef}.csv", index_col=0, parse_dates=True)
    m = pd.read_csv(f"{OUT}/comb_features_{ef}.csv", index_col=0, parse_dates=True)
    panels[ef] = c.join(m).reindex(TI)

# ── Step 2: is it a price MIRROR, and is it era-STABLE? ────────────────────
r_c2c = _RET["c2c"].reindex(TI).astype(float)
fwd   = r_c2c.shift(-2)                     # earliest tradable c2c leg after an EOD print
rows = []
for ef, p in panels.items():
    for f in FEATS:
        x = p[f].astype(float)
        blocks = {}
        for lo, hi in [(2009, 2013), (2014, 2018), (2019, 2022), (2023, 2026)]:
            m_ = (x.index.year >= lo) & (x.index.year <= hi)
            j = pd.concat([x[m_], fwd[m_]], axis=1).dropna()
            blocks[f"IC_{lo}_{hi}"] = (stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0]
                                       if len(j) > 200 else np.nan)
        ics = [v for v in blocks.values() if v == v]
        # non-stationarity: corr of the feature with log spot, per half
        ls = np.log(A["close"].astype(float).reindex(TI))
        h1 = (x.index.year <= 2017); h2 = ~h1
        rows.append(dict(panel=ef, family=FAMILY[f], feature=f, n=int(x.notna().sum()),
                         corr_same_day=x.corr(r_c2c), **blocks,
                         IC_sign_agree=(float(np.mean(np.sign(ics) == np.sign(ics[0])))
                                        if ics else np.nan),
                         corr_spot_h1=x[h1].corr(ls[h1]), corr_spot_h2=x[h2].corr(ls[h2])))
qa = pd.DataFrame(rows)
qa.to_csv(f"{OUT}/oi_diagnostics.csv", index=False)
print("=== QNT-104 step 2 diagnostics (call + combined OI features) ===")
print("corr_same_day: |r| large => the 'feature' is a mirror of TODAY's return, not positioning")
print("corr_spot_h1/h2: relation to price level per half; a SIGN FLIP is what killed put OI total")
print(qa.round(3).to_string(index=False))

MIRROR = set(qa.loc[qa.corr_same_day.abs() > 0.35, "feature"])
print(f"\nflagged as price mirrors (|corr_same_day| > 0.35): {sorted(MIRROR) or 'none'}")

# ── Step 3: the gate sweep ─────────────────────────────────────────────────
rows, pnl = [], {}
for reg, r_start, is_end, (o1s, o1e), (o2s, o2e), variants in REGIMES:
    for ef, p in panels.items():
        for f in FEATS:
            x = p[f].astype(float)
            xs = x.loc[r_start:] if r_start else x
            if xs.notna().sum() < 600:
                continue
            for tn, tf in TRANSFORMS.items():
                for w in WINDOWS:
                    sig = pd.Series(tf(x, w), index=TI).replace([np.inf, -np.inf], np.nan)
                    if (sig.loc[r_start:] if r_start else sig).notna().sum() < 400:
                        continue
                    for v in variants:
                        is_ = wstats(sig, v, start=r_start, end=is_end)
                        if is_ is None: continue
                        sg = is_["sign"]
                        o1 = wstats(sig, v, start=o1s, end=o1e, sign=sg) if o1s else None
                        o2 = wstats(sig, v, start=o2s, end=o2e, sign=sg) if o2s else None
                        full = wstats(sig, v, start=r_start, sign=sg)
                        if full is None: continue
                        cell = f"{reg}|{ef}|{f}|{tn}|w{w}|{v}"
                        rows.append(dict(
                            regime=reg, panel=ef, family=FAMILY[f], feature=f, transform=tn,
                            window=w, variant=v, cell=cell, sign=sg, mirror=f in MIRROR,
                            SR_IS=is_["SR_net"], n_IS=is_["n_bars"],
                            SR_OOS1=(o1 or {}).get("SR_net", np.nan),
                            SR_OOS2=(o2 or {}).get("SR_net", np.nan),
                            SRSR_OOS1=(o1 or {}).get("SR_of_SR", np.nan),
                            posyr_OOS1=(o1 or {}).get("positive_years", np.nan),
                            **{f"full_{a}": full[a] for a in
                               ("SR_net", "SR_of_SR", "positive_years", "yr_sr_min", "n_years",
                                "beta", "mean_exec_w", "abs_exec_w", "turnover_ann",
                                "held_pct", "max_dd_days", "n_bars")}))
                        pos = sig.shift(_SHIFT[v]) * sg
                        g = (pos * _RET[v] - pos.fillna(0).diff().abs() * _COST[v])
                        pnl[cell] = g.astype("float32")

sw = pd.DataFrame(rows)
sw = cta.house_gates(sw, prefix="full_", beta_mode="both")
sw.to_csv(f"{OUT}/oi_sweep_full.csv", index=False)
pd.DataFrame(pnl).to_pickle(f"{OUT}/oi_pnl.pkl")
print(f"\n=== step 3 sweep: {len(sw)} cells, {sw.feature.nunique()} features, "
      f"{sw.regime.nunique()} regimes ===")
print(sw.groupby("regime").size().to_string())
print("\ngate pass counts (QNT-100 beta_per_w rule):")
print(sw.groupby(["regime", "n_gates"]).size().unstack(fill_value=0).to_string())
g = sw[sw.passes] if "passes" in sw else sw[sw.n_gates == 4]
g.to_csv(f"{OUT}/oi_sweep_gated.csv", index=False)
print(f"\nfour-gate passers: {len(g)}  (raw-beta rule would give "
      f"{int((sw.gate_srsr & sw.gate_posyr & sw.gate_nyr & sw.gate_beta_raw).sum())})")
if len(g):
    print(g.sort_values("full_SR_net", ascending=False).head(25)[
        ["cell", "sign", "mirror", "SR_IS", "SR_OOS1", "SR_OOS2", "full_SR_net",
         "full_SR_of_SR", "full_positive_years", "full_beta", "beta_per_w",
         "full_abs_exec_w", "full_n_years"]].to_string(index=False))
