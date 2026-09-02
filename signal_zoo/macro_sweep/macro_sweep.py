"""QNT-12: macro (QNT-10 tidy layer) → MTX factor sweep.

Discipline, per the 台指期 standing brief:
  * every input is PIT-aligned via `load_macro_tw` (publication-lag shifted),
    never `load_macro`
  * scored on ROLL-ADJUSTED returns (`roll_adjusted=True`) — the default
    `close.pct_change()` books the calendar spread as P&L
  * the SIGN IS FROZEN ON THE IN-SAMPLE HALF and carried into OOS unchanged.
    An agent that lets auto_flip pick a sign on the full sample will always
    find one. Nothing here selects a sign for live use — that is Reggie's call.
  * the FULL sweep is written out, not the best cell

QNT-92 — READ BEFORE RE-RUNNING THIS SCRIPT
    `full_sweep.csv` / `gated_sweep.csv` in OUT are PRE-QNT-19-floor and are
    deliberately kept that way: they are the "before" side that
    `signal_zoo/qnt19_postfloor/compare_prefloor.py` diffs against. Running this
    script overwrites them and turns that comparison into a no-op, deleting
    QNT-19's evidence. They are also the published QNT-12/QNT-25 baseline.
    The post-floor grid already exists at `qnt19_postfloor/full_sweep.csv`, and
    QNT-92 verified current code reproduces it (SR_IS bit-identical on 522/522).
    If you need a fresh grid, redirect OUT to a scratch dir as QNT-19 and QNT-92
    both did. See `signal_zoo/qnt92_scratch/QNT92_NOTE.md`.
"""
import sys, warnings, itertools, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context

OUT = "/home/ubuntu/mtx/signal_zoo/macro_sweep"
IS_END, OOS_START = "2018-12-31", "2019-01-01"

# Cost models. STUB is what the framework defaults to and what the brief says
# understates; REAL adds 期交稅 (a further 2e-5/side) and ~1 index point of
# slippage (1 pt x 50 TWD/pt = 50 TWD, folded into fixed_per_side).
STUB = dict(fixed_per_side=20.0, fee_rate=0.00002)
REAL = dict(fixed_per_side=70.0, fee_rate=0.00004)

ctx = build_context()
ASSET = ctx.asset

# ── Inputs: economic rationale first, then transforms. ─────────────────────
# Each entry: series_id -> ('level'|'yoy', periods) and a one-line thesis.
FAMILIES = {
    "semis": [("us_semi_ip", "yoy", 12), ("us_semi_ip_nsa", "yoy", 12),
              ("us_semi_ppi", "yoy", 12), ("us_electronics_ppi", "yoy", 12)],
    "kr_tech_demand": [("kr_exports_sa", "yoy", 12), ("kr_exports", "yoy", 12),
                       ("kr_kospi", "yoy", 12)],
    "cn_cycle": [("cn_leading_idx", "level", 0), ("cn_exports", "yoy", 12),
                 ("cn_shanghai_comp", "yoy", 12)],
    "us_cycle": [("us_cfnai", "level", 0), ("us_mfg_new_orders", "yoy", 12),
                 ("us_freight_tsi", "yoy", 12), ("us_retail_inv_sales", "level", 0),
                 ("us_recession_prob", "level", 0)],
    "us_survey": [("us_empire_state", "level", 0), ("us_philly_fed", "level", 0)],
    "rates": [("us_term_premium_10y", "level", 0), ("us_real_10y", "level", 0),
              ("us_breakeven_5y5y", "level", 0), ("us_dgs5", "level", 0)],
    "risk": [("us_stlfsi", "level", 0), ("epu_global", "level", 0)],
    "fx": [("twd_usd", "level", 0), ("krw_usd", "level", 0), ("us_dxy_broad", "level", 0)],
    "commodity": [("copper", "yoy", 12), ("wti", "level", 0), ("igrea", "level", 0)],
}

TRANSFORMS = {
    "selfz":     lambda x, w: ops.selfz(x, w),
    "robustz":   lambda x, w: ops.robust_z(x, w),
    "bdtanh":    lambda x, w: ops.bd_selftanh(x, w),
    "rankc":     lambda x, w: ops.rank_c(x, w),
    "signth":    lambda x, w: ops.sign_thresh(x, w),
    "dev":       lambda x, w: ops.dev(x, w),
}
WINDOWS = (60, 120, 252)

# ── Build the raw inputs ───────────────────────────────────────────────────
raw_inputs, meta = {}, {}
cat = cta.macro_catalog()
for fam, entries in FAMILIES.items():
    for sid, kind, per in entries:
        try:
            if kind == "yoy":
                freq = cat.loc[sid, "freq"]
                per = {"M": 12, "Q": 4, "D": 252, "W": 52}.get(freq, 12)
                x = ctx.macro_yoy(sid, per)
            else:
                x = ctx.macro(sid)
        except Exception as e:
            print(f"  skip {sid}: {type(e).__name__}: {e}")
            continue
        key = f"{sid}_{kind}"
        raw_inputs[key] = x.astype(float)
        meta[key] = dict(family=fam, series=sid, kind=kind, periods=per,
                         pub_lag_days=int(cat.loc[sid, "pub_lag_days"]),
                         first_obs=str(cat.loc[sid, "first_obs"]))
print(f"inputs: {len(raw_inputs)}")

# ── Sweep ──────────────────────────────────────────────────────────────────
def frozen_sign_stats(sig, sign, **cost):
    """Stats with the sign supplied, NOT rediscovered."""
    return cta.signal_stats(sig * sign, ASSET, auto_flip=False,
                            roll_adjusted=True, **cost)

rows = []
for key, x in raw_inputs.items():
    for tname, tf in TRANSFORMS.items():
        for w in WINDOWS:
            try:
                sig = tf(x, w).replace([np.inf, -np.inf], np.nan)
                sig = cta.normalize_signal(sig, method="tanh", window=252)
                if sig.dropna().empty or sig.loc[:IS_END].dropna().shape[0] < 500:
                    continue
                # 1. sign frozen on IS only
                is_st = cta.signal_stats(sig, ASSET, end=IS_END, auto_flip=True,
                                         roll_adjusted=True, **REAL)
                sign = int(is_st["sign"])
                # 2. OOS with that sign, no rediscovery
                oos = frozen_sign_stats(sig, sign, start=OOS_START, **REAL)
                # 3. full sample with the IS sign — this is what the gates see
                full = frozen_sign_stats(sig, sign, **REAL)
                full_stub = frozen_sign_stats(sig, sign, **STUB)
                rows.append({
                    "cand": f"{key}|{tname}|w{w}", **meta[key],
                    "transform": tname, "window": w, "sign_IS": sign,
                    "SR_IS": is_st["SR_net"] * sign if False else is_st["SR_net"],
                    "SR_OOS": oos["SR_net"], "SR_full": full["SR_net"],
                    "SR_full_gross": full["SR_gross"],
                    "SR_full_stubcost": full_stub["SR_net"],
                    "SR_of_SR": full["SR_of_SR"], "positive_years": full["positive_years"],
                    "n_years": full["n_years"], "beta": full["beta"],
                    "mean_pos": float((sig * sign).shift(2).reindex(ASSET.index).mean()),
                    "max_dd_days": full["max_dd_days"], "max_dd_pct": full["max_dd_pct"],
                    "turnover_ann": full["turnover_ann"], "held_pct": full["held_pct"],
                    "tcost_pct_of_gross": full["tcost_pct_of_gross"],
                })
            except Exception as e:
                rows.append({"cand": f"{key}|{tname}|w{w}", **meta[key],
                             "transform": tname, "window": w,
                             "note": f"{type(e).__name__}: {e}"[:90]})

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/full_sweep.csv", index=False)
print(f"\ncandidates evaluated: {len(df)}  (written to full_sweep.csv)")

ok = df[df["SR_full"].notna()].copy()
ok["gate_srsr"]  = ok["SR_of_SR"] > 0.6
ok["gate_posyr"] = ok["positive_years"] >= 0.65
ok["gate_beta"]  = ok["beta"].abs() < 0.15
ok["gate_nyr"]   = ok["n_years"] >= 5
ok["gate_oos"]   = ok["SR_OOS"] > 0
ok["n_gates"]    = ok[["gate_srsr","gate_posyr","gate_beta","gate_nyr","gate_oos"]].sum(axis=1)
ok = ok.sort_values(["n_gates", "SR_full"], ascending=False)
ok.to_csv(f"{OUT}/gated_sweep.csv", index=False)

pd.set_option("display.width", 260, "display.max_columns", 40)
cols = ["cand","family","SR_IS","SR_OOS","SR_full","SR_full_stubcost","SR_of_SR",
        "positive_years","n_years","beta","mean_pos","turnover_ann","n_gates"]
print("\n=== TOP 25 by gates passed, then SR (realistic costs, roll-adjusted, IS-frozen sign) ===")
print(ok[cols].head(25).round(3).to_string(index=False))
print("\n=== ALL FIVE GATES ===")
allg = ok[ok["n_gates"] == 5]
print(f"{len(allg)} of {len(ok)} candidates" if len(allg) else "NONE of %d candidates" % len(ok))
if len(allg):
    print(allg[cols].round(3).to_string(index=False))

print("\n=== gate pass-rates across the whole sweep ===")
for g in ["gate_srsr","gate_posyr","gate_beta","gate_nyr","gate_oos"]:
    print(f"  {g:12s} {ok[g].mean():.1%}")
print("\n=== family medians (SR_full, realistic cost) ===")
print(ok.groupby("family")[["SR_IS","SR_OOS","SR_full","SR_full_stubcost","beta"]]
        .median().round(3).sort_values("SR_full", ascending=False).to_string())
print("\ncost drag: median SR_full_stubcost - SR_full =",
      round(float((ok.SR_full_stubcost - ok.SR_full).median()), 3))


# ── QNT-32 / QNT-25 reporting line ────────────────────────────────────────
# The top-25 table above ranks 522 cells. QNT-25 measured that the 18
# transform-window cells of one source series are worth ~1.5 independent
# tests, and that sd(SR) across cells is BELOW SE(SR|25y). So the ranking is
# largely a ranking of estimation noise, and the honest n is the number of
# SOURCE SERIES, not the number of cells. Printed last so it is the line that
# ends up in the write-up.
print(f"\n{'='*100}\n=== QNT-25 REPORTING LINE (quote this, not the best cell) ===")
cta.sweep_headline(df, "SR_full",  label="QNT-12 standalone macro sweep").print()
cta.sweep_headline(df, "SR_of_SR", label="QNT-12 standalone macro sweep, SR_of_SR").print()
cta.sweep_headline(df, "SR_OOS",   label="QNT-12 standalone macro sweep, SR_OOS").print()
