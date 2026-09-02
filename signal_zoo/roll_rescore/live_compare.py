"""QNT-13 step 3: compare the live page against the re-scored numbers.

The live page reads mtx_signal_values.pnl_1d, written by the runner via
cta.signals._base.compute_variant_pnl. This checks (a) which return series
that stored PnL is actually built on, and (b) how the live SR compares to
signal_stats under both roll_adjusted settings.

Strictly read-only: SELECT only, no writes to any table.
"""
import sys, pickle, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sqlalchemy import text
from db_utils import engine
import cta
from cta.signals._base import VARIANT_REGISTRY

pd.set_option("display.width", 260, "display.max_columns", 40)

def q(sql):
    with engine.connect() as c:
        return pd.DataFrame(c.execute(text(sql)).mappings().all())

A = cta.load_asset("mtx", "1d")
assert A.index.max() >= pd.Timestamp("2026-08-31")
PPY = int(A.periods_per_year)
D = pickle.load(open("signal_zoo/roll_rescore/signals.pkl", "rb"))
SIGS, META = D["sigs"], D["meta"]

cfg = q("select signal_name, enabled, live_date, recommended_variants from mtx_signal_config").set_index("signal_name")
vals = q("select date, signal_name, variant, position, pnl_1d from mtx_signal_values")
vals["date"] = pd.to_datetime(vals["date"])

roll = A.is_rollover
r_raw, r_adj = A["close"].pct_change(), A.returns

# ── (a) is the STORED c2c PnL on raw or roll-adjusted returns? ─────────────
print("="*118)
print("Which return series is the live (stored) c2c PnL built on?  — checked on the 305 rollover days")
print("="*118)
print(f"{'signal':45s} {'corr(stored,raw-recon)':>23s} {'corr(stored,adj-recon)':>23s} {'max|Δ| vs raw':>14s} {'max|Δ| vs adj':>14s}")
for name in sorted(SIGS):
    v = vals[(vals.signal_name == name) & (vals.variant == "c2c")].set_index("date").sort_index()
    if v.empty: continue
    pos = v["position"]                       # runner's own lagged position
    cost = VARIANT_REGISTRY["c2c"].cost_of(A).reindex(pos.index)
    tc = pos.fillna(0).diff().abs() * cost
    recon_raw = (pos * r_raw.reindex(pos.index) - tc)
    recon_adj = (pos * r_adj.reindex(pos.index) - tc)
    m = roll.reindex(pos.index).fillna(False)
    st = v["pnl_1d"]
    j = pd.concat([st, recon_raw, recon_adj], axis=1).dropna()
    jm = j[m.reindex(j.index).fillna(False)]
    print(f"{name:45s} {jm.iloc[:,0].corr(jm.iloc[:,1]):23.6f} {jm.iloc[:,0].corr(jm.iloc[:,2]):23.6f} "
          f"{(jm.iloc[:,0]-jm.iloc[:,1]).abs().max():14.2e} {(jm.iloc[:,0]-jm.iloc[:,2]).abs().max():14.2e}")

# ── (b) live SR (from stored pnl_1d) vs re-scored SR ───────────────────────
def sr(x):
    x = x.dropna()
    return float(np.sqrt(PPY) * x.mean() / x.std()) if len(x) > 30 and x.std() > 0 else np.nan
def yr_stats(x):
    x = x.dropna()
    ys = x.groupby(x.index.year).apply(lambda s: np.sqrt(PPY)*s.mean()/s.std() if len(s) > 20 and s.std() > 0 else np.nan).dropna()
    if len(ys) < 2: return np.nan, np.nan, 0
    return float(ys.mean()/ys.std()) if ys.std() > 0 else np.nan, float((ys > 0).mean()), len(ys)

REAL = dict(fixed_per_side=70.0, fee_rate=4e-5)
STUB = dict(fixed_per_side=20.0, fee_rate=2e-5)

rows = []
for name in sorted(SIGS):
    if name not in cfg.index or not cfg.loc[name, "enabled"]:
        continue
    live_var = (cfg.loc[name, "recommended_variants"] or ["c2c"])[0]
    for label, st_ in [("full", None), ("2019+", "2019-01-01")]:
        v = vals[(vals.signal_name == name) & (vals.variant == live_var)].set_index("date").sort_index()
        p = v["pnl_1d"]
        if st_: p = p.loc[st_:]
        s_live = sr(p); sos_live, py_live, ny_live = yr_stats(p)
        # roll-adjusted counterpart of the SAME variant, same stub cost, same stored position
        pos = v["position"]
        var = VARIANT_REGISTRY[live_var]
        tc = pos.fillna(0).diff().abs() * var.cost_of(A).reindex(pos.index)
        if live_var in ("c2c",):
            padj = (pos * r_adj.reindex(pos.index) - tc)
        elif live_var == "o2o":
            o = A["open"].astype(float)
            prev_o = o.shift(1)
            back_o = A["back_open"].shift(1) if "back_open" in A.columns else None
            padj = np.nan if back_o is None else (pos * (o/prev_o.where(~roll, back_o) - 1).reindex(pos.index) - tc)
        else:
            padj = None                                  # intra-contract: no roll adjustment applies
        if padj is not None and not isinstance(padj, float):
            if st_: padj = padj.loc[st_:]
            s_adj = sr(padj); sos_adj, py_adj, _ = yr_stats(padj)
        else:
            s_adj = sos_adj = py_adj = np.nan
        # my c2c signal_stats numbers for reference
        c2c_raw = cta.signal_stats(SIGS[name], A, start=st_, auto_flip=False, roll_adjusted=False, **REAL)
        c2c_adj = cta.signal_stats(SIGS[name], A, start=st_, auto_flip=False, roll_adjusted=True,  **REAL)
        rows.append({"signal": name, "window": label, "live_variant": live_var,
                     "SR_live_page": round(s_live, 3), "SR_live_rolladj": round(s_adj, 3) if s_adj == s_adj else np.nan,
                     "dSR_live": round(s_adj - s_live, 3) if s_adj == s_adj else np.nan,
                     "SoS_live": round(sos_live, 3), "SoS_live_rolladj": round(sos_adj, 3) if sos_adj == sos_adj else np.nan,
                     "posyr_live": round(py_live, 3), "posyr_live_rolladj": round(py_adj, 3) if py_adj == py_adj else np.nan,
                     "SR_c2c_real_raw": c2c_raw["SR_net"], "SR_c2c_real_adj": c2c_adj["SR_net"]})

L = pd.DataFrame(rows)
L.to_csv("signal_zoo/roll_rescore/live_compare.csv", index=False)
for label in ("full", "2019+"):
    print(f"\n{'='*130}\nLIVE PAGE vs ROLL-ADJUSTED — {label}  (live variant, live stub cost 20/side+2e-5, runner's own positions)\n{'='*130}")
    print(L[L.window == label].set_index("signal").drop(columns=["window"]).sort_values("SR_live_page", ascending=False).to_string())
print("\nwrote signal_zoo/roll_rescore/live_compare.csv")
