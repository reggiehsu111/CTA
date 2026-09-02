"""QNT-21: quantify the roll defect in cta/signals/_base.py's variant return
legs, and validate the proposed fix — WITHOUT touching _base.py.

The runner writes mtx_signal_values.pnl_1d through compute_variant_pnl, which
uses Variant.return_of.  Three of the six legs cross a contract boundary and
book the calendar spread as P&L:

    c2c        close[t] / close[t-1]              roll on t
    o2o        open[t]  / open[t-1]               roll on t
    noonpause  night_open[t+1] / close[t]         roll on t+1
               (night_open[t+1] is the 15:00 print of day t, but on the
                front contract of day t+1 — see the loader's (date,expiry)
                merge — while close[t] is 13:45 of t on front(t))

and three do not, because both prices come from the SAME row and therefore the
SAME contract:

    day        close[t] / open[t]
    ongap      open[t]  / night_close[t]
    night      night_close[t+1] / night_open[t+1]

This script proves the classification empirically, measures the size of the
contamination, and re-scores every signal under raw vs fixed legs.

Strictly read-only.  No writes to any table, no edit to _base.py.
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

pd.set_option("display.width", 250, "display.max_columns", 40)
OUT = "/home/ubuntu/mtx/signal_zoo/roll_rescore"


def q(sql):
    with engine.connect() as c:
        return pd.DataFrame(c.execute(text(sql)).mappings().all())


# ── preamble assertions (standing brief §1) ────────────────────────────────
A = cta.load_asset("mtx", "1d")
assert A.index.max() >= pd.Timestamp("2026-08-31"), "stale asset - do not proceed"
assert A["volume"].iloc[-1] == A["volume"].iloc[-1], "NaN volume = night-table bug"
assert A["open"].iloc[-1] != A["night_open"].iloc[-1], "day==night = corrupted row"
PPY = int(A.periods_per_year)
print(f"asset ok: {len(A)} rows, last {A.index.max().date()}, "
      f"last volume {A['volume'].iloc[-1]:,.0f}, ppy {PPY}")

_o, _c = A["open"].astype(float), A["close"].astype(float)
_no, _nc = A["night_open"].astype(float), A["night_close"].astype(float)
_bc = A["back_close"].astype(float)
roll = A.is_rollover

# ── back_OPEN, pulled straight from RDS (not present in the asset) ─────────
bo = q("""
  select date, expiry_month, open, close from tw_index_futures_pv
  where ticker='MTX' order by date, expiry_month
""")
bo["date"] = pd.to_datetime(bo["date"])
bo["_ei"] = pd.to_numeric(bo["expiry_month"], errors="coerce")
bo = bo.sort_values(["date", "_ei"])
bo["_rank"] = bo.groupby("date").cumcount()
back = (bo[bo["_rank"] == 1][["date", "expiry_month", "open", "close"]]
        .rename(columns={"expiry_month": "back_expiry_chk",
                         "open": "back_open", "close": "back_close_chk"})
        .set_index("date"))
back_open  = pd.to_numeric(back["back_open"], errors="coerce").reindex(A.index)
back_close_chk = pd.to_numeric(back["back_close_chk"], errors="coerce").reindex(A.index)
back_expiry = back["back_expiry_chk"].reindex(A.index)
print(f"back_open pulled from RDS: {back_open.notna().sum()}/{len(A)} non-null; "
      f"back_close matches asset on {np.isclose(back_close_chk, _bc, equal_nan=True).mean():.4f} of rows")

# ── does tomorrow's front equal today's back on roll days? ─────────────────
fe = A["front_expiry"].astype(str)
chk = pd.DataFrame({"front_next": fe.shift(-1), "back_today": back_expiry.astype(str),
                    "roll_next": roll.shift(-1).fillna(False).astype(bool)})
m = chk["roll_next"] & chk["back_today"].ne("nan")
print(f"roll-eve days where front_expiry[t+1] == back_expiry[t]: "
      f"{(chk.loc[m,'front_next'] == chk.loc[m,'back_today']).mean():.4f}  (n={int(m.sum())})")
m2 = roll & back_expiry.shift(1).astype(str).ne("nan")
print(f"roll days     where front_expiry[t]   == back_expiry[t-1]: "
      f"{(fe[m2] == back_expiry.shift(1).astype(str)[m2]).mean():.4f}  (n={int(m2.sum())})")

# ── the six legs: current (raw) and proposed (fixed) ───────────────────────
RAW = {k: v.return_of(A) for k, v in VARIANT_REGISTRY.items()}

prev_o_approx = _o.shift(1).where(~roll, (_o * _bc / _c).shift(1))   # close-measured spread
prev_o_exact  = _o.shift(1).where(~roll, back_open.shift(1))          # true back-month open
noon_den      = _c.where(~roll.shift(-1).fillna(False).astype(bool), _bc)          # tomorrow's front, today

FIX = {
    "c2c":       A.returns,                       # continuous_prev_close
    "o2o":       _o / prev_o_exact - 1,
    "day":       RAW["day"],                      # intra-contract
    "night":     RAW["night"],                    # intra-contract
    "ongap":     RAW["ongap"],                    # intra-contract
    "noonpause": _no.shift(-1) / noon_den - 1,
}
O2O_APPROX = _o / prev_o_approx - 1

# ── (1) contamination diagnostics ─────────────────────────────────────────
print("\n" + "=" * 118)
print("(1) Which legs cross a contract boundary?  mean |return| on affected days vs all other days")
print("=" * 118)
AFFECT = {"c2c": roll, "o2o": roll, "noonpause": roll.shift(-1).fillna(False).astype(bool),
          "day": roll, "ongap": roll, "night": roll.shift(-1).fillna(False).astype(bool)}
rows = []
for k in ["c2c", "o2o", "noonpause", "day", "ongap", "night"]:
    r, f, a = RAW[k], FIX[k], AFFECT[k]
    both = r.notna() & f.notna()
    aa, oo = a & both, (~a) & both
    d = (r - f).abs()
    rows.append({
        "variant": k, "n_affected": int(aa.sum()),
        "mean|ret| affected": round(float(r[aa].abs().mean()) * 1e4, 1),
        "mean|ret| other": round(float(r[oo].abs().mean()) * 1e4, 1),
        "ratio": round(float(r[aa].abs().mean() / r[oo].abs().mean()), 3),
        "mean|raw-fix| (bps)": round(float(d[aa].mean()) * 1e4, 1) if aa.sum() else 0.0,
        "max|raw-fix| (bps)": round(float(d[aa].max()) * 1e4, 1) if aa.sum() else 0.0,
        "ann.vol raw": round(float(r.std() * np.sqrt(PPY)), 4),
        "ann.vol fix": round(float(f.std() * np.sqrt(PPY)), 4),
    })
DIAG = pd.DataFrame(rows).set_index("variant")
print(DIAG.to_string())
print("\n(units: bps.  'affected' = rollover day for c2c/o2o/day/ongap, rollover EVE for noonpause/night)")

# ── (2) o2o: is the close-measured approximation good enough? ─────────────
print("\n" + "=" * 118)
print("(2) o2o — exact back_open adjustment vs the close-measured spread approximation")
print("=" * 118)
ex, ap = FIX["o2o"], O2O_APPROX
m = roll & ex.notna() & ap.notna()
print(f"roll days compared           : {int(m.sum())}")
print(f"mean |exact - approx| (bps)  : {float((ex-ap).abs()[m].mean())*1e4:.2f}")
print(f"max  |exact - approx| (bps)  : {float((ex-ap).abs()[m].max())*1e4:.2f}")
print(f"corr on roll days            : {ex[m].corr(ap[m]):.6f}")
print(f"mean |raw - exact| (bps)     : {float((RAW['o2o']-ex).abs()[m].mean())*1e4:.2f}   "
      f"-> the approximation removes {1 - float((ex-ap).abs()[m].mean())/float((RAW['o2o']-ex).abs()[m].mean()):.1%} of the error")

# ── (3) buy-and-hold reference per leg ────────────────────────────────────
def sr(x):
    x = pd.Series(x).dropna()
    return float(np.sqrt(PPY) * x.mean() / x.std()) if len(x) > 30 and x.std() > 0 else np.nan

print("\n" + "=" * 118)
print("(3) Buy-and-hold SR per leg (gross), raw vs fixed — full history and 2019+")
print("=" * 118)
bh = []
for k in ["c2c", "o2o", "noonpause", "day", "ongap", "night"]:
    for lab, st in [("full", None), ("2019+", "2019-01-01")]:
        r = RAW[k].loc[st:] if st else RAW[k]
        f = FIX[k].loc[st:] if st else FIX[k]
        bh.append({"variant": k, "window": lab, "BH_SR_raw": round(sr(r), 3),
                   "BH_SR_fix": round(sr(f), 3), "d": round(sr(f) - sr(r), 3)})
print(pd.DataFrame(bh).set_index(["variant", "window"]).to_string())

# ── (4) re-score every signal on the runner's OWN stored positions ────────
cfg = q("select signal_name, enabled, recommended_variants from mtx_signal_config").set_index("signal_name")
vals = q("select date, signal_name, variant, position, pnl_1d from mtx_signal_values")
vals["date"] = pd.to_datetime(vals["date"])

def yr(x):
    x = x.dropna()
    ys = x.groupby(x.index.year).apply(
        lambda s: np.sqrt(PPY)*s.mean()/s.std() if len(s) > 20 and s.std() > 0 else np.nan).dropna()
    if len(ys) < 2:
        return np.nan, np.nan
    return (float(ys.mean()/ys.std()) if ys.std() > 0 else np.nan), float((ys > 0).mean())

print("\n" + "=" * 130)
print("(4) Live-page PnL re-scored on the FIXED legs — runner's own stored positions, live stub cost")
print("=" * 130)
res = []
for name in sorted(cfg.index):
    live_vars = list(cfg.loc[name, "recommended_variants"] or [])
    for k in ["c2c", "o2o", "noonpause", "day", "ongap", "night"]:
        v = vals[(vals.signal_name == name) & (vals.variant == k)].set_index("date").sort_index()
        if v.empty:
            continue
        pos = v["position"]
        cost = VARIANT_REGISTRY[k].cost_of(A).reindex(pos.index)
        tc = pos.fillna(0).diff().abs() * cost
        stored = v["pnl_1d"]
        p_raw = pos * RAW[k].reindex(pos.index) - tc
        p_fix = pos * FIX[k].reindex(pos.index) - tc
        j = pd.concat([stored, p_raw, p_fix], axis=1).dropna()
        for lab, st in [("full", None), ("2019+", "2019-01-01")]:
            jj = j.loc[st:] if st else j
            if len(jj) < 60:
                continue
            s_r, s_f = sr(jj.iloc[:, 1]), sr(jj.iloc[:, 2])
            sos_r, py_r = yr(jj.iloc[:, 1])
            sos_f, py_f = yr(jj.iloc[:, 2])
            res.append({
                "signal": name, "variant": k, "window": lab,
                "live": bool(cfg.loc[name, "enabled"]) and k in live_vars,
                "max|stored-raw|": f"{float((jj.iloc[:,0]-jj.iloc[:,1]).abs().max()):.2e}",
                "max|stored-fix|": f"{float((jj.iloc[:,0]-jj.iloc[:,2]).abs().max()):.2e}",
                "SR_raw": round(s_r, 3), "SR_fix": round(s_f, 3), "dSR": round(s_f - s_r, 3),
                "SoS_raw": round(sos_r, 3), "SoS_fix": round(sos_f, 3),
                "posyr_raw": round(py_r, 3), "posyr_fix": round(py_f, 3),
            })
R = pd.DataFrame(res)
R.to_csv(f"{OUT}/base_ret_fix.csv", index=False)

live = R[R.live]
print("\n--- THE 9 ENABLED SIGNALS ON THEIR LIVE VARIANT ---")
for lab in ("full", "2019+"):
    print(f"\n[{lab}]")
    print(live[live.window == lab].drop(columns=["live", "window"])
          .set_index("signal").sort_values("SR_raw", ascending=False).to_string())

print("\n--- ALL 11 SIGNALS, THE 3 CONTAMINATED LEGS (full history) ---")
print(R[(R.window == "full") & R.variant.isin(["c2c", "o2o", "noonpause"])]
      .drop(columns=["window"]).set_index(["variant", "signal"]).sort_index().to_string())

# ── (5) how many stored rows would a backfill rewrite? ────────────────────
print("\n" + "=" * 118)
print("(5) Scale of a history backfill")
print("=" * 118)
tot = q("select variant, count(*) n, min(date) d0, max(date) d1 from mtx_signal_values group by variant order by variant")
print(tot.to_string(index=False))
changed = []
for k in ["c2c", "o2o", "noonpause"]:
    v = vals[vals.variant == k]
    d = (RAW[k] - FIX[k]).abs()
    aff = d[d > 0].index
    n = int(v[v.date.isin(aff)].shape[0])
    changed.append({"variant": k, "rows_stored": int(len(v)), "rows_that_would_change": n,
                    "pct": round(100*n/len(v), 2)})
print()
print(pd.DataFrame(changed).to_string(index=False))
print(f"\nwrote {OUT}/base_ret_fix.csv")
