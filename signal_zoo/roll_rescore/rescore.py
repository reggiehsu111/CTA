"""QNT-13 step 2: re-score all 11 registered signals under
`roll_adjusted` False (current default, close.pct_change) and True
(asset.returns, roll-adjusted) — full sample and 2019+, realistic MTX cost.

Signs are the FROZEN class signs (auto_flip=False). The auto-flip sign is
recorded separately as a diagnostic only; this script never changes a sign.
"""
import sys, pickle, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import cta

pd.set_option("display.width", 250, "display.max_columns", 60)

A = cta.load_asset("mtx", "1d")
assert A.index.max() >= pd.Timestamp("2026-08-31")

D = pickle.load(open("signal_zoo/roll_rescore/signals.pkl", "rb"))
SIGS, META = D["sigs"], D["meta"]

REAL = dict(fixed_per_side=70.0, fee_rate=4e-5)     # 20 comm + 2e-5 tax + ~1pt slippage
STUB = dict(fixed_per_side=20.0, fee_rate=2e-5)     # the understating stub, for reference
GATE = ["SR_of_SR", "positive_years", "beta", "n_years"]
COLS = ["SR_net", "SR_gross", "SR_of_SR", "positive_years", "beta", "n_years",
        "yr_sr_min", "max_dd_days", "turnover_ann", "alpha_ann_pct", "n_bars"]

def gates(r):
    """(passes4, dict of per-gate bool) under the house gates."""
    g = {"SR_of_SR>0.6":   bool(r["SR_of_SR"] > 0.6),
         "posyr>=0.65":    bool(r["positive_years"] >= 0.65),
         "|beta|<0.15":    bool(abs(r["beta"]) < 0.15),
         "n_years>=5":     bool(r["n_years"] >= 5)}
    return all(g.values()), g

def score(sig, start=None, cost=REAL):
    out = {}
    for ra in (False, True):
        s = cta.signal_stats(sig, A, start=start, auto_flip=False, roll_adjusted=ra, **cost)
        out[ra] = s
    return out

# ── how much do the two return series actually differ? ──────────────────────
r_raw, r_adj = A["close"].pct_change(), A.returns
d = (r_raw - r_adj).dropna()
nz = d[d.abs() > 1e-12]
print(f"return series differ on {len(nz)} of {len(d)} bars "
      f"(rollover days={int(A.is_rollover.sum())}); "
      f"mean|Δ|={nz.abs().mean()*1e4:.1f}bps  max|Δ|={nz.abs().max()*1e4:.1f}bps")

for label, st in [("full (2001-04-09→2026-08-31)", None), ("2019+", "2019-01-01")]:
    bh = pd.Series(1.0, index=A.index)
    b0 = cta.signal_stats(bh, A, start=st, auto_flip=False, roll_adjusted=False, **REAL)
    b1 = cta.signal_stats(bh, A, start=st, auto_flip=False, roll_adjusted=True,  **REAL)
    print(f"buy-and-hold {label}: SR_net raw={b0['SR_net']:.3f}  adj={b1['SR_net']:.3f}  "
          f"(Δ={b1['SR_net']-b0['SR_net']:+.3f}); beta raw={b0['beta']} adj={b1['beta']}")

rows = []
for name, sig in SIGS.items():
    for label, st in [("full", None), ("2019+", "2019-01-01")]:
        s = score(sig, start=st)
        raw, adj = s[False], s[True]
        p_raw, g_raw = gates(raw)
        p_adj, g_adj = gates(adj)
        # diagnostic only: what sign would the data pick, each way?
        af_raw = cta.signal_stats(sig, A, start=st, auto_flip=True, roll_adjusted=False, **REAL)["sign"]
        af_adj = cta.signal_stats(sig, A, start=st, auto_flip=True, roll_adjusted=True,  **REAL)["sign"]
        row = {"signal": name, "window": label, "enabled": META[name]["enabled"],
               "frozen_sign": META[name]["sign"], "rec_variants": ",".join(META[name]["recommended_variants"]) or "(auto)"}
        for k in COLS:
            row[f"{k}_raw"] = raw[k]; row[f"{k}_adj"] = adj[k]
            row[f"{k}_d"]   = round(adj[k] - raw[k], 4) if isinstance(raw[k], (int, float)) else None
        row["gates_raw"] = sum(g_raw.values()); row["gates_adj"] = sum(g_adj.values())
        row["pass4_raw"] = p_raw; row["pass4_adj"] = p_adj
        row["fails_adj"] = ",".join(k for k, v in g_adj.items() if not v) or "-"
        row["fails_raw"] = ",".join(k for k, v in g_raw.items() if not v) or "-"
        row["gate_status_changed"] = (g_raw != g_adj)
        row["gate_flips"] = ",".join(f"{k}:{g_raw[k]}→{g_adj[k]}" for k in g_raw if g_raw[k] != g_adj[k]) or "-"
        row["autoflip_sign_raw"] = af_raw; row["autoflip_sign_adj"] = af_adj
        rows.append(row)

T = pd.DataFrame(rows)
T.to_csv("signal_zoo/roll_rescore/rescore_full.csv", index=False)

for label in ("full", "2019+"):
    t = T[T.window == label].set_index("signal").sort_values("SR_net_adj", ascending=False)
    print(f"\n{'='*130}\n{label}  —  realistic cost (70/side + 4e-5), frozen signs, c2c exec_lag=2\n{'='*130}")
    show = t[["enabled", "SR_net_raw", "SR_net_adj", "SR_net_d",
              "SR_of_SR_raw", "SR_of_SR_adj", "SR_of_SR_d",
              "positive_years_raw", "positive_years_adj",
              "beta_raw", "beta_adj", "beta_d", "n_years_adj",
              "gates_raw", "gates_adj", "gate_flips"]]
    print(show.to_string())

print(f"\nmean |ΔSR_net| full  = {T[T.window=='full']['SR_net_d'].abs().mean():.4f}"
      f"   max = {T[T.window=='full']['SR_net_d'].abs().max():.4f}")
print(f"mean |ΔSR_net| 2019+ = {T[T.window=='2019+']['SR_net_d'].abs().mean():.4f}"
      f"   max = {T[T.window=='2019+']['SR_net_d'].abs().max():.4f}")
print("\nwrote signal_zoo/roll_rescore/rescore_full.csv")
