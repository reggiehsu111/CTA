"""QNT-100 — validate `beta / mean|exec_w|` as the replacement beta gate.

Four questions, in the order that decides the house rule:

A. Does the new gate CLOSE the leak? Re-run QNT-99's random long-night masks
   (more draws, more fractions) under both rules.
B. Does it have POWER against the thing it is supposed to admit? Same masks,
   but two-sided (long/short 50-50) — genuinely market-neutral noise. The new
   gate must not reject these for being sparse; if it does, it is just a
   held_pct gate in disguise.
C. What does it COST on grids already published? Re-gate every historical MTX
   sweep that recorded mean|exec_w|.
D. What does it cost on the LIVE book? Re-gate the registered signals.

Nothing here re-runs a sweep, picks a sign, or writes to a config table.
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102  -- gives A, wstats, _RET/_COST/_SHIFT, PV, FIXED, FEE
import numpy as np, pandas as pd
import cta

OUT = "/home/ubuntu/mtx/signal_zoo/qnt100"
TI  = A.index
_RET["night"]   = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"]  = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1

OLD = dict(beta_mode="raw")
NEW = dict(beta_mode="per_w")

def rate(df, **kw):
    return cta.house_gates(df, **kw)["passes"].mean()

# ══ A + B — random masks, one-sided (the leak) and two-sided (the control) ══
print("=== A/B. random night masks: does the fix close the leak without killing sparsity? ===")
rng = np.random.default_rng(7)
NDRAW = 100
rows = []
for kind in ("long_only", "two_sided"):
    for frac in (0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.40, 1.00):
        for rep in range(NDRAW):
            n = max(1, int(frac * len(TI)))
            idx = np.sort(rng.choice(len(TI), n, replace=False))
            m = pd.Series(0.0, index=TI)
            m.iloc[idx] = 1.0 if kind == "long_only" else rng.choice([-1.0, 1.0], n)
            st = wstats(m, "night", start="2010-01-01", sign=1)
            if st:
                rows.append(dict(kind=kind, frac=frac, rep=rep, **st))
R = pd.DataFrame(rows)
R["mean_abs_w"] = R["abs_exec_w"]
R = cta.house_gates(R, beta_mode="both")          # gate_beta_raw + gate_beta(per_w)
R["passes_old"] = (R.gate_srsr & R.gate_posyr & R.gate_nyr & R.gate_beta_raw)
R["passes_new"] = R["passes"]
R.to_csv(f"{OUT}/gate_leak_masks_v2.csv", index=False)

for kind in ("long_only", "two_sided"):
    K = R[R.kind == kind]
    print(f"\n  --- {kind} ({NDRAW} draws/row, night leg, 2010-, real costs) ---")
    t = K.groupby("frac").agg(
        SR_net=("SR_net", "mean"), SR_of_SR=("SR_of_SR", "mean"),
        beta=("beta", "mean"), abs_w=("abs_exec_w", "mean"),
        beta_per_w=("beta_per_w", "mean"),
        pass_OLD=("passes_old", "mean"), pass_NEW=("passes_new", "mean"))
    print(t.round(3).to_string())
print(f"\n  overall long_only : OLD {R[R.kind=='long_only'].passes_old.mean()*100:.1f}%  "
      f"-> NEW {R[R.kind=='long_only'].passes_new.mean()*100:.1f}%")
print(f"  overall two_sided : OLD {R[R.kind=='two_sided'].passes_old.mean()*100:.1f}%  "
      f"-> NEW {R[R.kind=='two_sided'].passes_new.mean()*100:.1f}%")

# ══ C — every published grid that recorded mean|exec_w| ════════════════════
print("\n=== C. re-gate the published MTX grids (no re-simulation) ===")
GRIDS = [
    ("QNT-99 event calendar",  "qnt99/event_calendar_sweep.csv",  "full_"),
    ("QNT-99 event surprise",  "qnt99/event_surprise_full.csv",   "full_"),
    ("QNT-99 put OI",          "qnt99/put_oi_sweep_full.csv",     "full_"),
    ("QNT-98 neighbourhood",   "qnt98/neighbourhood_sweep.csv",   ""),
    ("QNT-94 macro windows",   "macro_windows/window_sweep_full.csv", ""),
    ("QNT-94 slow windows",    "macro_windows/slow_window_sweep.csv", ""),
    ("QNT-94 registered",      "macro_windows/registered_window_sweep.csv", ""),
]
grid_rows = []
for label, rel, pref in GRIDS:
    d = pd.read_csv(f"/home/ubuntu/mtx/signal_zoo/{rel}")
    d = d[d.get("note").isna()] if "note" in d.columns else d
    g = cta.house_gates(d, beta_mode="both", prefix=pref)
    old = (g.gate_srsr & g.gate_posyr & g.gate_nyr & g.gate_beta_raw)
    new = g.passes
    surv = g[old]
    grid_rows.append(dict(
        grid=label, cells=len(g), pass_old=int(old.sum()), pass_new=int(new.sum()),
        pct_old=100*old.mean(), pct_new=100*new.mean(),
        surv_abs_w_med=float(surv["beta_per_w"].notna().pipe(lambda _: surv[
            [c for c in ("mean_abs_w","abs_exec_w","full_abs_exec_w") if c in surv][0]].median()))
            if len(surv) else np.nan,
        surv_bpw_med=float(surv["beta_per_w"].median()) if len(surv) else np.nan))
    g.to_csv(f"{OUT}/regated_{rel.split('/')[-1]}", index=False)
G = pd.DataFrame(grid_rows)
print(G.round(3).to_string(index=False))
print(f"\n  TOTAL cells {G.cells.sum()}:  survivors {G.pass_old.sum()} -> {G.pass_new.sum()}")
G.to_csv(f"{OUT}/regated_grids.csv", index=False)

# Parts D (the registered book) and E (power) live in `gate_fix_d.py` and
# `power_test.py`: they need the signal registry and a corrected execution lag,
# and keeping them here would re-run the 1,600 mask simulations above.
json.dump(dict(
    leak_old=float(R[R.kind=="long_only"].passes_old.mean()),
    leak_new=float(R[R.kind=="long_only"].passes_new.mean()),
    ctrl_old=float(R[R.kind=="two_sided"].passes_old.mean()),
    ctrl_new=float(R[R.kind=="two_sided"].passes_new.mean()),
    grid_cells=int(G.cells.sum()), grid_old=int(G.pass_old.sum()),
    grid_new=int(G.pass_new.sum())), open(f"{OUT}/summary.json", "w"), indent=1)
print("\nwrote", OUT)
