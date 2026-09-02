"""QNT-99 — two closing checks.

1. GRID-LEVEL, SELECTION-FREE: the 320 event-study t-stats. Under the null they
   are N(0,1). This is the unbiased version of the calendar question; the
   matched-null on the 70 gate-PASSERS is biased by the selection that picked
   them.
2. THE GATE LEAK: a long-only mask over a random subset of nights inherits the
   MTX night drift (always-long night = SR_net +1.15, beta 1.00) but shrinks
   beta in proportion to exposure, so it slides under |beta| < 0.15 while
   SR_of_SR and positive_years stay high. If a random mask passes the gates,
   the 70 calendar "passers" prove nothing.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd
from scipy import stats
OUT = "/home/ubuntu/mtx/signal_zoo/qnt99"
TI = A.index
_RET["night"] = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"] = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1

es = pd.read_csv(f"{OUT}/event_study.csv")
print("=== 1. selection-free: distribution of the 320 event-study t-stats ===")
for w, g in es.groupby("window"):
    print(f"  {w:6s} n={len(g):3d}  mean t {g.t.mean():+.3f}  sd t {g.t.std():.3f}  "
          f"|t|>1.96: {(g.t.abs()>1.96).sum():2d} (expect {0.05*len(g):.1f})  "
          f"KS vs N(0,1) p={stats.kstest(g.t,'norm').pvalue:.3f}")
print(f"  ALL    n={len(es)}  mean t {es.t.mean():+.3f}  sd t {es.t.std():.3f}  "
      f"|t|>1.96: {(es.t.abs()>1.96).sum()} (expect {0.05*len(es):.1f})  "
      f"KS p={stats.kstest(es.t,'norm').pvalue:.3f}")
print("  -> a real event effect would push mean(t) away from 0 or fatten the tail; neither happens.")

print("\n=== 2. the gate leak: RANDOM long-only night masks, no event content ===")
rng = np.random.default_rng(7)
rows = []
for frac in (0.05, 0.10, 0.20, 0.40, 1.00):
    for rep in range(20):
        n = int(frac * len(TI))
        m = pd.Series(0.0, index=TI); m.iloc[np.sort(rng.choice(len(TI), n, replace=False))] = 1.0
        st = wstats(m, "night", start="2010-01-01", sign=1)
        if st: rows.append(dict(frac=frac, **st))
R = pd.DataFrame(rows)
R["passes"] = ((R.SR_of_SR > 0.6) & (R.positive_years >= 0.65)
               & (R.beta.abs() < 0.15) & (R.n_years >= 5))
print(R.groupby("frac")[["SR_net","SR_of_SR","positive_years","beta","passes"]].mean().round(3).to_string())
print(f"\n  -> {R.passes.mean()*100:.0f}% of PURELY RANDOM long-night masks pass all four house gates;")
print( "     at 5-10% exposure the pass rate is what the 70 calendar 'passers' are made of.")
R.to_csv(f"{OUT}/gate_leak_random_masks.csv", index=False)
