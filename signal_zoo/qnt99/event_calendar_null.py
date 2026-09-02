"""QNT-99 Part A4 — is the calendar sweep's 70 gate-passers an EVENT effect, or
just the MTX night-session drift sampled on a subset of days?

Every passer has sign = +1 and most sit on the `night` leg, whose unconditional
mean is +4.7..+6.5 bps/day against ~0 for the day leg. A long-only mask over any
5-15% of nights inherits that drift. The control is therefore a MATCHED RANDOM
MASK: same number of on-days, same era, same window, 1000 draws.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
sys.path.insert(0, "/home/ubuntu/mtx/signal_zoo/qnt99")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd, event_inputs as ei
OUT = "/home/ubuntu/mtx/signal_zoo/qnt99"
TI = A.index
_RET["night"] = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"] = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1

print("=== unconditional legs, 2010- (always long 1 unit, real costs) ===")
for w in ("c2c", "o2o", "day", "ongap", "night"):
    s = pd.Series(1.0, index=TI)
    st = wstats(s, w, start="2010-01-01", sign=1)
    print(f"  always-long {w:6s} SR_net {st['SR_net']:+.3f}  SR_of_SR {st['SR_of_SR']:+.3f}  "
          f"pos_years {st['positive_years']:.2f}  beta {st['beta']:+.3f}")

cs = pd.read_csv(f"{OUT}/event_calendar_sweep.csv")
g = cs[(cs.full_SR_of_SR > 0.6) & (cs.full_positive_years >= 0.65)
       & (cs.full_beta.abs() < 0.15) & (cs.full_n_years >= 5)].copy()
print(f"\n=== matched random-mask null for the {len(g)} gate-passers ===")
print(f"  signs: {g['sign'].value_counts().to_dict()}   windows: {g['window'].value_counts().to_dict()}")

rng = np.random.default_rng(20260902)
NDRAW = 400
res = []
for _, r in g.iterrows():
    ev, off, hold, w = r["event"], int(r["offset"]), int(r["hold"]), r["window"]
    d = ei.event_tw_dates(ev, TI)
    pos = pd.Index(TI).get_indexer(pd.DatetimeIndex(d.values))
    pos = np.unique(np.clip(pos[pos >= 0] + off, 0, len(TI) - 1))
    n_on = len(pos)
    lo = TI[pos].min()
    pool = np.where(TI >= lo)[0]
    null = []
    for _ in range(NDRAW):
        pk = rng.choice(pool, size=min(n_on, len(pool)), replace=False)
        m = pd.Series(0.0, index=TI); m.iloc[np.sort(pk)] = 1.0
        sig = m.rolling(hold, min_periods=1).max()
        st = wstats(sig, w, sign=int(r["sign"]))
        if st: null.append(st["SR_net"])
    null = np.array(null)
    res.append(dict(cell=r["cell"], window=w, sign=int(r["sign"]), n_on=n_on,
                    SR=r["full_SR_net"], null_med=float(np.median(null)),
                    null_p90=float(np.percentile(null, 90)),
                    pct=float((null < r["full_SR_net"]).mean())))
    print(f"  {r['cell']:38s} SR {r['full_SR_net']:+.3f}  matched-null med "
          f"{np.median(null):+.3f} p90 {np.percentile(null,90):+.3f}  -> pctile "
          f"{(null < r['full_SR_net']).mean():.3f}", flush=True)
R = pd.DataFrame(res)
R.to_csv(f"{OUT}/event_calendar_null.csv", index=False)
print(f"\nof {len(R)} gate-passers, {(R.pct>0.95).sum()} beat their own matched random mask at the 95th pct "
      f"(expected by chance {0.05*len(R):.1f}); median percentile {R.pct.median():.3f}")
