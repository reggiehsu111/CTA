"""QNT-98 part 2 — the null for the argument sweep, plus the figures.

The neighbourhood sweep says a candidate's best cell is ~1.0-1.16 SR. This asks
what that number is worth: score the IDENTICAL 144-cell neighbourhoods against
CIRCULARLY SHIFTED returns. The position series, their autocorrelation, the
turnover, the cost model and the return distribution are all preserved; only the
alignment between signal and return is destroyed. Whatever "best cell" and
"4-gate pass rate" that produces is what search over the argument grid buys for
free.

Writes only to signal_zoo/qnt98/.
"""
import sys, os, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")

SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))   # noqa: S102

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cta
from cta.signals import _operators as ops

OUT = "/home/ubuntu/mtx/signal_zoo/qnt98"
sys.path.insert(0, OUT)
NB = pd.read_csv(f"{OUT}/neighbourhood_sweep.csv")

# reuse the palette + candidate table from the sweep script without re-running it
_ns = io.open(f"{OUT}/neighbourhood_sweep.py", encoding="utf-8").read()
_head = _ns.split("# ── Sweep ─")[0].split("SWEEP = ")[0]
exec(compile(_ns.split("# ── Reproduction check")[0].split("import numpy as np, pandas as pd\nimport cta")[1]
             .replace("from cta.signals import _operators as ops", ""), "<palette>", "exec"))
PALETTE_OK = "OPS" in dir()
print("palette ops:", len(OPS), "windows:", WINDOWS, "candidates:", len(CANDIDATES))

cat = cta.macro_catalog()
def raw_series(sid, kind):
    if kind == "yoy":
        per = {"M": 12, "Q": 4, "D": 252, "W": 52}.get(cat.loc[sid, "freq"], 12)
        return ctx.macro_yoy(sid, per).astype(float)
    return ctx.macro(sid).astype(float)

def wstats_at(sig, variant, cost="real", lag=None, **kw):
    keep_c, keep_s = _COST[variant], _SHIFT[variant]
    _COST[variant] = COST_LADDER[cost][variant]
    if lag is not None:
        _SHIFT[variant] = lag
    try:
        return wstats(sig, variant, **kw)
    finally:
        _COST[variant], _SHIFT[variant] = keep_c, keep_s

# ── build every candidate neighbourhood's signal once ─────────────────────
SIG = {}
for sid, kind, v, cop, cw, grid, conv in CANDIDATES:
    x = raw_series(sid, kind)
    freq = str(cat.loc[sid, "freq"]); lag = None if freq == "D" else 2
    for opn, fn in OPS.items():
        for w in WINDOWS:
            s = cta.normalize_signal(fn(x, w).replace([np.inf, -np.inf], np.nan),
                                     method="tanh", window=252)
            if not s.dropna().empty:
                SIG[(sid, opn, w)] = (s, v, conv, lag)
print(f"signals cached: {len(SIG)}")

# ── circular-shift null ───────────────────────────────────────────────────
def roll_returns(v, k):
    """Circularly shift a variant's return leg by k bars, preserving its NaN mask."""
    r = _RET[v].astype(float)
    m = r.notna().values
    vals = np.array(r.values, dtype=float, copy=True)
    vals[m] = np.roll(vals[m], k)
    return pd.Series(vals, index=r.index, name=r.name)

RNG = np.random.default_rng(20260902)
SHIFTS = [int(k) for k in RNG.integers(400, 1800, size=4)]
print("null circular shifts (bars):", SHIFTS)

null_rows = []
for k in SHIFTS:
    keep = {v: _RET[v] for v in _RET}
    for v in _RET:
        _RET[v] = roll_returns(v, k)
    try:
        for (sid, opn, w), (s, v, conv, lag) in SIG.items():
            is_st = wstats_at(s, v, "real", lag=lag, start=conv["start"], end=conv["is_end"])
            if is_st is None:
                continue
            full = wstats_at(s, v, "real", lag=lag, start=conv["start"], sign=is_st["sign"])
            if full is None:
                continue
            null_rows.append(dict(shift=k, series=sid, op=opn, op_family=FAMILY[opn], window=w,
                                  SR_net=full["SR_net"], SR_of_SR=full["SR_of_SR"],
                                  positive_years=full["positive_years"], beta=full["beta"],
                                  n_years=full["n_years"]))
    finally:
        for v in _RET:
            _RET[v] = keep[v]
    print(f"  shift {k}: {len(null_rows)} rows cumulative", flush=True)

NU = pd.DataFrame(null_rows)
NU["n_gates"] = ((NU.SR_of_SR > 0.6).astype(int) + (NU.positive_years >= 0.65).astype(int)
                 + (NU.beta.abs() < 0.15).astype(int) + (NU.n_years >= 5).astype(int))
NU.to_csv(f"{OUT}/null_circular_shift.csv", index=False)

real_pass = 100 * (NB[NB.role == "candidate"].n_gates == 4).mean()
ctl_pass  = 100 * (NB[NB.role == "control"].n_gates == 4).mean()
null_pass = 100 * (NU.n_gates == 4).mean()
best_real = NB[NB.role == "candidate"].groupby("series").SR_net.max()
best_ctl  = NB[NB.role == "control"].groupby("series").SR_net.max()
best_null = NU.groupby(["shift", "series"]).SR_net.max()
print(f"\n4-gate pass rate  candidates {real_pass:.1f}%  controls {ctl_pass:.1f}%  "
      f"CIRCULAR-SHIFT NULL {null_pass:.1f}%")
print(f"best-of-144       candidates med {best_real.median():.3f}  controls {best_ctl.median():.3f}  "
      f"null med {best_null.median():.3f} (p90 {best_null.quantile(.9):.3f}, max {best_null.max():.3f})")
print(f"candidate best cells above the null p90: "
      f"{int((best_real > best_null.quantile(.9)).sum())}/{len(best_real)}")

summ = dict(real_4gate_pct=real_pass, control_4gate_pct=ctl_pass, null_4gate_pct=null_pass,
            best_real_med=float(best_real.median()), best_ctl_med=float(best_ctl.median()),
            best_null_med=float(best_null.median()), best_null_p90=float(best_null.quantile(.9)),
            best_null_max=float(best_null.max()))
pd.Series(summ).to_csv(f"{OUT}/null_summary.csv")
print("\n-> null_circular_shift.csv, null_summary.csv")
