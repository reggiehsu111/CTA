"""QNT-19: what would a `pub_lag_days=1` floor on load_macro_tw actually cost?

The proposal on QNT-19 is to make `load_macro_tw` mirror `load_us_index_tw` and
apply a +1 calendar-day floor to US-close daily series, so the loader — not the
signal author — enforces PIT safety. Holding the variant default shifts fixed,
that floor is arithmetically IDENTICAL to running today's alignment one lag
later: c2c/o2o 2→3, day 1→2. So the cost is measurable with the machinery
QNT-14 already built.

This re-uses `macro_window_sweep.py` verbatim by exec'ing only its setup block
(everything above the "Build the signals once" marker), so `_RET`, `_COST` and
`wstats` are bit-identical to the published evidence and nothing here overwrites
`window_sweep_full.csv` or `lag_decomposition.csv`.

Reports, per variant, the full-regime median paired ΔSR_net for:
    legal - illegal   (lag 2 vs 1 on c2c/o2o; 1 vs 2 on day)  = the trap's size
    floored - legal   (lag 3 vs 2 on c2c/o2o; 2 vs 1 on day)  = the floor's cost
plus the four-gate passer count at each lag.

Nothing here selects a sign, a variant or a default. Reggie decides.

QNT-82 CAVEAT — re-read this before quoting `floor_cost.csv`
------------------------------------------------------------
QNT-19 SHIPPED the floor: `load_macro_tw` now applies the +1 calendar-day floor
itself. This script measures lags on top of whatever the loader already does, so
a re-run no longer answers the question the docstring above poses. On the CSV
re-run 2026-09-01 10:32, `lag1` is the floored default and `lag2`/`lag3` are one
and two days BEYOND it — the "trap" and "floor cost" columns are therefore a
second floor, not the original one. The pre-floor CSV that backs the QNT-19
decision is kept at `macro_windows/qnt82/pre_rerun/floor_cost.csv` (2026-09-01
03:54). Quote that one for the floor decision; quote the current one only for
"what would ANOTHER day of lag cost".
"""
import sys, warnings, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")

SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
_marker = "# ── Build the signals once"
assert _marker in _src, "macro_window_sweep.py changed shape - re-check the split point"
exec(compile(_src.split(_marker)[0], SWEEP, "exec"))          # noqa: S102

import numpy as np, pandas as pd
import cta

OUT = "/home/ubuntu/mtx/signal_zoo/macro_windows"

# rebuild the identical 198-cell grid
raw = {sid: ctx.macro(sid).astype(float) for sid in DAILY}
SIGS = {}
for sid, x in raw.items():
    for tn, tf in TRANSFORMS.items():
        for w in WINDOWS:
            s = tf(x, w).replace([np.inf, -np.inf], np.nan)
            s = cta.normalize_signal(s, method="tanh", window=252)
            if s.dropna().empty:
                continue
            SIGS[f"{sid}|{tn}|w{w}"] = s
print(f"signals built: {len(SIGS)}")

R_START, IS_END, OOS_START = None, "2018-12-31", "2019-01-01"
VARIANTS = ("c2c", "o2o", "day")

rows = []
for cand, s in SIGS.items():
    for v in VARIANTS:
        for lag in (1, 2, 3):
            _SHIFT[v], keep = lag, _SHIFT[v]
            try:
                is_st = wstats(s, v, start=R_START, end=IS_END)
                if is_st is None:
                    continue
                full = wstats(s, v, start=R_START, sign=is_st["sign"])
                oos = wstats(s, v, start=OOS_START, sign=is_st["sign"])
                if full:
                    rows.append(dict(cand=cand, variant=v, lag=lag, sign_IS=is_st["sign"],
                                     SR_IS=is_st["SR_net"],
                                     SR_OOS=(oos or {}).get("SR_net", np.nan),
                                     SR_net=full["SR_net"], SR_of_SR=full["SR_of_SR"],
                                     positive_years=full["positive_years"],
                                     n_years=full["n_years"], beta=full["beta"],
                                     mean_exec_w=full["mean_exec_w"]))
            finally:
                _SHIFT[v] = keep

d = pd.DataFrame(rows)
d["pass4"] = ((d.SR_of_SR > 0.6) & (d.positive_years >= 0.65)
              & (d.beta.abs() < 0.15) & (d.n_years >= 5))
d.to_csv(f"{OUT}/floor_cost.csv", index=False)
print(f"cells: {len(d)} -> floor_cost.csv\n")

p = d.pivot_table(index=["cand", "variant"], columns="lag", values="SR_net").dropna()

# lag that is PIT-legal TODAY (pub_lag_days=0), and under the proposed +1 floor
LEGAL   = {"c2c": 2, "o2o": 2, "day": 1}
FLOORED = {"c2c": 3, "o2o": 3, "day": 2}

print("=== median paired ΔSR_net, full regime, net of real MTX costs ===")
print(f"{'variant':8s} {'n':>4s}  {'trap (illegal-legal)':>22s}  {'floor cost (floored-legal)':>28s}")
for v in VARIANTS:
    q = p.xs(v, level="variant")
    lg, fl = LEGAL[v], FLOORED[v]
    ill = lg - 1
    # `day` is already legal at its variant default (lag 1), so it has no trap
    # column - lag 0 is not a thing an author could reach by copying o2o:1.
    if ill in q.columns:
        dt = q[ill] - q[lg]
        trap = f"lag{ill}-lag{lg} {dt.median():+.3f} ({(dt>0).mean():5.1%})"
    else:
        trap = "n/a (default already legal)"
    dfl = q[fl] - q[lg]
    print(f"{v:8s} {len(q):4d}  {trap:>22s}   "
          f"lag{fl}-lag{lg} {dfl.median():+.3f} ({(dfl>0).mean():5.1%})")

print("\n=== four-gate passers per (variant, lag) ===")
print(d.pivot_table(index="variant", columns="lag", values="pass4", aggfunc="sum"))
print("\nlegal-today lags:", LEGAL, " under proposed floor:", FLOORED)
