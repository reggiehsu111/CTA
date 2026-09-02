"""QNT-18 part 3d: the window channel alone, at MATCHED lag, on all three grids.

Part 2 scored the registered signals with `day` at its declared shift(1) against
`c2c` at shift(2) — that conflates the window with an extra session of
information. Part 1 held the slow macro at shift(2) for both. To answer "does the
WINDOW effect generalise" the three grids have to be lined up the same way, so
this re-scores the registered signals at day@2 vs c2c@2 and puts all three
side by side against QNT-14's own lag_decomposition.csv.

Also reports day@1 vs day@2 per grid = the information channel, where legal.
"""
import sys, warnings, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd

OUT = "/home/ubuntu/mtx/signal_zoo/macro_windows"
SWEEP = f"{OUT}/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))   # noqa: S102
import cta
from cta.signals import SIGNAL_REGISTRY
from cta.signals._base import VARIANT_REGISTRY
pd.set_option("display.width", 210)

# ── registered signals at BOTH lags, declared frozen sign ─────────────────
rows = []
for name, obj in sorted(SIGNAL_REGISTRY.items()):
    cls = type(obj)
    raw = obj.compute_raw(ctx).astype(float)
    s = raw if cls.pre_normalized else cta.normalize_signal(raw, method="tanh", window=252)
    s = (s * cls.sign).replace([np.inf, -np.inf], np.nan)
    for v in ("c2c", "day"):
        for lag in (1, 2):
            keep, _SHIFT[v] = _SHIFT[v], lag
            try:
                st = wstats(s, v, sign=1)
                if st:
                    rows.append(dict(signal=name, variant=v, lag=lag, SR=st["SR_net"],
                                     srsr=st["SR_of_SR"], tilt=st["mean_exec_w"],
                                     beta=st["beta"], enabled=bool(cls.enabled)))
            finally:
                _SHIFT[v] = keep
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/registered_lag_matched.csv", index=False)

def dpair(d, key, va, la, vb, lb, val="SR"):
    a = d[(d.variant == va) & (d.lag == la)].set_index(key)[val]
    b = d[(d.variant == vb) & (d.lag == lb)].set_index(key)[val]
    return (a - b).dropna()

# QNT-14's own decomposition file, full regime
L = pd.read_csv(f"{OUT}/lag_decomposition.csv").query("regime=='full'")
L = L.rename(columns={"SR_net": "SR", "SR_of_SR": "srsr"})
S = pd.read_csv(f"{OUT}/slow_window_sweep.csv")
S["lag"] = S["shift"]; S = S.rename(columns={"SR_net": "SR", "SR_of_SR": "srsr"})

GRIDS = [("daily macro  (198 cells)", L, "cand"),
         ("slow macro   (378 cells)", S, "cand"),
         ("registered   (11 signals)", R, "signal")]

print("=== WINDOW CHANNEL ALONE:  day@lag2  −  c2c@lag2   (information held constant) ===")
print(f"{'grid':28s} {'n':>4s} {'medΔSR':>8s} {'meanΔSR':>8s} {'win':>7s} {'medΔSR_of_SR':>13s}")
for g, d, k in GRIDS:
    x = dpair(d, k, "day", 2, "c2c", 2)
    y = dpair(d, k, "day", 2, "c2c", 2, "srsr")
    print(f"{g:28s} {len(x):4d} {x.median():+8.4f} {x.mean():+8.4f} {(x>0).mean():7.1%} {y.median():+13.4f}")

print("\n=== INFORMATION CHANNEL ALONE:  day@lag1 − day@lag2 ===")
print("  (legal for all three grids: daily macro pub_lag 0 → day lag 1; the registered")
print("   sources all publish by 15:00-20:35 TPE of t-1; NOT legal for slow macro, shown")
print("   as a counterfactual only)")
for g, d, k in GRIDS:
    x = dpair(d, k, "day", 1, "day", 2)
    print(f"{g:28s} {len(x):4d} {x.median():+8.4f} {x.mean():+8.4f} {(x>0).mean():7.1%}")

print("\n=== HEADLINE:  day@declared-lag − c2c@2  (what a variant switch would actually do) ===")
for g, d, k in GRIDS:
    dl = 1 if "slow" not in g else 2      # slow macro cannot use lag 1
    x = dpair(d, k, "day", dl, "c2c", 2)
    y = dpair(d, k, "day", dl, "c2c", 2, "srsr")
    print(f"{g:28s} day@{dl}  n={len(x):4d} medΔSR {x.median():+.4f} win {(x>0).mean():5.1%}  "
          f"medΔSR_of_SR {y.median():+.4f}")

print("\n=== registered detail, both lags ===")
print(R.pivot_table(index="signal", columns=["variant", "lag"], values="SR").round(3).to_string())
