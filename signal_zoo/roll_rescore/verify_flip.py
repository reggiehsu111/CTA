"""QNT-13 close-out verification: the roll_adjusted default is now True, and
calling signal_stats with NO flag reproduces the roll-adjusted column of the
QNT-13 delta table under the current (post-QNT-21) code."""
import sys, pickle, inspect, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cta

A = cta.load_asset("mtx", "1d")
print(f"asset rows={len(A)} last={A.index.max().date()} vol={A['volume'].iloc[-1]} "
      f"open={A['open'].iloc[-1]} night_open={A['night_open'].iloc[-1]}")
assert A.index.max() >= pd.Timestamp("2026-08-31"),    "stale asset"
assert A["volume"].iloc[-1] == A["volume"].iloc[-1],   "NaN volume"
assert A["open"].iloc[-1] != A["night_open"].iloc[-1], "day==night"

d = inspect.signature(cta.signal_stats).parameters["roll_adjusted"].default
print(f"\nsignal_stats default roll_adjusted = {d}")
assert d is True

D = pickle.load(open("signal_zoo/roll_rescore/signals.pkl", "rb"))
SIGS, META = D["sigs"], D["meta"]
REAL = dict(fixed_per_side=70.0, fee_rate=4e-5)
T = pd.read_csv("signal_zoo/roll_rescore/rescore_full.csv").set_index(["signal", "window"])
COLS = ["SR_net", "SR_of_SR", "positive_years", "beta", "n_years"]

# 1. default == explicit True, and == the published _adj column
worst_def, worst_pub = 0.0, 0.0
rows = []
for name, sig in SIGS.items():
    for w, st in [("full", None), ("2019+", "2019-01-01")]:
        nofl = cta.signal_stats(sig, A, start=st, auto_flip=False, **REAL)           # no flag
        expl = cta.signal_stats(sig, A, start=st, auto_flip=False, roll_adjusted=True, **REAL)
        raw  = cta.signal_stats(sig, A, start=st, auto_flip=False, roll_adjusted=False, **REAL)
        for k in COLS:
            worst_def = max(worst_def, abs(nofl[k] - expl[k]))
            worst_pub = max(worst_pub, abs(nofl[k] - T.loc[(name, w), f"{k}_adj"]))
        if w == "full":
            g = lambda r: sum([r["SR_of_SR"] > .6, r["positive_years"] >= .65,
                               abs(r["beta"]) < .15, r["n_years"] >= 5])
            rows.append(dict(signal=name, en=META[name]["enabled"],
                             SR_raw=round(raw["SR_net"],3), SR_now=round(nofl["SR_net"],3),
                             d=round(nofl["SR_net"]-raw["SR_net"],3),
                             SoS=round(nofl["SR_of_SR"],3), posyr=round(nofl["positive_years"],3),
                             beta=round(nofl["beta"],3), gates=f"{g(raw)}->{g(nofl)}"))
print(f"max |default - explicit True| over 11 signals x 2 windows x 5 metrics = {worst_def:.2e}")
print(f"max |default - published _adj column|                                 = {worst_pub:.2e}")

# 2. batch_signal_stats inherits the new default
b_def = cta.batch_signal_stats(SIGS, A, auto_flip=False, **REAL)
b_raw = cta.batch_signal_stats(SIGS, A, auto_flip=False, roll_adjusted=False, **REAL)
print(f"\nbatch_signal_stats: max |default - published _adj| = "
      f"{max(abs(b_def.loc[n,k]-T.loc[(n,'full'),f'{k}_adj']) for n in SIGS for k in COLS):.2e}")
print(f"batch B&H-style sanity: mean |SR_net default - SR_net raw| = "
      f"{(b_def['SR_net']-b_raw['SR_net']).abs().mean():.4f}")

bh = pd.Series(1.0, index=A.index)
for w, st in [("full", None), ("2019+", "2019-01-01")]:
    r0 = cta.signal_stats(bh, A, start=st, auto_flip=False, roll_adjusted=False, **REAL)["SR_net"]
    r1 = cta.signal_stats(bh, A, start=st, auto_flip=False, **REAL)["SR_net"]
    print(f"buy-and-hold {w:6s}: raw {r0:.3f} -> default(now) {r1:.3f}")

print("\n" + pd.DataFrame(rows).sort_values("SR_now", ascending=False).to_string(index=False))
