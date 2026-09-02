"""QNT-13 step 1: build the 11 registered signals exactly as the runner does
(compute_raw -> normalize_signal(tanh,252) unless pre_normalized -> x sign),
and pickle them alongside the asset for the re-scoring step.

Read-only. Does not import or invoke cta.signals.runner.
"""
import sys, pickle, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")

import pandas as pd
import cta
from cta.signals._base import SIGNAL_REGISTRY
from cta.signals._ctx import Context

A = cta.load_asset("mtx", "1d")

# ── standing-brief preamble: never trust a stale/corrupt asset ──────────────
print(f"asset rows={len(A)} last={A.index.max().date()} "
      f"last_vol={A['volume'].iloc[-1]} open={A['open'].iloc[-1]} night_open={A['night_open'].iloc[-1]}")
assert A.index.max() >= pd.Timestamp("2026-08-31"),      "stale asset - do not proceed"
assert A["volume"].iloc[-1] == A["volume"].iloc[-1],     "NaN volume = night-table bug"
assert A["open"].iloc[-1] != A["night_open"].iloc[-1],   "day==night = corrupted row"

ctx = Context(A)

meta, sigs = {}, {}
for name, s in sorted(SIGNAL_REGISTRY.items()):
    raw = s.compute_raw(ctx)
    norm = raw if s.pre_normalized else cta.normalize_signal(raw, method="tanh", window=252)
    sigs[name] = (norm * s.sign).rename(name)
    meta[name] = dict(enabled=s.enabled, sign=s.sign, live_date=s.live_date,
                      pre_normalized=s.pre_normalized,
                      recommended_variants=tuple(s.recommended_variants),
                      variants=tuple(s.variants),
                      shift_override=dict(s.shift_override),
                      cn_name=s.cn_name)
    v = sigs[name].dropna()
    print(f"{name:45s} enabled={str(s.enabled):5s} sign={s.sign:+d} "
          f"n={len(v):5d} {v.index.min().date()}→{v.index.max().date()} rec={s.recommended_variants}")

# rollover diagnostics
roll = A.is_rollover
spread = (A["back_close"].shift(1) / A["close"].shift(1) - 1).where(roll).dropna()
print(f"\nrollover days={int(roll.sum())} mean|spread|={spread.abs().mean()*1e4:.1f}bps "
      f"max|spread|={spread.abs().max()*1e4:.1f}bps")

with open("signal_zoo/roll_rescore/signals.pkl", "wb") as f:
    pickle.dump({"sigs": sigs, "meta": meta}, f)
print("\nwrote signal_zoo/roll_rescore/signals.pkl")
