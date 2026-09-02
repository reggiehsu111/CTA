"""Do the three consumers now score the same return series?"""
import sys, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cta
from cta.signals import _base as B
A = cta.load_asset("mtx","1d")
df = A.df if hasattr(A,"df") else A
legs = {n:f for n,f in vars(B).items() if n.startswith("_") and n.endswith("_ret") and callable(f)}
print("legs found:", sorted(legs))
r_adj, r_raw = A.returns, A["close"].pct_change()
roll = A.is_rollover
for n,f in sorted(legs.items()):
    try: s = f(df)
    except Exception as e: print(f"{n:16s} ERR {e}"); continue
    s = pd.Series(s).reindex(A.index)
    d_adj = (s - r_adj).abs().max(); d_raw = (s - r_raw).abs().max()
    print(f"{n:16s} max|leg-asset.returns|={d_adj:.3e}  max|leg-close.pct_change|={d_raw:.3e}")
# Simulate path
sim = cta.Simulate(pd.Series(1.0, index=A.index), plot=False) if False else None
print(f"\nrollover days={int(roll.sum())}  mean|adj-raw| on roll days="
      f"{(r_adj-r_raw).abs()[roll].mean()*1e4:.1f}bps")
