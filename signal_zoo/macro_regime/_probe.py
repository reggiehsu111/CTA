import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta
from cta.signals._ctx import build_context
import cta.signals as S

ctx = build_context()
A = ctx.asset
print("asset rows", len(A.index), A.index.min().date(), A.index.max().date())
print("last vol", A["volume"].iloc[-1], "open", A["open"].iloc[-1], "night_open", A["night_open"].iloc[-1])
assert A.index.max() >= pd.Timestamp("2026-08-29")
assert A["volume"].iloc[-1] == A["volume"].iloc[-1]
assert A["open"].iloc[-1] != A["night_open"].iloc[-1]
print("preamble OK")

from cta.signals._base import SIGNAL_REGISTRY
print("registry", len(SIGNAL_REGISTRY))
for n, s in SIGNAL_REGISTRY.items():
    print(f"  {n:44s} sign={s.sign:+d} enabled={s.enabled} rec={s.recommended_variants} pre_norm={s.pre_normalized}")

cat = cta.macro_catalog()
for sid in ["igrea", "us_stlfsi", "epu_global"]:
    r = cat.loc[sid]
    print(sid, "|", r["label"], "|", r["freq"], "| lag", r["pub_lag_days"], "|", r["first_obs"], "->", r["last_obs"], "| n", r["n_obs"], "|", r["units"])
    x = ctx.macro(sid)
    print("   tw-aligned nonnull:", int(x.notna().sum()), "first", x.first_valid_index(), "last val", x.dropna().iloc[-1])
n = ctx.nfci_tw("nfci")
print("nfci_tw nonnull", int(n.notna().sum()), "first", n.first_valid_index(), "last", n.dropna().index[-1].date(), n.dropna().iloc[-1])
