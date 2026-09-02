"""QNT-15 follow-up: does an IS regime split predict the OOS one?

The full-sample dSR is a weighted average of IS and OOS, so "the significant
cells also agree IS/OOS" is mechanically induced, not evidence. The honest
test is: freeze the split direction on IS only, then look at OOS.
"""
import pandas as pd, numpy as np
D = "/home/ubuntu/mtx/signal_zoo/macro_regime"
d = pd.read_csv(f"{D}/regime_deltas.csv")
d = d[~d.signal.str.startswith("_")]
piv = d.pivot_table(index=["signal", "dim"], columns="window", values="dSR").dropna().reset_index()
p = piv[piv.dim.isin(["igrea", "stlfsi", "epu", "nfci"])]
print(f"all {len(p)} primary splits: IS/OOS sign agreement "
      f"{(np.sign(p.IS) == np.sign(p.OOS)).mean():.1%}")
for q in (0.5, 0.75, 0.9):
    thr = p.IS.abs().quantile(q)
    s = p[p.IS.abs() >= thr]
    print(f"  top {int((1-q)*100):>2}% by |dSR_IS| (>= {thr:.2f}): OOS agrees "
          f"{(np.sign(s.IS) == np.sign(s.OOS)).mean():.0%} of {len(s)}; "
          f"mean sign(IS)*dSR_OOS = {(np.sign(s.IS) * s.OOS).mean():+.3f}")
print(f"corr(dSR_IS, dSR_OOS) = {p.IS.corr(p.OOS):.3f}")
print(f"mean sign(IS)*dSR_OOS over all = {(np.sign(p.IS) * p.OOS).mean():+.3f}")
