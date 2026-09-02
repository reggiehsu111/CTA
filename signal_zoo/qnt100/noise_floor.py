"""QNT-100 — where is `beta/mean|exec_w|` measurable at all?

Dividing by `mean|exec_w|` divides the estimation error by it too. On books
whose TRUE beta is zero by construction (informed two-sided masks, part E, and
random two-sided masks, part B), `sd(beta_per_w)` is fitted against exposure to
find the exposure below which the 0.15 threshold is inside the noise.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
OUT = "signal_zoo/qnt100"
E = pd.read_csv(f"{OUT}/power_informed_masks.csv")
R = pd.read_csv(f"{OUT}/gate_leak_masks_v2.csv")
N = pd.concat([E.assign(src="informed"),
               R[R.kind == "two_sided"].assign(src="random")])   # true beta == 0 in both
t = N.groupby(["src", "frac"]).beta_per_w.agg(["size", "std"]).reset_index()
t = t[t["size"] >= 30]
# sd(beta_per_w) = k / sqrt(frac)  ->  fit k in logs
k = float(np.exp(np.mean(np.log(t["std"].values) + 0.5 * np.log(t.frac.values))))
t["fit"] = k / np.sqrt(t.frac)
print(f"sd(beta_per_w) ~ {k:.3f} / sqrt(mean|exec_w|)   (night leg, 2010-, ~4,100 bars)")
print(t.round(3).to_string(index=False))
w_meas = (k / 0.15) ** 2
print(f"\n  sd = the 0.15 threshold at mean|exec_w| = {w_meas:.3f}")
print(f"  sd = half the threshold (a 2-sigma read) at mean|exec_w| = {(k/0.075)**2:.3f}")
print("\n  -> below mean|exec_w| ~ {:.2f} the beta gate is inside its own noise: a truly".format(w_meas))
print("     neutral book fails it by chance and a directional one cannot be told apart.")
print("     That is a case for an EXPOSURE FLOOR, not for a beta threshold.")
from math import erfc, sqrt
print("\n     exposure  sd(b/w)  P(a TRUE-zero-beta book fails |b/w|<0.15 by noise alone)")
for f in (0.05, 0.10, 0.20, 0.31, 0.50, 1.00):
    sd = k / np.sqrt(f)
    print(f"       {f:.2f}      {sd:.3f}     {erfc(0.15/(sd*sqrt(2))):.2f}")
print("     (at full exposure sd = 0.084 = SE(beta) itself, so the new rule is exactly")
print("      as noisy as the old one where the old one was being used.)")
json.dump(dict(k=k, w_measurable=w_meas, w_2sigma=(k/0.075)**2), open(f"{OUT}/noise_floor.json","w"), indent=1)
