"""QNT-100 part E (corrected) — does the new gate have POWER?

Part B showed the fix does not admit sparse *noise*. That is not the question a
gate has to answer: noise fails the other three gates anyway. The question is
whether a sparse book with a REAL two-sided edge still gets through, i.e.
whether `beta/mean|exec_w|` is a directionality gate or just a sparsity gate in
disguise.

On a random `frac` of nights the mask takes that night's realised direction
with probability `p`, and the opposite otherwise. So: genuine skill, beta ~ 0
by construction, `mean|exec_w| = frac`.

The first attempt got 0.6% pass rate because the mask was built on the night of
day `t` while `wstats` executes it with `shift(1)`, i.e. one night late - the
"informed" masks were informed about the wrong night and had no edge at all
(mean SR_net -0.19). The target here is `nr.shift(-lag)`.
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd
import cta
OUT = "/home/ubuntu/mtx/signal_zoo/qnt100"
TI = A.index
_RET["night"]   = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"]  = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1
LAG = _SHIFT["night"]

nr  = _RET["night"].reindex(TI)
tgt = np.sign(nr.shift(-LAG).fillna(0).values)   # the night this bar's position will trade
tgt[tgt == 0] = 1.0

rng = np.random.default_rng(11)
rows = []
for frac in (0.05, 0.10, 0.20, 0.50):
    for p in (0.52, 0.55, 0.60):
        for rep in range(60):
            n = max(1, int(frac * len(TI)))
            idx = np.sort(rng.choice(len(TI), n, replace=False))
            ok = rng.random(n) < p
            m = pd.Series(0.0, index=TI)
            m.iloc[idx] = np.where(ok, tgt[idx], -tgt[idx])
            st = wstats(m, "night", start="2010-01-01", sign=1)
            if st:
                rows.append(dict(frac=frac, p=p, rep=rep, **st))
E = pd.DataFrame(rows)
E["mean_abs_w"] = E["abs_exec_w"]
E = cta.house_gates(E, beta_mode="both")
E["passes_old"] = E.gate_srsr & E.gate_posyr & E.gate_nyr & E.gate_beta_raw
print("=== E. sparse but genuinely informed two-sided masks (60 draws/row) ===")
print(E.groupby(["frac", "p"]).agg(
    SR_net=("SR_net", "mean"), SR_of_SR=("SR_of_SR", "mean"), beta=("beta", "mean"),
    abs_w=("abs_exec_w", "mean"), beta_per_w=("beta_per_w", "mean"),
    pass_OLD=("passes_old", "mean"), pass_NEW=("passes", "mean")).round(3).to_string())
print(f"\n  overall: OLD {E.passes_old.mean()*100:.1f}%  ->  NEW {E.passes.mean()*100:.1f}%")
kept = E[E.passes_old].passes.mean() if E.passes_old.any() else float("nan")
print(f"  of the masks the OLD gate admitted, the NEW gate keeps {kept*100:.1f}%")
lost = E[E.passes_old & ~E.passes]
print(f"  lost {len(lost)} of {int(E.passes_old.sum())}"
      + (f"; their median |beta/w| {lost.beta_per_w.abs().median():.3f} "
         f"vs {E[E.passes].beta_per_w.abs().median():.3f} for the kept" if len(lost) else ""))
E.to_csv(f"{OUT}/power_informed_masks.csv", index=False)
json.dump(dict(pow_old=float(E.passes_old.mean()), pow_new=float(E.passes.mean()),
               retained=float(kept)), open(f"{OUT}/summary_power.json", "w"), indent=1)
