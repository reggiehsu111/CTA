"""QNT-100 part D+E — the cost of the fix.

D. The LIVE book: every registered signal, built exactly as the runner builds
   it (declared sign, tanh-252 unless pre_normalized), scored on its declared
   variants, under the old and new beta rule.
E. POWER: does the new gate reject a sparse book that has a REAL two-sided
   edge? Part B showed it does not reject sparse *noise*, but noise fails the
   other gates anyway. Here the mask is informed - on a random `frac` of
   nights it takes the sign of that night's return with probability `p` - so
   it has genuine skill, beta ~ 0, and mean|exec_w| = frac. If the new gate
   admits these at the old rate, it is a directionality gate and not a
   sparsity gate.
"""
import sys, io, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd
import cta
from cta.signals import SIGNAL_REGISTRY, VARIANT_REGISTRY

OUT = "/home/ubuntu/mtx/signal_zoo/qnt100"
TI  = A.index
_RET["night"]   = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"]  = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1
NIGHT_START = str(A["night_close"].first_valid_index().date())

# ══ D — the registered book ════════════════════════════════════════════════
print("=== D. registered signals, declared sign, under OLD vs NEW beta rule ===")
rows, skipped, built = [], set(), {}
for name, obj in sorted(SIGNAL_REGISTRY.items()):
    cls = type(obj)
    print("  building", name, flush=True)
    raw = obj.compute_raw(ctx).astype(float)
    s = raw if cls.pre_normalized else cta.normalize_signal(raw, method="tanh", window=252)
    s = (s * cls.sign).replace([np.inf, -np.inf], np.nan)
    built[name] = s
    # macro_window_sweep's harness carries c2c/o2o/day/ongap + the night leg
    # added above; `noonpause` has no return series here, so it is scored via
    # cta.batch_signal_stats below instead of being silently dropped.
    vs = [v for v in (cls.recommended_variants or cls.variants) if v in _RET]
    skipped.update(set(cls.recommended_variants or cls.variants) - set(vs))
    for v in vs:
        lag = cls.shift_override.get(v, VARIANT_REGISTRY[v].shift_days)
        keep, _SHIFT[v] = _SHIFT[v], lag
        try:
            st = wstats(s, v, start=(NIGHT_START if v in ("night", "ongap", "noonpause") else None),
                        sign=1)
        finally:
            _SHIFT[v] = keep
        if st:
            rows.append(dict(signal=name, variant=v, enabled=bool(cls.enabled),
                             rec=bool(cls.recommended_variants), **st))
D = pd.DataFrame(rows)
D["mean_abs_w"] = D["abs_exec_w"]
D = cta.house_gates(D, beta_mode="both")
D["passes_old"] = D.gate_srsr & D.gate_posyr & D.gate_nyr & D.gate_beta_raw
cols = ["signal","variant","enabled","SR_net","SR_of_SR","positive_years","n_years",
        "beta","abs_exec_w","beta_per_w","held_pct","passes_old","passes"]
print(D[cols].round(3).to_string(index=False))
dem = D[D.passes_old & ~D.passes]
print(f"\n  registered signal-variants: {len(D)}   "
      f"pass OLD {int(D.passes_old.sum())} -> pass NEW {int(D.passes.sum())}")
print("  DEMOTED by the fix:",
      list(dem[["signal","variant"]].itertuples(index=False, name=None)) if len(dem) else "none")
print(f"  min mean|exec_w| across the registered book: {D.abs_exec_w.min():.3f} "
      f"(so the fix's denominator is never near zero here)")
print("  variants with no return leg in this harness (scored c2c below):",
      sorted(skipped) or "none")
D.to_csv(f"{OUT}/registered_regated.csv", index=False)

# The same book through the shipped code path, which now emits beta_per_w itself.
print("\n=== D2. same book via cta.batch_signal_stats (c2c, exec_lag 2) ===")
B = cta.house_gates(cta.batch_signal_stats(built, A, auto_flip=False), beta_mode="both")
B["passes_old"] = B.gate_srsr & B.gate_posyr & B.gate_nyr & B.gate_beta_raw
print(B[["SR_net","SR_of_SR","positive_years","n_years","beta","mean_abs_w","beta_per_w",
         "held_pct","passes_old","passes"]].round(3).to_string())
print(f"  c2c view: pass OLD {int(B.passes_old.sum())} -> NEW {int(B.passes.sum())}; "
      f"mean|exec_w| range {B.mean_abs_w.min():.3f}-{B.mean_abs_w.max():.3f}")
B.to_csv(f"{OUT}/registered_regated_c2c.csv")

# ══ E — power against a sparse book with a real two-sided edge ═════════════
print("\n=== E. power: sparse but INFORMED two-sided masks (real edge, beta~0) ===")
rng = np.random.default_rng(11)
nr = _RET["night"].reindex(TI)
erows = []
for frac in (0.05, 0.10, 0.20):
    for p in (0.55, 0.60):
        for rep in range(60):
            n = max(1, int(frac * len(TI)))
            idx = np.sort(rng.choice(len(TI), n, replace=False))
            truth = np.sign(nr.iloc[idx].fillna(0).values)
            truth[truth == 0] = 1.0
            correct = rng.random(n) < p
            m = pd.Series(0.0, index=TI)
            m.iloc[idx] = np.where(correct, truth, -truth)
            st = wstats(m, "night", start="2010-01-01", sign=1)
            if st:
                erows.append(dict(frac=frac, p=p, rep=rep, **st))
E = pd.DataFrame(erows)
E["mean_abs_w"] = E["abs_exec_w"]
E = cta.house_gates(E, beta_mode="both")
E["passes_old"] = E.gate_srsr & E.gate_posyr & E.gate_nyr & E.gate_beta_raw
print(E.groupby(["frac","p"]).agg(SR_net=("SR_net","mean"), SR_of_SR=("SR_of_SR","mean"),
      beta=("beta","mean"), beta_per_w=("beta_per_w","mean"),
      pass_OLD=("passes_old","mean"), pass_NEW=("passes","mean")).round(3).to_string())
print(f"\n  overall informed: OLD {E.passes_old.mean()*100:.1f}% -> NEW {E.passes.mean()*100:.1f}%"
      f"   (power retained: {100*E.passes.sum()/max(E.passes_old.sum(),1):.0f}% of old passers)")
E.to_csv(f"{OUT}/power_informed_masks.csv", index=False)

json.dump(dict(reg_rows=len(D), reg_old=int(D.passes_old.sum()), reg_new=int(D.passes.sum()),
               demoted=[list(x) for x in dem[["signal","variant"]].values],
               reg_min_absw=float(D.abs_exec_w.min()),
               pow_old=float(E.passes_old.mean()), pow_new=float(E.passes.mean())),
          open(f"{OUT}/summary_d.json","w"), indent=1)
print("\nwrote", OUT)
