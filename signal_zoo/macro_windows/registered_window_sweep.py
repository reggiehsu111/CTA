"""QNT-18 part 2: the 11 REGISTERED signals scored on `day` alongside every
declared variant, at each signal's own PIT-legal lag.

QNT-14's +0.071 median ΔSR for `day` over `c2c` was measured on 198 cells built
from 11 daily macro series. This asks whether it is a property of the execution
window (and so should show up on unrelated sources) or of those series.

Discipline:
  * the SIGN IS THE DECLARED, FROZEN `cls.sign`. Nothing here re-discovers a
    sign - auto_flip would guarantee a "win" on every cell.
  * normalization exactly as the runner does it: tanh(rolling_z(252)), skipped
    for `pre_normalized` classes.
  * lag = cls.shift_override.get(variant, VARIANT_REGISTRY[variant].shift_days).
    The five us_*/tv_* signals read ctx.us_index (pit_lag_days=1) so their
    o2o:1 override is legal - see the QNT-19 trap note in _base.py. No signal
    declares a `day` override, so `day` runs at its variant default of 1.
  * return legs re-derived here, roll-adjusted, and priced with REAL costs
    (fixed_per_side 70, fee_rate 4e-5) - reusing macro_window_sweep.py's setup
    block verbatim so `wstats` is identical to the QNT-14 evidence.
  * two regimes, mirroring QNT-14: `full` (c2c/o2o/day, full history) and
    `night` (all six variants, from the first night_close) so the three
    night-session windows are only ever compared on the window they exist on.
  * REPORT ONLY. No sign chosen, no recommended_variants chosen, no DB write.
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
# provides: ctx, A, _RET, _COST, _SHIFT, wstats, PV/FIXED/FEE, PPY

import numpy as np, pandas as pd
import cta
from cta.signals import SIGNAL_REGISTRY
from cta.signals._base import VARIANT_REGISTRY

OUT = os.environ.get("MTX_SWEEP_OUT", "/home/ubuntu/mtx/signal_zoo/macro_windows")
IS_END, OOS_START = "2023-12-31", "2024-01-01"   # see note below

# ── The two night-session legs, added to the harness ──────────────────────
# _base.py labelling, verified there against 1-minute MXF bars:
#     night_open[t] = 15:00 of t-1 ;  night_close[t] = 05:00 of t
# so a window that STARTS on day t uses the t+1 row (shift(-1)):
#     noonpause[t] 13:45 t -> 15:00 t   = night_open[t+1] / close[t]
#     night[t]     15:00 t -> 05:00 t+1 = night_close[t+1] / night_open[t+1]
_no1 = A["night_open"].astype(float).shift(-1)
_nc1 = A["night_close"].astype(float).shift(-1)
_roll1 = A.is_rollover.shift(-1).fillna(False).astype(bool)

# ROLL: night[] is same-row so intra-contract. noonpause[] is NOT - it divides
# a 15:00 print of the front-of-day-(t+1) contract by the 13:45 close of the
# front-of-day-t contract, so on the eve of a roll it books the calendar
# spread. Use yesterday's BACK close (= tomorrow's front) as the denominator
# on exactly those days. This leg is EXACT (both prints are closes, so the
# back-month close is the right price) - unlike the o2o approximation that
# QNT-52 retired above, which had to proxy an OPEN with a close-measured ratio.
# Identical to _base.py's _next_front_close.
_np_den = _c.where(~_roll1, _bc)
_RET["noonpause"] = _no1 / _np_den - 1
_RET["night"]     = _nc1 / _no1 - 1
_ENTRY["noonpause"], _ENTRY["night"] = _np_den, _no1
for k in ("noonpause", "night"):
    _COST[k] = FIXED / (_ENTRY[k] * PV) + FEE
_SHIFT["noonpause"] = VARIANT_REGISTRY["noonpause"].shift_days
_SHIFT["night"]     = VARIANT_REGISTRY["night"].shift_days

_nadj = int((_roll1 & _RET["noonpause"].notna()).sum())
_unadj = (_no1 / _c - 1)
print(f"noonpause roll-adjusted on {_nadj} days; "
      f"mean |spread| booked if unadjusted "
      f"{(_unadj - _RET['noonpause'])[_roll1].abs().mean()*1e4:.1f} bps")

# ── Build every registered signal exactly as the runner does ──────────────
sigs, sigmeta = {}, {}
for name, obj in sorted(SIGNAL_REGISTRY.items()):
    cls = type(obj)
    raw = obj.compute_raw(ctx).astype(float)
    s = raw if cls.pre_normalized else cta.normalize_signal(raw, method="tanh", window=252)
    sigs[name] = (s * cls.sign).replace([np.inf, -np.inf], np.nan)
    sigmeta[name] = dict(sign=cls.sign, enabled=bool(cls.enabled),
                         sources="+".join(cls.sources),
                         live_date=cls.live_date,
                         pre_normalized=bool(cls.pre_normalized),
                         variants="|".join(cls.variants),
                         rec_variants="|".join(cls.recommended_variants) or "-",
                         shift_override=str(cls.shift_override or "-"))
print(f"registered signals built: {len(sigs)}")

NIGHT_START = str(A["night_close"].first_valid_index().date())
# IS/OOS split: these classes were declared in 2026 on their own IS windows, so
# an IS/OOS split here cannot be a fresh out-of-sample test of the signal. It is
# reported only as a stability read on the WINDOW comparison, split late so both
# halves carry enough night-era bars.
REGIMES = [("full",  None,        ("c2c", "o2o", "day")),
           ("night", NIGHT_START, ("c2c", "o2o", "day", "noonpause", "night", "ongap"))]

rows = []
for reg, r_start, variants in REGIMES:
    for name, s in sigs.items():
        cls = type(SIGNAL_REGISTRY[name])
        for v in variants:
            lag = cls.shift_override.get(v, VARIANT_REGISTRY[v].shift_days)
            keep, _SHIFT[v] = _SHIFT[v], lag
            try:
                full = wstats(s, v, start=r_start, sign=1)    # sign already applied
                if full is None:
                    continue
                is_ = wstats(s, v, start=r_start, end=IS_END, sign=1)
                oos = wstats(s, v, start=OOS_START, sign=1)
                rows.append(dict(regime=reg, signal=name, variant=v, lag=lag,
                                 declared=v in cls.variants,
                                 recommended=v in cls.recommended_variants,
                                 **sigmeta[name],
                                 SR_IS=(is_ or {}).get("SR_net", np.nan),
                                 SR_OOS=(oos or {}).get("SR_net", np.nan),
                                 **{k: full[k] for k in
                                    ("SR_net", "SR_gross", "SR_of_SR", "positive_years",
                                     "yr_sr_min", "n_years", "beta", "mean_exec_w",
                                     "abs_exec_w", "max_dd_pct", "max_dd_days",
                                     "turnover_ann", "held_pct", "n_bars",
                                     "start_date", "end_date")}))
            finally:
                _SHIFT[v] = keep

df = pd.DataFrame(rows)
df["gate_srsr"]  = df.SR_of_SR > 0.6
df["gate_posyr"] = df.positive_years >= 0.65
df["gate_beta"]  = df.beta.abs() < 0.15
df["gate_nyr"]   = df.n_years >= 5
df["n_gates"] = df[["gate_srsr", "gate_posyr", "gate_beta", "gate_nyr"]].sum(axis=1)
df.to_csv(f"{OUT}/registered_window_sweep.csv", index=False)
print(f"rows: {len(df)} -> registered_window_sweep.csv")

pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 200)

print("\n=== buy-and-hold reference, same costs/regimes (gross, roll-adjusted) ===")
for reg, r_start, variants in REGIMES:
    for v in variants:
        r = _RET[v].loc[r_start:].dropna()
        print(f"  {reg:5s} {v:10s} SR {np.sqrt(PPY)*r.mean()/r.std():+.3f}  "
              f"ann.vol {r.std()*np.sqrt(PPY):.3f}  n={len(r)}")

for reg in ("full", "night"):
    d = df[df.regime == reg]
    print(f"\n{'='*100}\n=== REGIME {reg}  (start {d.start_date.min()}) ===")
    print(d.pivot_table(index="signal", columns="variant", values="SR_net").round(3).to_string())
    print("\n  -- paired vs c2c (same signal, same window) --")
    piv = d.pivot_table(index="signal", columns="variant",
                        values=["SR_net", "SR_of_SR", "beta", "positive_years", "mean_exec_w"])
    for v in [x for x in piv["SR_net"].columns if x != "c2c"]:
        for m in ["SR_net", "SR_of_SR"]:
            z = (piv[m][v] - piv[m]["c2c"]).dropna()
            print(f"   {v:10s} Δ{m:9s} median {z.median():+.4f}  mean {z.mean():+.4f}  "
                  f"win {(z>0).mean():.0%} ({int((z>0).sum())}/{len(z)})")
        print(f"   {v:10s} |beta| med {piv['beta'][v].abs().median():.3f} "
              f"(c2c {piv['beta']['c2c'].abs().median():.3f})   "
              f"|mean_exec_w| med {piv['mean_exec_w'][v].abs().median():.3f} "
              f"(c2c {piv['mean_exec_w']['c2c'].abs().median():.3f})")
    print("\n  -- median by variant --")
    print(d.groupby("variant")[["SR_net", "SR_of_SR", "positive_years", "n_years",
                                "beta", "mean_exec_w", "turnover_ann"]]
           .median().round(3).to_string())
    print("\n  -- four-gate passers --")
    print(d.groupby("variant")["n_gates"].agg(
        n="size", pass4=lambda s: int((s == 4).sum()), mean="mean").round(2).to_string())

print(f"\n{'='*100}\n=== FULL DETAIL, regime=full, every gate + mean_exec_w + beta ===")
cols = ["signal", "variant", "lag", "SR_net", "SR_of_SR", "positive_years", "n_years",
        "beta", "mean_exec_w", "turnover_ann", "n_gates", "SR_IS", "SR_OOS",
        "enabled", "recommended", "n_bars"]
print(df[df.regime == "full"].sort_values(["signal", "variant"])[cols].round(3).to_string(index=False))

print(f"\n=== FULL DETAIL, regime=night ===")
print(df[df.regime == "night"].sort_values(["signal", "variant"])[cols].round(3).to_string(index=False))

print("\n=== every cell passing all four house gates ===")
allg = df[df.n_gates == 4]
print(f"{len(allg)} of {len(df)}")
if len(allg):
    print(allg[["regime"] + cols].round(3).to_string(index=False))


# ── QNT-32 / QNT-25 reporting line ────────────────────────────────────────
# On this grid a cell IS a source series (11 registered signals, one row per
# signal x variant), so the n was already honest — the headline says so rather
# than silently inflating it. What still needs printing is the noise floor:
# sd(SR_c2c) across the 11 signals ~ 0.26 against SE(SR|25y) ~ 0.21.
print(f"\n{'='*100}\n=== QNT-25 REPORTING LINE ===")
for _reg in df.regime.unique():
    _d = df[df.regime == _reg]
    _ny = float(_d.n_years.median())
    _p = _d.pivot_table(index="signal", columns="variant", values="SR_net")
    if "c2c" not in _p.columns:
        continue
    cta.sweep_headline(_p.reset_index(), "c2c", series_col="signal", n_years=_ny,
                       label=f"[{_reg}] registered-signal window sweep, SR_net on c2c").print()
    for _v in [c for c in _p.columns if c != "c2c"]:
        _q = _p.dropna(subset=[_v, "c2c"])
        cta.paired_headline(_q[_v], _q["c2c"], n_years=_ny,
                            label=f"[{_reg}] registered-signal window sweep",
                            a_name=_v, b_name="c2c").print()
