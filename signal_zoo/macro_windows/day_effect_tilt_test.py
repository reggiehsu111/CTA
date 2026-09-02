"""QNT-18 part 3c: the decisive test — is QNT-14's `day` gain ANYTHING but the tilt?

c2c chains as  day · noonpause · night · ongap, so to first order

    c2c_pnl[t] - day_pnl[t]  =  position[t] x overnight_ret[t]

i.e. the ONLY difference between the two variants is whether the position also
holds the overnight leg — and the overnight leg is where essentially all of the
TAIEX drift lives. A position with average tilt E[p] therefore carries a static
E[p]-sized bet on that drift, and switching to `day` removes it. That predicts:
DEMEAN the position and the day-vs-c2c difference should collapse to ~zero.

Demeaning uses an EXPANDING mean (min 252 obs), so the test is PIT-safe rather
than a full-sample-mean cheat; the full-sample version is reported alongside.

Diagnostic only. Nothing here proposes a tradeable construction.
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
pd.set_option("display.width", 200)

# ── the overnight leg, stated explicitly ──────────────────────────────────
_over = _o.shift(-1) / _c - 1                     # 13:45 t -> 08:45 t+1
for nm, r in [("c2c  (24h)", _RET["c2c"]), ("day  (08:45-13:45)", _RET["day"]),
              ("overnight (13:45 t -> 08:45 t+1)", _over)]:
    r = r.dropna()
    print(f"  B&H {nm:34s} SR {np.sqrt(252)*r.mean()/r.std():+.3f}   "
          f"mean {r.mean()*1e4:+6.2f} bps   ann.vol {r.std()*np.sqrt(252):.3f}   n={len(r)}")

SIGS = {}
for sid in DAILY:
    x = ctx.macro(sid).astype(float)
    for tn, tf in TRANSFORMS.items():
        for w in WINDOWS:
            s = cta.normalize_signal(tf(x, w).replace([np.inf, -np.inf], np.nan),
                                     method="tanh", window=252)
            if not s.dropna().empty:
                SIGS[f"{sid}|{tn}|w{w}"] = s
print(f"\ndaily-macro cells: {len(SIGS)}")

def demean(s, how):
    if how == "raw":      return s
    if how == "expand":   return s - s.expanding(252).mean()
    if how == "full":     return s - s.mean()

rows = []
for cand, s in SIGS.items():
    sgn = (wstats(s, "c2c", end="2018-12-31") or {}).get("sign")
    if sgn is None:
        continue
    for how in ("raw", "expand", "full"):
        d = demean(s, how)
        for v in ("c2c", "day"):
            st = wstats(d, v, sign=sgn)
            if st:
                rows.append(dict(cand=cand, how=how, variant=v, SR=st["SR_net"],
                                 srsr=st["SR_of_SR"], tilt=st["mean_exec_w"],
                                 beta=st["beta"]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/day_effect_tilt_test.csv", index=False)

print("\n=== paired ΔSR(day − c2c) on the SAME 198 cells, before and after removing the tilt ===")
print(f"{'position':28s} {'n':>4s} {'med tilt':>9s} {'medΔSR':>8s} {'meanΔSR':>8s} {'win':>7s} {'medΔSR_of_SR':>13s}")
for how, lab in [("raw", "raw (QNT-14's number)"), ("expand", "− expanding mean (PIT-safe)"),
                 ("full", "− full-sample mean")]:
    p = df[df.how == how].pivot_table(index="cand", columns="variant", values=["SR", "srsr", "tilt"])
    d  = (p["SR"]["day"] - p["SR"]["c2c"]).dropna()
    d2 = (p["srsr"]["day"] - p["srsr"]["c2c"]).dropna()
    print(f"{lab:28s} {len(d):4d} {p['tilt']['c2c'].abs().median():9.4f} "
          f"{d.median():+8.4f} {d.mean():+8.4f} {(d>0).mean():7.1%} {d2.median():+13.4f}")

print("\n=== four-gate passers, raw vs de-tilted (day, shift 1) ===")
for how in ("raw", "expand"):
    sub = df[(df.how == how) & (df.variant == "day")]
    print(f"  {how:7s} median SR_net {sub.SR.median():+.3f}  median SR_of_SR {sub.srsr.median():+.3f}  "
          f"median |beta| {sub.beta.abs().median():.3f}")
