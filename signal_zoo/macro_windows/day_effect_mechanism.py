"""QNT-18 part 3b: two checks on QNT-14's +0.071.

(a) DRIFT ARITHMETIC. day = c2c minus the overnight leg, and the overnight leg
    carries nearly all the TAIEX drift (day B&H SR 0.230 vs c2c 0.700). So a
    position with a NET LONG tilt must lose when moved to `day`, and a net short
    tilt must gain. Test against mean_exec_w, which is the tilt itself and needs
    no regression.

(b) SIGN-REFIT CONFOUND. QNT-14 (and part 1/2 here) pick the sign on the IS half
    SEPARATELY FOR EACH VARIANT. If `day` and `c2c` disagree on sign, part of
    "day beats c2c" is just a second sign search, not a window effect. Re-run the
    daily-macro grid with the sign frozen on the c2c IS half for BOTH variants
    and see how much of the +0.071 survives.
"""
import sys, warnings, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd

OUT = "/home/ubuntu/mtx/signal_zoo/macro_windows"
pd.set_option("display.width", 200)

# ── (a) drift arithmetic, on all three grids ──────────────────────────────
def paired(path, key, query=None):
    d = pd.read_csv(f"{OUT}/{path}")
    if query: d = d.query(query)
    p = d.pivot_table(index=key, columns="variant",
                      values=["SR_net", "mean_exec_w", "beta", "sign_IS"] if "sign_IS" in d
                      else ["SR_net", "mean_exec_w", "beta"])
    o = pd.DataFrame({"dSR": p["SR_net"]["day"] - p["SR_net"]["c2c"],
                      "tilt_c2c": p["mean_exec_w"]["c2c"],
                      "tilt_day": p["mean_exec_w"]["day"]})
    if "sign_IS" in p:
        o["sign_agree"] = (p["sign_IS"]["day"] == p["sign_IS"]["c2c"])
    return o.dropna(subset=["dSR"])

GR = {"daily macro":  paired("window_sweep_full.csv", "cand", "regime=='full'"),
      "slow macro":   paired("slow_window_sweep.csv", "cand"),
      "registered":   paired("registered_window_sweep.csv", "signal", "regime=='full'")}

print("=== (a) ΔSR(day−c2c) split by the sign of the position's net tilt ===")
print(f"{'grid':14s} {'net-LONG tilt (mean_exec_w>0)':>34s}   {'net-SHORT tilt':>28s}")
for g, d in GR.items():
    L, S = d[d.tilt_c2c > 0], d[d.tilt_c2c < 0]
    print(f"{g:14s}  n={len(L):4d} medΔSR {L.dSR.median():+.3f} win {(L.dSR>0).mean():5.1%}"
          f"      n={len(S):4d} medΔSR {S.dSR.median():+.3f} win {(S.dSR>0).mean():5.1%}")
ALL = pd.concat(GR.values())
L, S = ALL[ALL.tilt_c2c > 0], ALL[ALL.tilt_c2c < 0]
print(f"{'POOLED':14s}  n={len(L):4d} medΔSR {L.dSR.median():+.3f} win {(L.dSR>0).mean():5.1%}"
      f"      n={len(S):4d} medΔSR {S.dSR.median():+.3f} win {(S.dSR>0).mean():5.1%}")
print(f"  corr(ΔSR, tilt_c2c) pooled = {np.corrcoef(ALL.dSR, ALL.tilt_c2c)[0,1]:+.3f}")

print("\n=== sign agreement between the day and c2c IS fits ===")
for g, d in GR.items():
    if "sign_agree" in d:
        print(f"  {g:14s} agree {d.sign_agree.mean():5.1%} ({int(d.sign_agree.sum())}/{len(d)})   "
              f"medΔSR when agree {d[d.sign_agree].dSR.median():+.3f} | "
              f"when disagree {d[~d.sign_agree].dSR.median():+.3f}")

# ── (b) re-run the daily-macro grid with ONE sign, frozen on c2c IS ────────
SWEEP = f"{OUT}/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))   # noqa: S102
import cta

SIGS = {}
for sid in DAILY:
    x = ctx.macro(sid).astype(float)
    for tn, tf in TRANSFORMS.items():
        for w in WINDOWS:
            s = cta.normalize_signal(tf(x, w).replace([np.inf, -np.inf], np.nan),
                                     method="tanh", window=252)
            if not s.dropna().empty:
                SIGS[f"{sid}|{tn}|w{w}"] = s
print(f"\n=== (b) daily-macro grid rebuilt: {len(SIGS)} cells ===")

rows = []
for cand, s in SIGS.items():
    ref = wstats(s, "c2c", end="2018-12-31")          # ONE sign, from c2c IS
    if ref is None:
        continue
    sgn = ref["sign"]
    for v, per_variant_sign in (("c2c", None), ("day", None)):
        own = wstats(s, v, end="2018-12-31")
        frz = wstats(s, v, sign=sgn)
        fre = wstats(s, v, sign=(own or {}).get("sign", sgn))
        if frz is None or fre is None:
            continue
        rows.append(dict(cand=cand, variant=v, SR_frozen=frz["SR_net"],
                         SR_ownsign=fre["SR_net"], beta_frozen=frz["beta"],
                         tilt_frozen=frz["mean_exec_w"],
                         srsr_frozen=frz["SR_of_SR"], srsr_own=fre["SR_of_SR"]))
z = pd.DataFrame(rows).pivot_table(index="cand", columns="variant",
                                   values=["SR_frozen", "SR_ownsign", "srsr_frozen",
                                           "srsr_own", "tilt_frozen"])
for tag, a, b in [("per-variant IS sign (QNT-14's method)", "SR_ownsign", "srsr_own"),
                  ("ONE sign, frozen on the c2c IS fit  ", "SR_frozen", "srsr_frozen")]:
    d  = (z[a]["day"] - z[a]["c2c"]).dropna()
    d2 = (z[b]["day"] - z[b]["c2c"]).dropna()
    print(f"  {tag}:  medΔSR {d.median():+.4f}  win {(d>0).mean():5.1%}  n={len(d)}   "
          f"medΔSR_of_SR {d2.median():+.4f}")
t = z["tilt_frozen"]["c2c"]
d = (z["SR_frozen"]["day"] - z["SR_frozen"]["c2c"]).dropna()
print(f"  under the ONE frozen sign, split by tilt:  long n={int((t>0).sum())} "
      f"medΔSR {d[t>0].median():+.3f} | short n={int((t<0).sum())} medΔSR {d[t<0].median():+.3f}")
