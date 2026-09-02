"""QNT-19: pre-floor vs post-floor comparison of every published macro number.

Reggie approved the +1 calendar-day PIT floor on `cta.load_macro_tw` (QNT-19,
2026-09-01). This re-runs nothing — it diffs the pre-floor artifacts against the
post-floor re-runs in this directory, which were produced by executing each
published script verbatim with only its `OUT` path redirected here. Nothing in
the pre-floor directories is overwritten.

QNT-92 FIX — the pre-floor paths are now PINNED, not read from the live harness
output dirs. The original version read `macro_windows/window_sweep_full.csv` and
`macro_windows/slow_window_sweep.csv`, which QNT-52 (10:06) and QNT-82 (10:34)
then re-ran IN PLACE. From that moment this script silently compared eras rather
than the floor, and running it overwrote the artifact with a wrong diff. It is
pinned to the copies those tickets archived before re-running; both are verified
to reproduce the QNT19_FOOTNOTE.md numbers exactly (grid 2 pre-floor per-cell
dSR +0.0708, 64% win, corr 0.415). Grid 1 and the combo scoreboard were never
re-run, so they still point at their published paths.

Writes qnt19_prefloor_vs_postfloor.csv (one row per cell that moved) and prints
the footnote tables.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = "/home/ubuntu/mtx/signal_zoo"
POST = f"{ROOT}/qnt19_postfloor"
# QNT-92: pinned pre-floor sources. Do NOT swap these back to the live harness
# output dirs — those are re-run in place and no longer hold the pre-floor grid.
PRE_G2 = f"{ROOT}/macro_windows/qnt52/prior_run/window_sweep_full.csv"   # archived by QNT-52
PRE_G3 = f"{ROOT}/macro_windows/qnt82/pre_rerun/slow_window_sweep.csv"   # archived by QNT-82
pd.set_option("display.width", 200)

out_rows = []


def _emit(grid, key_cols, pre, post, srcol):
    m = pre.merge(post, on=key_cols, suffixes=("_pre", "_post"))
    m = m[m[f"{srcol}_pre"].notna() & m[f"{srcol}_post"].notna()].copy()
    m["dSR"] = m[f"{srcol}_post"] - m[f"{srcol}_pre"]
    moved = m[m.dSR.abs() > 1e-9]
    for _, r in moved.iterrows():
        out_rows.append(dict(grid=grid, cell=" | ".join(str(r[c]) for c in key_cols),
                             SR_pre=r[f"{srcol}_pre"], SR_post=r[f"{srcol}_post"],
                             dSR=r["dSR"]))
    return m


# ── QNT-12 / QNT-10: the 522-cell macro sweep ──────────────────────────────
pre = pd.read_csv(f"{ROOT}/macro_sweep/full_sweep.csv")
post = pd.read_csv(f"{POST}/full_sweep.csv")
m = _emit("QNT-12 522-cell sweep", ["cand"], pre, post, "SR_full")
print("=== QNT-12 / QNT-10 — 522-cell macro sweep ===")
print(f"  cells {len(m)}; moved {int((m.dSR.abs()>1e-9).sum())} "
      f"({m[m.dSR.abs()>1e-9].series_pre.nunique()} of {m.series_pre.nunique()} source series)")
print(f"  ΔSR_full median {m.dSR.median():+.3f}  mean {m.dSR.mean():+.3f}  max|Δ| {m.dSR.abs().max():.3f}"
      f"  IS-sign flips {int((m.sign_IS_post != m.sign_IS_pre).sum())}")
for tag in ("pre", "post"):
    q = m[[f"SR_of_SR_{tag}", f"positive_years_{tag}", f"beta_{tag}",
           f"n_years_{tag}", f"SR_OOS_{tag}", f"SR_full_{tag}"]]
    q.columns = ["srsr", "posyr", "beta", "nyr", "oos", "sr"]
    g5 = int(((q.srsr > .6) & (q.posyr >= .65) & (q.beta.abs() < .15) & (q.nyr >= 5) & (q.oos > 0)).sum())
    print(f"  {tag:4s}  5-gate passers {g5}   gate_srsr {100*(q.srsr>.6).mean():.1f}%   "
          f"best SR_of_SR {q.srsr.max():.3f}   best SR {q.sr.max():.3f}")
print("  per-series Δ (only series that moved):")
print(m[m.dSR.abs() > 1e-9].groupby("series_pre").agg(
    n=("dSR", "size"), median=("dSR", "median"),
    max_abs=("dSR", lambda x: x.abs().max())).round(3).to_string())

# ── QNT-14: the 198-cell daily-macro window grid ───────────────────────────
pre = pd.read_csv(PRE_G2)
post = pd.read_csv(f"{POST}/window_sweep_full.csv")
m = _emit("QNT-14 window grid", ["regime", "cand", "variant"], pre, post, "SR_net")


def _pass4(d, tag):
    return int(((d[f"SR_of_SR_{tag}"] > .6) & (d[f"positive_years_{tag}"] >= .65)
                & (d[f"beta_{tag}"].abs() < .15) & (d[f"n_years_{tag}"] >= 5)).sum())


print("\n=== QNT-14 — 198-cell daily-macro window grid ===")
for reg in ("full", "night"):
    q = m[m.regime == reg]
    print(f"  regime={reg}")
    for v, d in q.groupby("variant"):
        print(f"    {v:6s} n={len(d):3d}  ΔSR median {d.dSR.median():+.3f}  win {100*(d.dSR>0).mean():5.1f}%"
              f"  max|Δ| {d.dSR.abs().max():.3f}  sign-flips {int((d.sign_IS_post!=d.sign_IS_pre).sum()):3d}"
              f"  4-gate {_pass4(d,'pre')} -> {_pass4(d,'post')}")
q = m[m.regime == "full"]
for tag in ("pre", "post"):
    p = q.pivot_table(index="cand", columns="variant", values=f"SR_net_{tag}").dropna()
    dd = p["day"] - p["c2c"]
    print(f"  QNT-14 headline day−c2c ({tag}-floor): n={len(dd)} median {dd.median():+.3f} "
          f"win {100*(dd>0).mean():.1f}%")

# ── QNT-18: the 378-cell slow-input grid ───────────────────────────────────
pre = pd.read_csv(PRE_G3)
post = pd.read_csv(f"{POST}/slow_window_sweep.csv")
m = _emit("QNT-18 slow grid", ["cand", "variant"], pre, post, "SR_net")
print("\n=== QNT-18 — 378-cell slow-input grid ===")
print(f"  cells {len(m)}; moved {int((m.dSR.abs()>1e-9).sum())}  "
      f"(all 21 slow inputs carry pub_lag_days >= 7, so the floor never binds)")

# ── QNT-16: the EW3 macro combination ──────────────────────────────────────
pre = pd.read_csv(f"{ROOT}/macro_combo/combo_scoreboard.csv")
post = pd.read_csv(f"{POST}/combo_scoreboard.csv")
kc = "combo" if "combo" in pre.columns else pre.columns[0]
m = pre.merge(post, on=kc, suffixes=("_pre", "_post"))
print("\n=== QNT-16 — macro combination scoreboard ===")
cols = [c for c in ("SR_full", "SR_of_SR", "positive_years", "beta") if f"{c}_pre" in m.columns]
for _, r in m.iterrows():
    delta = "  ".join(f"{c} {r[f'{c}_pre']:.3f}->{r[f'{c}_post']:.3f}" for c in cols)
    print(f"  {r[kc]:32s} {delta}")
    for c in cols:
        if abs(r[f"{c}_pre"] - r[f"{c}_post"]) > 1e-9:
            out_rows.append(dict(grid="QNT-16 combo", cell=f"{r[kc]}|{c}",
                                 SR_pre=r[f"{c}_pre"], SR_post=r[f"{c}_post"],
                                 dSR=r[f"{c}_post"] - r[f"{c}_pre"]))

d = pd.DataFrame(out_rows)
d.to_csv(f"{POST}/qnt19_prefloor_vs_postfloor.csv", index=False)
print(f"\nrows that moved: {len(d)} -> qnt19_prefloor_vs_postfloor.csv")
