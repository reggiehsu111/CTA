"""QNT-32 regression check for `cta.sweep_headline` / `cta.paired_headline`.

Re-runs NO sweep. It reads the CSVs already on disk from QNT-12/14/18, asserts the
helper reproduces the numbers QNT-25 published by hand (`qnt25_perseries.py` /
`qnt25_report.txt`), and then dry-runs the reporting block that QNT-32 appended to
each of the four sweep harnesses — so a change to the helper or to a harness's
tail is caught here rather than in the middle of a five-hour sweep.

    python3 signal_zoo/qnt32_verify.py

REFERENCE ERAS — read this before "fixing" a FAILURE
----------------------------------------------------
A constant here is only meaningful next to the CSV era it was frozen on. QNT-82
re-derived grids 2/3/4 after three patches landed on 2026-09-01 and moved the
sweeps; QNT-92 then repointed grid 1 at its post-floor CSV, so all four are now
one era. The QNT-25 originals are kept commented beside each block, because the
QNT-14 -> QNT-19 move on grid 2 IS the evidence that the day-window effect was a
PIT artefact and deleting it loses that.

  grid  CSV                                 era of the constants below
  ----  ----------------------------------  --------------------------------
   1    qnt19_postfloor/full_sweep.csv      2026-09-01, post QNT-19 (+21/+52 are
                                            no-ops on this grid: it never touches
                                            an o2o leg and passes roll_adjusted
                                            explicitly). QNT-92 repointed this
                                            from macro_sweep/full_sweep.csv,
                                            which stays PRE-floor on purpose —
                                            it is the "before" side of
                                            compare_prefloor.py. See grid 1.
   2    macro_windows/window_sweep_full.csv 2026-09-01, post QNT-19 + 21 + 52
   3    macro_windows/slow_window_sweep.csv 2026-09-01, post QNT-19 + 21 + 52
   4    macro_windows/registered_...csv     2026-09-01, post QNT-19 + 21 + 52

The three patches:
  QNT-19  `load_macro_tw` floors US-close daily obs at +1 calendar day (PIT).
  QNT-21  `A.returns` is roll-adjusted, so the c2c leg no longer books the
          calendar spread.
  QNT-52  the o2o leg uses `_base`'s exact open-to-open return, not the retired
          approximation.

Re-derive with `signal_zoo/macro_windows/qnt82/rederive_refs.py`, which prints
every constant below straight off the CSVs on disk. Date-stamp what you paste.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta

Z = "/home/ubuntu/mtx/signal_zoo"
TOL = 5e-3
fails = []
_era = "ref"          # set per grid so a FAILURE names the era it drifted from

def check(name, got, want, tol=TOL):
    ok = np.isfinite(got) and abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'} {name:44s} got {got:+.4f}  {_era} {want:+.4f}")
    if not ok:
        fails.append(name)


def grid(title, csv):
    """Print the grid header WITH the CSV's mtime, so era drift is visible."""
    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(csv)))
    print(f"{title}\n       {csv.rsplit('/', 2)[-2]}/{csv.rsplit('/', 1)[-1]}  (mtime {ts})")
    return pd.read_csv(csv)

# ── Grid 1: QNT-12 standalone macro sweep ─────────────────────────────────
# QNT-92 repointed this block from `macro_sweep/full_sweep.csv` (pre-floor) to
# the post-floor copy, so all four grids are now one era. Two reasons it moved
# rather than the CSV being re-generated in place:
#   * `macro_sweep/full_sweep.csv` is the PRE-floor side of QNT-19's evidence —
#     `qnt19_postfloor/compare_prefloor.py` diffs the two. Overwriting it makes
#     that comparison a no-op and deletes the evidence, the same mistake
#     QNT-82 avoided on grid 2.
#   * it is also the published QNT-12/QNT-25 baseline several tickets quote.
# QNT-92 re-ran `macro_sweep.py` verbatim into a scratch dir (never over the
# published path) to confirm the post-floor CSV is what current code produces:
# SR_IS bit-identical on 522/522 cells, every other column within 0.034 (one
# extra trading day, 2026-09-01). See `qnt92_scratch/QNT92_NOTE.md`.
_era = "QNT-19"
g1 = grid("GRID 1 — QNT-12 standalone macro sweep",
          f"{Z}/qnt19_postfloor/full_sweep.csv")
s = cta.sweep_headline(g1, "SR_full", label="g1").stats
check("ICC(series) on SR_full", s["icc"], 0.622)          # QNT-25 pre-floor: 0.615
check("n_eff on SR_full", s["n_eff"], 45.1, tol=0.1)      # QNT-25 pre-floor: 45.6
check("per-series median SR_full", s["per_series_median"], 0.1050)  # was 0.0890
check("n positive series", s["n_positive"], 27, tol=0)    # QNT-25 pre-floor: 26
check("sd(SR_full) across cells", s["sd_cells"], 0.136)   # QNT-25 pre-floor: 0.139
check("SE(SR | 26y)", s["se_sr"], 0.200)                  # unchanged
s2 = cta.sweep_headline(g1, "SR_of_SR", label="g1srsr").stats
check("ICC(series) on SR_of_SR", s2["icc"], 0.487)        # QNT-25 pre-floor: 0.489
check("n_eff on SR_of_SR", s2["n_eff"], 56.3, tol=0.1)    # QNT-25 pre-floor: 56.0
check("sd(SR_of_SR) across cells", s2["sd_cells"], 0.151) # QNT-25 pre-floor: 0.147
check("best cell SR_of_SR", s2["best_cell"], 0.577)       # unchanged, igrea is monthly
check("expected max at n_eff", s2["expected_max_null"], 0.428)      # was 0.417

# ── Grid 2: QNT-14 daily-macro window sweep (paired) ──────────────────────
_era = "2026-09-01"
g2 = grid("GRID 2 — QNT-14 daily-macro window sweep (paired day - c2c)",
          f"{Z}/macro_windows/window_sweep_full.csv")
g2 = g2[g2.regime == "full"]
p2 = g2.pivot_table(index=["series", "transform", "window"],
                    columns="variant", values="SR_net").reset_index()
s = cta.paired_headline(p2["day"], p2["c2c"], series=p2["series"],
                        n_years=g2.n_years.median()).stats
# refs frozen 2026-09-01, post QNT-19 PIT floor + QNT-21 roll fix + QNT-52 o2o exact
check("ICC(series) on dSR", s["icc"], 0.692)
check("n_eff on dSR", s["n_eff"], 15.5, tol=0.1)
check("per-CELL median dSR", s["cell_median"], 0.0136)
check("per-SERIES median dSR", s["per_series_median"], 0.0387)
check("n positive series", s["n_positive"], 6, tol=0)
check("per-series Wilcoxon p", s["wilcoxon_p"], 0.638)
check("corr(day, c2c)", s["rho"], 0.153)
check("SE of the paired dSR", s["se_delta"], 0.261)
# QNT-25 era (pre 2026-09-01), kept as evidence — the collapse from +0.0708 to
# +0.0136 per cell and rho 0.415 -> 0.153 IS the QNT-18/19 finding that the
# day-window effect was a PIT artefact. Do not delete; re-derive alongside.
#     check("ICC(series) on dSR", s["icc"], 0.495)
#     check("n_eff on dSR", s["n_eff"], 21.0, tol=0.1)
#     check("per-CELL median dSR", s["cell_median"], 0.0708)
#     check("per-SERIES median dSR", s["per_series_median"], 0.0890)
#     check("n positive series", s["n_positive"], 8, tol=0)
#     check("per-series Wilcoxon p", s["wilcoxon_p"], 0.102)
#     check("corr(day, c2c)", s["rho"], 0.415)
#     check("SE of the paired dSR", s["se_delta"], 0.217)

# ── Grid 3: QNT-18 slow-macro window sweep (paired) ───────────────────────
_era = "2026-09-01"
g3 = grid("GRID 3 — QNT-18 slow-macro window sweep (paired day@2 - c2c@2)",
          f"{Z}/macro_windows/slow_window_sweep.csv")
p3 = g3.pivot_table(index=["series", "transform", "window"],
                    columns="variant", values="SR_net").reset_index()
s = cta.paired_headline(p3["day"], p3["c2c"], series=p3["series"],
                        n_years=g3.n_years.median()).stats
# refs frozen 2026-09-01, post QNT-19 PIT floor + QNT-21 roll fix + QNT-52 o2o exact.
# The slow grid holds only NON-daily macro inputs, which the QNT-19 floor does not
# touch (QNT-19 verified 0/1512 cells moved); what moved these is QNT-21 + QNT-52,
# and it moved them by ~0.002 — the verdict is unchanged.
check("ICC(series) on dSR", s["icc"], 0.520)
check("n_eff on dSR", s["n_eff"], 38.4, tol=0.1)
check("per-CELL median dSR", s["cell_median"], -0.0338)
check("per-SERIES median dSR", s["per_series_median"], -0.0105)
check("per-series Wilcoxon p", s["wilcoxon_p"], 0.257)
# QNT-25 era (pre 2026-09-01), kept as evidence:
#     check("ICC(series) on dSR", s["icc"], 0.525)
#     check("n_eff on dSR", s["n_eff"], 38.1, tol=0.1)
#     check("per-CELL median dSR", s["cell_median"], -0.0322)
#     check("per-SERIES median dSR", s["per_series_median"], -0.0108)
#     check("per-series Wilcoxon p", s["wilcoxon_p"], 0.243)

# ── Grid 4: QNT-18 registered sweep — a cell IS a series ──────────────────
_era = "2026-09-01"
g4 = grid("GRID 4 — QNT-18 registered sweep (a cell IS a series)",
          f"{Z}/macro_windows/registered_window_sweep.csv")
g4 = g4[g4.regime == "full"]
p4 = g4.pivot_table(index="signal", columns="variant",
                    values="SR_net").dropna(subset=["c2c", "day"])
s = cta.paired_headline(p4["day"], p4["c2c"], n_years=g4.n_years.median()).stats
# refs frozen 2026-09-01, post QNT-19 PIT floor + QNT-21 roll fix + QNT-52 o2o exact
check("per-signal median dSR", s["per_series_median"], -0.0881)
check("n positive signals", s["n_positive"], 4, tol=0)
check("Wilcoxon p", s["wilcoxon_p"], 0.175)
check("n_eff (cell IS a series)", s["n_eff"], 11.0, tol=0)
s = cta.sweep_headline(p4.reset_index(), "c2c", series_col="signal",
                       n_years=g4.n_years.median(), label="g4").stats
check("sd(SR_c2c) across signals", s["sd_cells"], 0.261)
check("SE(SR | 25y)", s["se_sr"], 0.210)
check("ratio sd/SE", s["ratio"], 1.242)
# QNT-25 era (pre 2026-09-01), kept as evidence — this grid moved only
# marginally, which is itself the point: the registered signals are not the
# ones the PIT floor was biting.
#     check("per-signal median dSR", s["per_series_median"], -0.0894)
#     check("Wilcoxon p", s["wilcoxon_p"], 0.147)
#     check("sd(SR_c2c) across signals", s["sd_cells"], 0.257)
#     check("ratio sd/SE", s["ratio"], 1.23)

# ── Degenerate inputs must not silently mis-state ─────────────────────────
print("EDGE CASES")
for label, hl, want in [
    ("all-NaN grid returns a message",
     cta.sweep_headline(pd.DataFrame({"SR_net": [np.nan] * 3, "series": list("abc")})),
     "nothing to report"),
    ("single-series grid says ICC n/a",
     cta.sweep_headline(pd.DataFrame({"SR_net": [0.1, 0.2], "series": ["a", "a"]})),
     "ICC n/a"),
    ("no series column is declared",
     cta.sweep_headline(g1.drop(columns=["series"]), "SR_full"),
     "a cell IS a series"),
]:
    ok = want in hl.detail()
    print(f"  {'ok ' if ok else 'FAIL'} {label}")
    if not ok:
        fails.append(label)

# ── The reporting block appended to each harness must still execute ───────
print("HARNESS BLOCKS (dry-run against the CSV each harness writes)")
MARK = "# ── QNT-32 / QNT-25 reporting line"
REGIMES = [("full", None, "2018-12-31", "2019-01-01", ("c2c", "o2o", "day")),
           ("night", "2017-05-16", "2021-12-31", "2022-01-01", ("c2c", "o2o", "day", "ongap"))]
for script, csv, extra in [
    (f"{Z}/macro_sweep/macro_sweep.py",               f"{Z}/macro_sweep/full_sweep.csv", {}),
    (f"{Z}/macro_windows/macro_window_sweep.py",      f"{Z}/macro_windows/window_sweep_full.csv", {"REGIMES": REGIMES}),
    (f"{Z}/macro_windows/slow_window_sweep.py",       f"{Z}/macro_windows/slow_window_sweep.csv", {}),
    (f"{Z}/macro_windows/registered_window_sweep.py", f"{Z}/macro_windows/registered_window_sweep.csv", {}),
]:
    name = script.rsplit("/", 1)[1]
    src = open(script).read()
    if src.count(MARK) != 1:
        print(f"  FAIL {name}: QNT-32 reporting block missing or duplicated")
        fails.append(name); continue
    body = src.split(MARK, 1)[1].split("\n", 1)[1]
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(body, script, "exec"),
                 {"cta": cta, "pd": pd, "np": np, "df": pd.read_csv(csv), **extra})
        lines = [l for l in buf.getvalue().splitlines() if "cells =" in l]
        print(f"  ok   {name}: block ran, {len(lines)} headline line(s)")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        fails.append(name)

print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
