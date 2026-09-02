# QNT-92 — is `macro_sweep/full_sweep.csv` (grid 1) safe to leave pre-floor?

**Decision: do NOT re-run `macro_sweep.py` over the published path. Repoint
`qnt32_verify.py`'s GRID 1 at `qnt19_postfloor/full_sweep.csv` instead.**
All four grids in the regression file are now post-QNT-19 and nothing published
was overwritten.

## Why not overwrite

`macro_sweep/full_sweep.csv` is not only the published QNT-12/QNT-25 baseline —
it is the **pre-floor side of QNT-19's own evidence**. `compare_prefloor.py`
diffs it against `qnt19_postfloor/full_sweep.csv` to produce
`qnt19_prefloor_vs_postfloor.csv` (1,530 moved cells). Regenerating it in place
turns that comparison into a no-op and deletes the evidence — exactly the
mistake QNT-82 avoided on grid 2 ("the QNT-14 -> QNT-19 move IS the evidence").

## Evidence that the post-floor CSV is era-current

`macro_sweep.py` was re-run **verbatim** with only `OUT` redirected here
(`/tmp/qnt92_macro_sweep_snap.py`, byte-identical to the published script apart
from that one line). Runtime ~7 min. Published path never written.

| grid-1 CSV | what it is |
|---|---|
| A `macro_sweep/full_sweep.csv` | pre-floor, 2026-09-01 03:13, data to 2026-08-31 |
| B `qnt19_postfloor/full_sweep.csv` | post-floor, 2026-09-01 07:12, data to 2026-08-31 |
| C `qnt92_scratch/full_sweep.csv` | current code, 2026-09-02, data to **2026-09-01** |

    A -> B   SR_full  144/522 moved, median +0.0000, mean +0.0055, max|d| 0.242, 15 sign_IS flips
    B -> C   SR_IS    0/522 moved (bit-identical); SR_full max|d| 0.010, SR_OOS max|d| 0.034, 0 sign flips
    A -> C   SR_full  median +0.0015, max|d| 0.238, 15 sign_IS flips

A->B reproduces the QNT-19 footnote exactly (144 of 522, 15 flips). B->C is the
one extra trading day and nothing else: `SR_IS` is bit-identical because the IS
half ends 2018-12-31. So **no code change after 2026-09-01 07:12 moves this
grid** — the 10:21 `asia_macro.py`, 14:30 `global_macro.py`/`_base.py` and 14:42
`_sources.py` edits are all no-ops here.

QNT-21 and QNT-52 are structurally no-ops on grid 1: `macro_sweep.py` passes
`roll_adjusted=True` explicitly to `cta.signal_stats` and never touches a
variant leg (no `o2o`, no `day`). The floor is the whole difference.

## Conclusions are unchanged on every version

| | A pre-floor | B post-floor | C current |
|---|---|---|---|
| ICC(series) SR_full | 0.615 | 0.622 | 0.622 |
| n_eff SR_full | 45.6 | 45.1 | 45.1 |
| per-series median SR_full | +0.089 (26/29) | +0.105 (27/29) | +0.111 (27/29) |
| sd/SE ratio, SR_full | 0.69 | 0.68 | 0.68 |
| sd/SE ratio, SR_of_SR | 0.73 | 0.75 | 0.75 |
| best cell SR_of_SR | 0.577 | 0.577 | 0.577 |
| best cell | `igrea_level|bdtanh|w252` | same | same |
| five-gate passers | 0 / 522 | 0 / 522 | 0 / 522 |

Both noise-floor ratios stay below 1 on all three, so QNT-78 rule 1 still
forbids quoting a best cell off this grid. QNT-25's reading is untouched.

## What changed on disk

* `signal_zoo/qnt32_verify.py` — GRID 1 reads `qnt19_postfloor/full_sweep.csv`,
  constants re-derived (`_era = "QNT-19"`), QNT-25 originals kept in comments.
* `signal_zoo/macro_windows/qnt82/rederive_refs.py` — same path change.
* This directory — the scratch re-run + its log, kept as the evidence above.
* **Nothing under `macro_sweep/` was modified.**

`python3 signal_zoo/qnt32_verify.py` -> ALL CHECKS PASS, four grids one era.

---

## Found while checking the blast radius: `compare_prefloor.py` had gone era-blind

Its docstring assumed the pre-floor artifacts were "left in place" — but it read
the **live harness output dirs**, and QNT-52 re-ran `window_sweep_full.csv` in
place at 10:06 and QNT-82 re-ran `slow_window_sweep.csv` in place at 10:34. From
10:06 on 2026-09-01 the script silently compared *eras* instead of the floor, and
running it rewrote `qnt19_prefloor_vs_postfloor.csv` with a wrong diff (grid 3
reported 1,491 of 1,512 cells moved, against a footnote that says **0**).

Fixed by pinning the two pre-floor sources to the copies those tickets archived
before re-running:

* grid 2 → `macro_windows/qnt52/prior_run/window_sweep_full.csv`
* grid 3 → `macro_windows/qnt82/pre_rerun/slow_window_sweep.csv`

Both verified against the published footnote before pinning — the grid-2 archive
reproduces pre-floor per-cell dSR **+0.0708, 64% win, corr(day,c2c) 0.415**
exactly. Grid 1 and `macro_combo/combo_scoreboard.csv` were never re-run in
place, so they still read their published paths.

`qnt19_prefloor_vs_postfloor.csv` is restored to **1,530 rows / 169,098 bytes**,
identical to the original, and the script now reproduces every QNT19_FOOTNOTE.md
table: grid 1 144 moved / 15 sign flips; grid 2 c2c +0.022, day −0.020, o2o
−0.027, night `day` 4-gate 37→10, `ongap` 11→6; grid 3 **0 of 1512**.

**Generalisation:** three harnesses in `signal_zoo/` now write into directories
that other tickets treat as frozen evidence. A comparison script must pin its
"before" side to an archived copy, never to a path a harness still writes.
