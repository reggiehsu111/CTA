# QNT-100 — the `|beta| < 0.15` gate is scale-evadable; what replacing it actually buys

> **DECISION (Reggie, 2026-09-02): REJECTED — the house beta gate stays `|beta| < 0.15`.**
> "We don't need to completely remove exposures." The proposal bundled the exposure floor
> `mean|exec_w| >= 0.30`, and the noise analysis below is exactly why it could not be unbundled:
> under ~0.31 exposure the ratio is inside its own standard error, so at sparse exposure the rule
> closes the leak mainly by dropping sparse books wholesale — which is not a trade the house
> wants. Everything measured below still stands as a fact about *results*; it is no longer a
> proposal about *rules*. What changed on the box after the decision is in **What shipped**.

Ticket: QNT-100 (from QNT-99 Part A3). Code: `cta/gates.py`, `cta/signal_stats.py`, `cta/sweep_report.py`.
Evidence: `signal_zoo/qnt100/` — `gate_fix.py` (A–C), `gate_fix_d.py` (D), `power_test.py` (E),
`noise_floor.py`, `figures.py` → `qnt100_gate_fix.png`. Nothing here re-runs a sweep, picks a
sign, or writes to a config table.

---

## The claim, confirmed

`beta` is measured on realised PnL, so it scales with exposure while `SR_of_SR` and
`positive_years` do not. The MTX night session's unconditional +1.146 SR drift can therefore be
kept and hidden by shrinking size.

**A — purely random, zero-information long-night masks** (100 draws per exposure, 2010–, real
costs; QNT-99 used 20):

| mean\|exec_w\| | SR_net | SR_of_SR | beta | beta/\|w\| | pass OLD | pass NEW |
|---|---|---|---|---|---|---|
| 0.02 | 0.110 | 0.231 | 0.019 | 0.956 | **14%** | 0% |
| 0.05 | 0.189 | 0.242 | 0.049 | 0.983 | **14%** | 0% |
| 0.075 | 0.210 | 0.277 | 0.073 | 0.980 | **9%** | 0% |
| 0.10 | 0.248 | 0.330 | 0.100 | 1.001 | **16%** | 0% |
| 0.15 | 0.290 | 0.326 | 0.149 | 0.994 | 4% | 0% |
| 0.20 | 0.316 | 0.364 | 0.198 | 0.993 | 1% | 0% |
| 0.40 | 0.564 | 0.704 | 0.394 | 0.985 | 0% | 0% |
| 1.00 | 1.146 | 1.764 | 1.000 | 1.000 | 0% | 0% |

Overall **7.2% → 0.0%**. `beta/mean|exec_w|` reads 0.96–1.00 at *every* exposure — that is the
whole design claim, and it holds.

**C — re-gating seven published grids** (8,930 cells, no re-simulation; every MTX sweep that
recorded `mean|exec_w|`):

| grid | cells | pass OLD | pass NEW |
|---|---|---|---|
| QNT-99 event calendar | 800 | 70 (8.8%) | **0** |
| QNT-99 event surprise | 813 | 19 | 10 |
| QNT-99 put OI | 1,584 | 0 | 0 |
| QNT-98 neighbourhood | 2,736 | 246 | 183 |
| QNT-94 macro windows | 1,386 | 27 | 18 |
| QNT-94 slow windows | 1,512 | 15 | 9 |
| QNT-94 registered | 99 | 26 | 22 |
| **total** | **8,930** | **403** | **242** |

**All 70 of QNT-99's calendar passers die.** They were the leak, exactly as that ticket said.

The 403→242 split by exposure is the argument in one table:

| mean\|exec_w\| | old passers | kept | median \|beta\| | median \|beta/w\| |
|---|---|---|---|---|
| <0.2 (sparse) | 92 | **10 (11%)** | 0.037 | **0.602** |
| 0.2–0.4 | 6 | 4 (67%) | 0.044 | 0.112 |
| 0.4–0.6 | 35 | 24 (69%) | 0.056 | 0.107 |
| 0.6–0.8 | 259 | 193 (75%) | 0.065 | 0.090 |
| >0.8 | 11 | 11 (100%) | 0.052 | 0.065 |

Sparse survivors read `|beta|` 0.037 — apparently the most market-neutral book on the board —
and 0.602 per unit of exposure. Fully-invested survivors read 0.09 either way and mostly keep
their status.

**D — the live book pays nothing.** All 11 registered signals, built as the runner builds them
(declared sign, tanh-252 unless `pre_normalized`), on their declared variants: **31
signal-variants, 3 pass OLD → 3 pass NEW, zero demotions.** Same result through
`cta.batch_signal_stats` on c2c. Minimum `mean|exec_w|` across the registered book is 0.191, so
the denominator is never near zero there.

Two incidental readings, both `enabled=True`:

* `nfci_loose_drift_d3_12` — `mean|exec_w|` **0.191**, held 19% of bars, `beta` 0.153,
  **`beta/w` 0.805**. It misses the old gate by 0.003 and looks like a rounding argument; under
  the new metric it is 80% a long index bet whenever it is on. (0.805 is >2σ even at that
  exposure — see below — so this reading is real, not noise.)
* `month_turn_drift_d5_d10` — `beta` 0.729, `beta/w` **1.004**. A pure index bet at full size;
  both rules reject it.

---

## The part the ticket did not anticipate: the ratio has its own noise

Dividing beta by `mean|exec_w|` divides its standard error by it too. Fitted on books whose true
beta is **zero by construction** — informed two-sided masks (part E) and random two-sided masks
(part B), exposures 0.02–1.00, both families on one line:

```
sd(beta_per_w) ≈ 0.084 / sqrt(mean|exec_w|)
```

| exposure | sd(beta/w) | P(a genuinely NEUTRAL book fails \|beta/w\|<0.15 by noise alone) |
|---|---|---|
| 1.00 | 0.084 | 7% |
| 0.50 | 0.119 | 21% |
| **0.31** | **0.151** | 32% |
| 0.20 | 0.188 | 43% |
| 0.10 | 0.266 | **57%** |
| 0.05 | 0.376 | **69%** |

At full exposure `sd = 0.084 = SE(beta)` — the new rule is exactly as noisy as the old one
wherever the old one was actually doing its job. But **below `mean|exec_w| ≈ 0.31` the threshold
is inside its own noise.** A directional sparse book and a neutral sparse book are not
distinguishable there, and the reason `pass NEW` is 0.0% in part A is only partly that the metric
is sharp — it is also that at 5% exposure the gate rejects sparse books nearly wholesale.

**E — power, measured.** Sparse masks with a *genuine* two-sided edge (on a random fraction of
nights, take that night's realised direction with probability p; `beta ≈ 0` by construction,
mean `beta/w` ±0.02 at every exposure). Of the masks the old gate admitted, the new gate keeps
**66%**; the ones lost have median `|beta/w|` 0.204 against 0.072 for the kept — borderline
readings pushed over by estimation error, not directional books.

A noise-scaled threshold (`0.15 + 2·sd(w)`) restores power to the old level exactly — and
reopens the leak to 3.4%. You cannot have both from beta alone: with ~200 held nights the
directionality of a 5%-exposure book is not estimable to ±0.15 by any rule.

**So the two halves do different jobs, and both are needed:**

* `|beta| / mean|exec_w| < 0.15` — scale-invariant, does the real work from ~0.3 exposure
  upward. It is what caught `month_turn_drift` (1.004) and `nfci_loose_drift` (0.805).
* an **exposure floor** `mean|exec_w| ≥ 0.30` — because below it *no* beta gate is informative.
  This is the ticket's "optional minimum `held_pct`", but on exposure rather than held_pct: a
  book can be held every single bar at 5% size and `held_pct` would read 100%.

Threshold sensitivity, all 8,930 cells (survivors / of which sparse `w<0.2`):

| rule | survivors | sparse |
|---|---|---|
| `|beta/w| < 0.15` | 243 | 10 |
| `|beta/w| < 0.20` | 307 | 13 |
| `|beta/w| < 0.225` | 330 | 15 |
| `|beta/w| < 0.25` | 356 | 15 |
| `|beta/w| < 0.30` | 403 | 16 |
| `|beta| < 0.15` (old) | 403 | **92** |

0.15 tightens the board by 40%; 0.30 keeps the same headline count while cutting sparse
survivors 92 → 16. A typical tanh-normalised MTX book runs `mean|exec_w|` ≈ 0.65, so 0.15 on the
ratio is ~1.5× stricter than 0.15 on raw beta for the signals we actually build. That is a
separate decision from closing the leak.

---

## One more hole, guarded

The ratio is evadable from the other end too: lever *up* and divide the beta away. The QNT-99
put-OI grid has 135 of 1,584 cells with `mean|exec_w|` up to 1.0e5 (unnormalised signals). MTX
positions are bounded in [−1,+1] by house convention, so the denominator is clipped at 1.0 —
such a book falls back to plain `|beta|`, the conservative answer for a book whose exposure unit
is undefined. The *reported* `mean_abs_w` is unclipped so the anomaly stays visible.

## What shipped

The gate is unchanged. The evidence is kept, and moved into the reporting layer, which is where
a rejected rule still has to do its work.

* `cta/gates.py` (`cta.house_gates`) — the four gates in **one place**, because the reason this
  defect survived is that `df.beta.abs() < 0.15` was copy-pasted into eight sweep scripts.
  `beta_mode` defaults to **`"raw"` = the house rule**; re-gating all 8,930 published cells with
  the default reproduces the pre-QNT-100 board exactly (403 survivors), and the registered book
  (31 signal-variants) passes 3/3 as before. `beta_mode="per_w"` / `"both"` and the `min_abs_w` /
  `min_held_pct` floors remain as **opt-in diagnostics** (`ABS_W_MIN = HELD_PCT_MIN = None`) —
  that is how the 70 QNT-99 calendar "passers" were shown to be the leak.
* `cta/signal_stats.py` — every stats dict carries `mean_abs_w` and `beta_per_w`. Additive; no
  existing column or default changed. These are **reported diagnostics, not gates**.
* `cta/sweep_report.py` — `cta.sweep_headline` now appends an exposure line to its detail block
  whenever the grid carries `mean|exec_w|`:

  > `exposure: median mean|exec_w| = 0.05, median |beta per unit exposure| = 0.69; 790/800 cells
  > sit below 0.31 exposure, where |beta| < 0.15 cannot tell a directional book from a neutral
  > one (QNT-100) — quote beta_per_w and mean_abs_w for any of those you headline.`

  That is the QNT-99 calendar grid, and QNT-78 rule 1 already requires this line on any positive
  sweep claim. So the leak is now visible at the point where a number gets quoted, without the
  board losing a single sparse signal.
* **No sweep script's threshold was changed and no config table was touched.**

## What is still true, and open

The leak is real and is not closed: a zero-information long-night mask at 5-10% exposure passes
all four house gates 10-15% of the time, and that is an accepted risk, not a fixed defect. The
consequence for how results are read:

* A sparse survivor (`mean|exec_w| < 0.31`) has **not** been shown to be market-neutral by the
  beta gate — the gate cannot resolve it there. Quote `beta_per_w` and `mean_abs_w` beside it.
* `nfci_loose_drift_d3_12` (enabled=True) reads `beta` 0.153, `mean|exec_w|` 0.191,
  `beta_per_w` 0.805 — it misses the current gate by 0.003, and it is ~80% a long index bet
  whenever it is on. Unchanged by this ticket; recorded so it is not rediscovered.
* `qnt32_verify.py` passes unchanged after the reporting-line addition.
