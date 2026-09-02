# QNT-78 — house rules for MTX factor sweeps

**Status: IN FORCE from 2026-09-01.** Reggie answered "Both" on QNT-78: Rule 1 (reporting gate
on positive claims) and Rule 2 (pre-run power target) are both adopted. The condensed text is in
the 台指期 standing brief; this file is the long form — full rule text, the power tables and the
evidence each way. QNT-25 measured the problem, QNT-32 shipped the helper
(`cta.sweep_headline` / `cta.paired_headline`) and wired the four harnesses to print it.
Ratification changed no code and re-ran no sweep. (Was `QNT78_PROPOSED_RULES.md` until adoption.)

---

## Rule 1 — the reporting line is required for any POSITIVE sweep claim

> **Sweep reporting gate (in force).** A write-up that quotes a positive number off a factor grid must
> carry the `cta.sweep_headline` / `cta.paired_headline` output: the **per-series `n`** (never
> the cell count) beside any p-value, `ICC(series)` and `n_eff`, and `sd(SR) across cells`
> beside `SE(SR | n_years)`. **If that ratio is below 1, no "best cell" may be quoted as a
> result at all** — the ranking is a ranking of estimation error. A paired ΔSR must use its own
> larger `SE(Δ) = SE(SR)·sqrt(2(1−ρ))`. Never quote a full-sample or IS SR from an IS-fitted
> sign as evidence of edge.
>
> **Negative conclusions are exempt.** "0 of N pass the gates" only gets safer with fewer
> effective tests, so a no-result write-up needs no headline line.

**For.** The harnesses print the line last, but nothing stops an agent quoting the top-25 table
above it, and that is exactly what QNT-12/14/16 did before QNT-25. The failure mode is
one-directional: every collapse so far (igrea, EW3, the day window) was a *positive* claim that
died at per-series `n`. The gate costs nothing to satisfy — the number is already on stdout.

**Against.** It is a rule about prose, not computation, so it cannot be enforced mechanically;
it works only as something a session is told to check. `qnt32_verify.py` proves the *helper* is
right, not that a write-up used it.

**Scope choice.** The exemption for negative results is deliberate and matches QNT-25's own
finding. The stricter alternative — require the line on every sweep write-up regardless of sign
— is simpler to remember but taxes the common case (most sweeps here return nothing).

**Addendum, 2026-09-02 (QNT-100).** The reporting line now also carries an exposure clause
whenever the grid records `mean|exec_w|`: median exposure, median `|beta per unit exposure|`,
and a count of cells below 0.31 exposure. This is not a new rule — it is the same rule applied
to a second way a grid produces fake positives. The `|beta| < 0.15` house gate is measured on
realised PnL, so it shrinks with exposure; a zero-information long-night mask at 5-10% exposure
passes all four gates 10-15% of the time (QNT-99 Part A3, QNT-100). Replacing the gate was
**rejected** (2026-09-02) — the house keeps sparse books — so a sparse survivor has not been
shown market-neutral by the gate, and quoting one means quoting `beta_per_w` and `mean_abs_w`
with it. On QNT-99's 800-cell calendar grid the clause reads: median exposure 0.05, median
`|beta/w|` 0.69, 790/800 cells sparse.

---

## Rule 2 — a new sweep states its target effect size and required S before it runs

> **Pre-run power target (in force).** A ticket that commissions a new sweep must state, *before the run*:
> the target effect size `d` in SR units, the assumed between-series sd, and the number of
> source series it implies —
>
> ```
> S_required  =  (2.80 · sd / d)²          d_min(S) = 2.80 · sd / sqrt(S)
> ```
>
> (two-sided 5%, 80% power; 2.80 = z₀.₉₇₅ + z₀.₈₀. Between-series sd measured at **0.119** on
> single-leg SR grids and **0.132** on paired ΔSR grids — use 0.13 absent a better number.)
> If the available `S` is below `S_required`, say so in the ticket and either add source series
> or declare the run descriptive-only. **Do not widen the transform palette** — `n_eff` per
> series asymptotes at `1/ICC ≈ 1.61` and 18 transform-windows already buy 1.56, so tripling
> the palette adds 0.03 of a test while one more source series adds a whole one.

Reference table at sd = 0.13:

| target d | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 | 0.20 |
| -- | -- | -- | -- | -- | -- | -- |
| S required | 53 | 24 | 13 | 9 | 6 | 3 |

| S | 5 | 11 | 17 | 20 | 29 | 40 |
| -- | -- | -- | -- | -- | -- | -- |
| smallest resolvable d | 0.163 | 0.110 | 0.088 | 0.081 | 0.068 | 0.058 |

**For.** It is answerable in advance and it would have changed a real decision. QNT-14 ran 11
daily series against a +0.089 target whose floor at S = 11 is 0.112 — the effect it was looking
for was below its own resolution before a single cell was computed. That grid then cost QNT-18
and QNT-19 to unwind. The check is one line of arithmetic on numbers already published.

**And the series were available.** `macro_series_meta` has only 12 daily series (11 used +
`us_iorb`), but the wide tables carry 14 more daily columns — `us_rates_daily`
(fed_funds_eff, sofr, dgs10, dgs2, t10y2y, dgs3mo), `us_risk_daily` (vix, vix_3m, hy_oas,
ig_oas, baa_10y_spread, aaa_10y_spread) and `jp_markets_daily` (nikkei225, usdjpy) — for a raw
daily universe of **25–26**. QNT-14 was underpowered by scope, not by data.

**Against — and this is the honest caveat.** `S` in the formula is a count of *independent*
draws, and the source series are not independent. Measured here (20-day changes, 2005→,
correlation of the raw inputs):

| set | S | eigenvalue `n_eff` | mean abs pairwise corr |
| -- | -- | -- | -- |
| QNT-14's 11 daily | 11 | **4.29** | 0.318 |
| full daily universe | 26 | **6.75** | 0.265 |

So going 11 → 25 series buys about **+2.2 effective series**, not +14. The same ICC collapse
that killed the cell axis applies to the series axis too, just far more weakly (0.27 vs 0.62).
**A power target stated in raw `S` is therefore an upper bound on power, not the power.** The
number the rule actually wants is `n_eff` of the *per-series SR statistic*, which needs the
per-series PnLs and has not been measured — filed as a follow-up. Until then the rule should be
read as "state `S` and acknowledge it is optimistic", not as a certificate.

---

## What adoption did NOT change

No code change (QNT-32 already shipped the helper; there is no pre-run power calculator and these
rules do not add one — the formula above is two multiplications). No sign selection, no
`recommended_variants`, no write to `mtx_signal_config`, no sweep re-run, and no retro-editing of
the posted QNT-10/12/14/16 comments. Both rules bind write-ups and tickets, not the interpreter.

**QNT-94 — MEASURED 2026-09-01. Rule 2's `S` is optimistic by 1.52x, and the fix is proposed
below but NOT enacted (ratifying it is Reggie's call).** QNT-94 rebuilt the per-cell net PnLs of
both macro grids, collapsed each source series to one PnL, and measured `n_eff` of the
*per-series SR statistic* across series:

| grid / statistic | S | mean pairwise PnL corr | eigenvalue `n_eff` | design-effect `n_eff` |
| -- | -- | -- | -- | -- |
| QNT-12, 29 series, c2c | 29 | +0.077 | **11.33** | 9.19 |
| QNT-14, 11 daily series, c2c | 11 | +0.133 | **4.61** | 4.73 |
| QNT-14, paired dSR(day-c2c) | 11 | +0.159 | 5.38 | 4.24 |

`n_eff/S` = 0.39-0.49 across five statistics, **mean 0.43**. The statistic-level number tracks
QNT-78's cheap input-level proxy closely on the same set (4.61 vs 4.29), so the proxy was sound.

*Proposed amendment (pending Reggie):* `n_eff ~ 0.43*S`, hence
`d_min(S) = 2.80*sd/sqrt(0.43*S) = 4.26*sd/sqrt(S)` and `S_required = 2.33*(2.80*sd/d)^2`.
Restated at sd = 0.13 — S = 5/11/17/20/29/40 resolves d = 0.248/0.167/0.134/0.124/0.103/0.088,
and a target d = 0.10 needs **S = 31**, not 14. A ticket that can measure `n_eff` on its own
series should quote the measured value rather than the 0.43 default.

QNT-14 against the corrected floor: its claimed +0.089 sat at 0.81x the raw-S floor, but at
0.57x-0.40x the corrected one; QNT-19's post-floor value of the same statistic (+0.038) sits at
0.17x-0.24x. Until the amendment is ratified, Rule 2 still reads as written and a ticket
satisfying it must say its `S` is optimistic — now quantifiably, by ~1.5x on `d`.
Full write-up and artefacts: `signal_zoo/qnt94/QNT94_FINDINGS.md`.
