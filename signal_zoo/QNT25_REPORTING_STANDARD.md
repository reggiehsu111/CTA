# Reporting line for MTX factor sweeps

**Status: ADOPTED and implemented (QNT-32, 2026-09-01).** QNT-25 measured it and proposed it;
QNT-32 shipped it as `cta.sweep_headline` / `cta.paired_headline` and wired the four sweep
harnesses to print it. Reggie has not been asked to ratify it as a *house* convention — see
"Open decision" at the bottom.

Every sweep here is `n_series x 6 transforms x 3 windows`. Measured on four grids, the
18 transform-window cells of one source series are worth **~1.5 independent tests**
(eigenvalue n_eff 1.42–1.64; mean pairwise PnL correlation 0.61–0.82; PC1 77–83%).
A "522-cell sweep" is **~45 tests**, and a "198-cell paired median" is **~21**.

## The line every sweep write-up should carry

> `<grid>`: **N cells = S source series x 18 transform-windows**, n_eff ≈ *E* independent tests
> (ICC(series) = *r*). Headline collapsed to one number per series: **median X, k/S positive,
> Wilcoxon p = *p***. Cross-cell sd(SR) = *a* vs SE(SR | *Y* years) = *b* (ratio *a/b*).

Rules that go with it:

1. **Aggregate to one number per source series before any test.** Median over the family, not
   the best cell. Report the per-series `n`, never the cell count, next to any p-value.
2. **Print the noise floor beside the dispersion.** `SE(SR) ≈ sqrt((1 + SR²/2) / n_years)`
   — about **0.20 at 25 years**. If `sd(SR)` across cells is below that, the ranking is a
   ranking of estimation error and no "best cell" may be quoted as a result.
3. **A paired ΔSR needs its own SE**: `SE(Δ) ≈ SE(SR)·sqrt(2(1−ρ))`, ρ = corr of the two
   legs across cells. On the day-vs-c2c grids ρ ≈ 0.42, so SE(Δ) ≈ **0.22** — three times the
   +0.071 that was reported as an effect.
4. **Never report a full-sample or IS SR from an IS-fitted sign as evidence of edge.** On the
   QNT-12 grid this alone turns 11/29 into 28/29.
5. **Correct the best cell for search.** Report the Šidák-corrected p across the number of
   *series* searched, and the expected best-of-S under the observed cross-series sd.
6. **`SR_of_SR` is itself an estimate.** It is a Sharpe of the yearly-SR sample, so
   `SE ≈ sqrt((1 + SR_of_SR²/2) / n_years) ≈ 0.21` at 25 years. Quote it as `0.54 ± 0.21`,
   never as a bare number against the 0.60 gate.

## The helper (implemented — `cta/sweep_report.py`)

```python
import cta

# single-leg grid (one SR per cell)
cta.sweep_headline(df, "SR_net", series_col="series", n_years=None, label="my sweep").print()

# paired grid (variant A vs variant B on the same cells) — rule 3, its own larger SE
cta.paired_headline(piv["day"], piv["c2c"], series=piv["series"],
                    n_years=25, a_name="day", b_name="c2c").print()
```

Both return a `Headline` whose `str()` is the one line and whose `.detail()` adds the
rule-2/3/4 warnings; `.stats` is the dict of every number for programmatic use.
`cta.se_sr(sr, n_years)` and `cta.icc_neff(df, value, group)` are exported too.

Reproduces QNT-25's published numbers exactly on all four grids — `qnt32_verify.py` is the
regression check (ICC 0.615/0.489, n_eff 45.6/56.0, per-series median +0.089 at 26/29,
sd 0.139/0.147 vs SE 0.200).

Wired into: `macro_sweep/macro_sweep.py`, `macro_windows/macro_window_sweep.py`,
`macro_windows/slow_window_sweep.py`, `macro_windows/registered_window_sweep.py` — each
prints the line **last**, so it is what lands in a write-up. Per QNT-25's own instruction the
posted QNT-10/12/14/16 comments were **not** retro-edited.

Edge cases the helper handles rather than silently mis-stating: a missing `series_col`, a grid
where a cell *is* a series (it says "n was already honest" instead of claiming ICC = 1.00), a
single-series grid, all-NaN values, and unbalanced groups (Sokal–Rohlf `k0`, which equals
`mean(k)` when balanced so the QNT-25 numbers are unchanged).

## Power: the thing to add is SERIES, not transforms (QNT-32)

QNT-32 measured what each grid could have resolved (two-sided 5%, 80% power, on the
**between-series** sd):

| grid | S | between-series sd | smallest resolvable | observed |
| -- | -- | -- | -- | -- |
| QNT-12 single-leg SR | 29 | 0.119 | **0.062** | +0.089 — resolvable |
| QNT-14 paired dSR(day−c2c) | 11 | 0.132 | **0.112** | +0.089 — **not** resolvable; needed S ≈ 17 |

So the blanket claim "the marginal macro sweep has near-zero information value" is too strong,
and in a specific way: the **cell-level ranking** is worthless, but the **per-series aggregate
test** was adequately powered at S = 29 and underpowered at S = 11. The QNT-14 day-window
positive was simply below its own resolution floor.

And the cell axis is exhausted. `n_eff` per series asymptotes at `1/ICC`:

| cells/series | 6 | 18 | 54 | 180 |
| -- | -- | -- | -- | -- |
| n_eff per series (ICC 0.62) | 1.46 | 1.56 | 1.59 | 1.61 |

18 transform-windows already buy 1.56 of a hard ceiling of 1.61. **Tripling the transform
palette adds 0.03 of a test.** Adding one more source series adds a whole one.

## Status — both rules adopted (QNT-78, 2026-09-01)

The helper is in and the harnesses print it. Reggie ratified both house rules on 2026-09-01:

1. **the reporting line is required for any positive sweep claim** — a write-up quoting a best
   cell without per-series `n`, `ICC`/`n_eff` and `sd(SR)` vs `SE(SR)` is rejected, and if that
   ratio is below 1 no best cell may be quoted at all. Negative conclusions are exempt.
2. **a new sweep states its target effect `d` and required `S` before it runs** —
   `S_required = (2.80·sd/d)²`, sd 0.13; short on `S` means add source series or declare the run
   descriptive-only.

Both are in the 台指期 standing brief. Full text, power tables and the evidence each way:
**`QNT78_SWEEP_RULES.md`** (was `QNT78_PROPOSED_RULES.md`). Caveat carried by Rule 2: raw `S`
overstates power because source series correlate — QNT-94 measures the honest `n_eff`.
