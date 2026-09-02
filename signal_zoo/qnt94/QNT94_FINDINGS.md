# QNT-94 — `n_eff` ACROSS source series, measured on the per-series SR statistic

**Answer: yes, replace raw `S` with `n_eff`. Measured `n_eff ≈ 0.43·S`, so QNT-78's
`d_min = 2.80·sd/√S` understates the true floor by 1.52×.**

## What was measured

QNT-78 measured `n_eff` of the raw macro INPUTS (11 daily series → 4.29; the full 26-series
daily universe → 6.75) and flagged that the power rule actually needs `n_eff` of the *per-series
SR statistic*, which needs PnLs. That is what this ticket computes.

`signal_zoo/qnt94/series_neff.py` rebuilds the per-cell **net PnL** streams of the two macro
grids already on disk — QNT-12's 29 series × 18 transform-windows at `c2c`, and QNT-14's
11 daily series × 18 × {c2c, day, o2o} — with the sign frozen on the in-sample half exactly as
the original sweeps did and REAL costs (70 TWD/side + 4e-5) priced off each variant's own entry
price. Each source series is collapsed to ONE PnL two ways (equal-weight across its 18 cells,
and its median-SR cell); redundancy is then measured ACROSS series.

Validity check (`verify.py`, `qnt94_verify_grid2.csv`): the rebuilt per-cell SRs reproduce the
on-disk grid at `corr = 0.981`, mean |diff| 0.009 SR. `c2c` and `day` agree to ≤0.010 SR. Only
`o2o` diverges (max 0.56) — the on-disk CSV predates QNT-52's exact `back_open` roll, my rebuild
uses `_base.py`'s current definition. No reported number depends on `o2o`.

## Results

| grid / statistic | S | mean pairwise PnL corr | eigenvalue `n_eff` | design-effect `n_eff` | between-series sd(SR) |
| -- | -- | -- | -- | -- | -- |
| G1 QNT-12, EW-of-18, c2c | 29 | +0.077 (mean abs 0.184) | **11.33** | 9.19 | 0.135 |
| G1 QNT-12, median-cell, c2c | 29 | +0.054 (0.169) | 12.67 | 11.57 | 0.119 |
| G2 QNT-14, EW-of-18, c2c | 11 | +0.133 (0.310) | **4.61** | 4.73 | 0.141 |
| G2 QNT-14, median-cell, c2c | 11 | +0.073 (0.319) | 4.68 | 6.37 | 0.126 |
| G2 QNT-14, paired Δ(day−c2c) | 11 | +0.159 (0.247) | 5.38 | 4.24 | 0.165 |

Common sample 2006→2026 (20.0y); PC1 explains 20.6% of G1's per-series PnL variance and 36.7% of
G2's. The eigenvalue and design-effect forms agree within ~15% everywhere; the design-effect one
is technically what `d_min` wants (it is the SE of a *mean* across series), the eigenvalue one is
what QNT-78 tabulated. They do not change the conclusion.

**The statistic-level `n_eff` matches QNT-78's input-level `n_eff` closely on the same series
set** (11 daily: 4.61 measured on PnL vs 4.29 measured on 20-day input changes). So the cheap
input-level proxy was not misleading. G1's 29 series are more diverse (multiple macro families,
not just US rates + FX), which is why 29 → 11.3 is a better ratio than 11 → 4.6 — but the ratio
is stable: `n_eff/S` = 0.39, 0.44, 0.42, 0.43, 0.49 across the five statistics, **mean 0.43**.

## Restated power table (sd = 0.13, `n_eff ≈ 0.43·S`)

| S | 5 | 11 | 17 | 20 | 26 | 29 | 40 | 60 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| `n_eff` | 2.2 | 4.8 | 7.4 | 8.6 | 11.2 | 12.5 | 17.3 | 25.9 |
| `d_min` as QNT-78 wrote it | 0.163 | 0.110 | 0.088 | 0.081 | 0.071 | 0.068 | 0.058 | 0.047 |
| **`d_min` corrected** | **0.248** | **0.167** | **0.134** | **0.124** | **0.109** | **0.103** | **0.088** | **0.071** |

| target d | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 | 0.20 |
| -- | -- | -- | -- | -- | -- | -- |
| `S_required`, QNT-78 as written | 53 | 24 | 14 | 9 | 6 | 4 |
| **`S_required`, corrected** | **123** | **55** | **31** | **20** | **14** | **8** |

Closed form: `d_min = 4.26·sd/√S` and `S_required = 2.33·(2.80·sd/d)²`.

## Was QNT-14's +0.089 further below its floor than raw-S said?

Yes, roughly twice as far. The statistic is the per-series median `ΔSR(day@1 − c2c@2)`, S=11,
measured `n_eff` 5.38 (eigenvalue) / 4.24 (design-effect), and its own between-series sd is 0.165
(higher than the single-leg 0.13 — the paired-Δ inflation is already inside this measured sd, so
do NOT also apply `√(2(1−ρ))` on top of it).

| floor | raw-S (QNT-78 as written) | corrected, eigenvalue | corrected, design-effect |
| -- | -- | -- | -- |
| at house sd 0.13 | 0.110 | 0.157 | 0.177 |
| at measured sd 0.165 | 0.139 | 0.199 | 0.224 |

QNT-14's claimed +0.089 sat at **0.81×** its raw-S floor. Corrected it sits at **0.57×–0.40×**.
And the number itself did not survive QNT-19: the post-floor value of the same statistic is
**+0.038**, which is 0.17×–0.24× the corrected floor. The grid could never have resolved the
effect it went looking for, by a factor of ~2 rather than the ~1.2 the raw-S arithmetic implied.

## Recommendation — PROPOSED, not enacted

Replace the raw-`S` form of Rule 2 with the `n_eff` form. The rule as ratified is not merely
optimistic, it is optimistic by a *stable, measurable, and large* factor (1.52× on `d`, 2.33× on
`S_required`), which is exactly the case where a correction is worth making rather than a caveat.
Concretely:

> `n_eff ≈ 0.43·S` (measured, QNT-94, on two macro grids, five statistics, range 0.39–0.49).
> `d_min(S) = 2.80·sd/√(0.43·S) = 4.26·sd/√S` · `S_required = 2.33·(2.80·sd/d)²`.
> A ticket that can measure `n_eff` on its own series (one correlation matrix of per-series PnL)
> should report the measured value instead of the 0.43 default.

Enacting this is Reggie's call — QNT-78 was ratified by him and this changes what a sweep is
allowed to claim it can resolve. Nothing in the rules file has been rewritten; the follow-up
note there now carries these numbers and points here.

## Caveats

* `0.43` is measured on macro series only. A grid of, say, options or flow series could be more
  or less redundant; the rule should say "0.43 absent a better number", as sd=0.13 already does.
* `n_eff` of a PnL panel is a proxy for `n_eff` of the SR *estimates*. They coincide to first
  order (the SR estimation error is a function of the same return stream) but not exactly.
* The per-cell sign is fitted in-sample, which induces a little shared structure across series
  and therefore slightly *depresses* `n_eff`. The bias runs toward conservatism.
* Correlations are computed on the 20-year common sample; the pairwise-complete version gives
  `n_eff` 12.67 (G1) and 4.81 (G2), i.e. marginally higher. Conclusion unchanged.

Artefacts: `qnt94_neff.csv`, `qnt94_power_table.csv`, `qnt94_series_neff.png`,
`g1_series_pnl_ew.csv`, `g2_series_pnl_c2c_ew.csv`, `g2_series_dpnl.csv`, `qnt94_verify_grid2.csv`.
