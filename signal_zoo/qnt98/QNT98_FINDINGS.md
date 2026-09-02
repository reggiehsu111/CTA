# QNT-98 — argument-sweep robustness on every "working" macro cell

**Run 2026-09-02.** Reggie's clarification on the ticket redefined the robustness test:
> "sweep the arguments around the working signals … not just along the window dimension, you
> should also sweep swapping operations, like try using InstStdev, InstSkew etc. to replace
> InstMean"

This is that test. Nothing was published, no sign or variant was chosen, no config was written.

## What was run

11 **candidate** cells — every cell that passes all four house gates in the current-era
(post-QNT-19) grids, one per source series, plus the G1 near-miss `igrea`. Each gets a
**144-cell neighbourhood**: 16 operators × 9 windows (20/40/60/90/120/180/252/378/504) on the
same series and the same execution variant, scored under that cell's own published sample and
IS/OOS convention.

8 **control** series that pass no gate anywhere get the identical treatment.

Operator palette, split deliberately:

| family | operators | why |
|---|---|---|
| **location** (6) | `selfz` `robustz` `bdtanh` `rankc` `signth` `dev` | the published palette. All monotone in `x`, so they agree *by construction* — coherence here proves nothing |
| **swap** (10) | `instStdev` `instSkew` `instKurt` `minmax` `chg` `slope` `dStdev` `dSkew` `ac1` `pctpos` | Reggie's ask. The window statistic itself is replaced, so these are genuinely different signals |

Discipline: PIT inputs via `ctx.macro` (`load_macro_tw`), roll-adjusted legs re-derived from
`_base.py`, realistic costs (70 TWD + 4e-5/side), **sign frozen on the IS half**, and shift(2)
held on every variant for non-daily inputs (only the daily US-close series earn shift(1) on
`day`/`ongap`) — the `slow_window_sweep.py` convention.

**Reproduction check:** the 11 published cells were re-scored capped at the grids' own last bar.
9 of 11 are bit-identical, `us_dxy_broad` moves −0.0053 (QNT-52 o2o), `igrea` +0.0003.
Max |diff| 0.0053. → `claim_reproduction.csv`.

## Results

### 1. The window axis is coherent, but the published window is never the peak
SR rises monotonically with `w` for 9 of 11 names and peaks at 378–504, not the 252 the
published 3-window grid could reach. `twd_usd` and `kr_kospi` **flip sign** between w=120 and
w=180 — they agree with their own published sign on only 40% / 49% of their neighbourhood.

### 2. The operator swap kills it — and that is the finding
Median SR across the 11 candidates at w=252:

| location ops | | genuine swaps | |
|---|---|---|---|
| `dev` | +0.66 | `chg` | +0.62 |
| `signth` | +0.59 | `minmax` | +0.61 |
| `bdtanh` | +0.57 | `pctpos` | +0.54 |
| `selfz` | +0.52 | `slope` | +0.48 |
| `rankc` | +0.52 | `dSkew` | +0.43 |
| `robustz` | +0.46 | `ac1` | +0.20 |
| | | `instKurt` | +0.19 |
| | | `instSkew` | +0.17 |
| | | `dStdev` | +0.08 |
| | | **`instStdev`** | **−0.03** |

Everything that is a monotone read of the **level** of the macro series scores +0.45…+0.66.
Every **dispersion / shape** statistic — `instStdev`, `instSkew`, `instKurt`, `dStdev` — lands at
zero. `chg`/`slope`/`minmax`/`pctpos` are trend reads of the level, so they belong with the first
group. There is one idea here (the level of a slow macro series vs MTX), replicated ten ways;
there is no volatility, skew or kurtosis edge at all. Measured directly: the 144 positions in a
neighbourhood have `n_eff` **4.9–6.9**, not 144.

### 3. The published number shrinks by 0.31 SR the moment the arguments move
Median across the 11: published cell 0.63 → neighbourhood median 0.32. The published cell ranks
**5th to 47th** of its own 144 — for `us_empire_state` 46 of its own neighbours beat it.

### 4. The null: sweeping 144 arguments buys ~0.70 SR for free
Same 144 neighbourhoods, scored against **circularly shifted** returns (4 shifts, 400–1800 bars;
positions, autocorrelation, turnover, cost model and return distribution all preserved, only the
alignment destroyed):

| | best-of-144 SR | 4-gate pass rate |
|---|---|---|
| circular-shift **null** | median **0.699**, p90 0.858, max **1.362** | **4.0%** |
| candidates | median 0.848 | 14.2% |
| controls | median 0.772 | 1.8% |

Only **4 of 11** candidate series clear the null p90 — as do **2 of 8** controls
(`wti` 0.998, `us_real_10y` 0.900). Mann-Whitney on best-of-144, candidates vs controls,
p = 0.119. The headline `cny_usd/ongap/dev/w252 = 1.04` sits inside the null's own range.
The aggregate 4-gate rate (14.2% vs null 4.0%) is the one statistic that does exceed the null,
but the candidate *set* was chosen on full-sample gates, so it is not an independent test.

### 5. Where the PnL comes from
* The 10 night-era cells are not 10 bets: mean pairwise net-PnL correlation +0.26,
  `n_eff` = **4.6**. `us_semi_ip`↔`us_semi_ip_nsa` 0.94, `kr_kospi`↔`epu_global` 0.91.
* **Median 55% of each cell's whole 10-year net PnL was earned in 2025–26** — 20 months out of
  120. Median SR across the 10 by era: 2017-21 **+0.40**, 2022-24 **+0.49**, 2025-26 **+1.56**.
  A common, recent regime, not eleven findings.
* Every gate-passing cell in the program lives in the 10-year night era. The 25-year `c2c` grid
  (G1, 522 cells) still passes **zero**.

### 6. Transaction costs are not the constraint
Median turnover 8.6×/yr, held 99.6% of days. Ladder on the published cells: gross → stub
(20+2e-5) → real (70+4e-5) → 3× real costs the neighbourhood median 0.326 → 0.317 → 0.301 →
0.247, i.e. **0.025 SR** from gross to realistic. Stability is the binding constraint, as
QNT-12/16/25 already found.

## QNT-78 Rule 1 reporting line

```
QNT-98 candidate neighbourhoods (11 series x 144 arguments): 1584 cells = 11 source series x 144
transform-windows, n_eff ~ 229 (ICC(series) = 0.04). Per-series median SR_net +0.292, 11/11
positive, Wilcoxon p = 0.001. sd(SR_net) across cells = 0.343 vs SE(SR | 10y) = 0.319
(ratio 1.08) -> dispersion ~ the noise floor - ranking is mostly noise.
  best cell SR_net = +1.158; expected max of 229 independent draws ~ +1.162
    -> the best cell is what a null grid this size produces.
  rule 4: sign fitted IS - per-series median SR_IS +0.453 vs SR_OOS +0.238.
  11/11 series disagree with themselves on sign across their own cells.
```
Full text in `qnt98_headline.txt`. Rule 2 (pre-run power target) does not bind: this is a
neighbourhood robustness sweep around cells that already exist, not a discovery sweep, and it
adds **zero** source series. Per QNT-94 the argument axis asymptotes at `1/ICC`, which is exactly
what the measured `n_eff` 4.9–6.9 per 144 cells confirms.

## Verdict

**No new tradable signal.** The macro→MTX cells survive the window axis but not the operator
axis, their published values are inside a circular-shift null's best-of-144 range, they are 4.6
effective bets rather than 10, and over half their lifetime PnL is 20 months old.

## Artefacts

| file | contents |
|---|---|
| `neighbourhood_sweep.py` | the sweep (candidates + controls, 2,736 cells) |
| `neighbourhood_sweep.csv` | full grid, one row per cell, 4 cost ladders each |
| `claim_reproduction.csv` | 11 published cells re-scored against the published grids |
| `neighbourhood_redundancy.csv` | `n_eff` of the 144 positions per series |
| `null_and_report.py` / `null_circular_shift.csv` / `null_summary.csv` | the circular-shift null |
| `claim_cell_pnl.csv` | daily net PnL of the 11 published cells |
| `figures.py` / `qnt98_argument_robustness.png` / `qnt98_pnl_costs.png` | the two figures |
| `qnt98_headline.txt` | QNT-78 Rule 1 line |
| `breadth_ledger.py` / `qnt98_breadth_ledger.png` | the earlier breadth+power ledger (unchanged) |
