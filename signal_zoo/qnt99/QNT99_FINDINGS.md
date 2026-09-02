# QNT-99 — expanding the sweep onto MACRO EVENTS, and put-OI as a standalone signal

**Run 2026-09-02, lane `taiex-futures`, `/home/ubuntu/mtx`.** Nothing was published, no sign or
variant was chosen, no `mtx_signal_config` row was touched.

---

## Part 0 — what the MTX macro sweep has covered so far

| ticket | what was swept | outcome |
|---|---|---|
| QNT-10 | built the tidy macro layer (`macro_series`, `macro_series_meta`) | data, not a result |
| QNT-12 | **Grid 1** — 29 source series x 6 transforms x 3 windows = 522 cells, `c2c`, sign frozen IS | **0/522 pass all four gates**; `igrea` the near-miss |
| QNT-14 | **Grid 2** — the 11 daily series at `day`/`ongap`/`o2o`/`c2c` (1,386 cells) | claimed a +0.089 day-window uplift |
| QNT-16/18 | day-window generalisation | claim **retracted** on effective-n |
| QNT-19 | PIT bug: `load_macro_tw` did not floor the US observation at +1 | fixed; post-floor the day-window ΔSR is **+0.013, p=0.52** |
| QNT-13/combo | EW3 / reweighted macro combinations | best `SR_of_SR` **0.542**, under the 0.6 gate |
| — | macro regime overlay | regimes do not condition the MTX book; episode count binds |
| QNT-25 | effective-n of the transform axis | 18 transform cells = **~1.5 tests** (ICC 0.62) |
| QNT-32 | `cta.sweep_headline` / `paired_headline` + `qnt32_verify.py` | the reporting helper |
| QNT-52 | retired the o2o roll approximation | sweeps now share `_base.py`'s o2o |
| QNT-78 | **house sweep rules ratified** | reporting line on positive claims; pre-run power target |
| QNT-92 | grid provenance frozen | 4 grids, one era |
| QNT-94 | n_eff **across source series** | `n_eff ~ 0.43*S` — raw S overstates power by 1.52x on `d` |
| QNT-98 | 144-argument neighbourhood + circular-shift null on every "working" cell | best-of-144 buys **SR 0.70 for free**; every dispersion/shape operator swap is flat |

**Union of macro SOURCE series ever swept: 32.** The transform axis is exhausted (QNT-25/98);
QNT-78 Rule 2 says the only lever left is **more source series**. That is what this ticket adds.

---

## Part A — the expansion: macro EVENTS

Two axes that no previous grid used. Both come out of `us_macro_releases`, the exact-ALFRED
release-timestamp table that until now only served as a PIT *filter*.

**PIT.** Each `reference_date` is mapped through `cta.us_macro._available_from_tw` to the first
TAIFEX 08:45 open **strictly after** the exact `release_ts`, so an input labelled `t` is public
before `t`'s open. The production pipeline only computes at 15:31 TPE
(`project_mtx_pit_compute_schedule`), so the harness's `_SHIFT = {c2c:2, o2o:2, day:1, ongap:1}`
is the correct execution lag on top of that and is used unchanged. Real costs
(70 TWD + 4e-5/side), roll-adjusted legs from `_base.py`, sign **frozen on the IS half**.

**Stated contamination:** the DB stores FINAL revised values, not first prints. A surprise built
from revised data knows a little of the future — which biases the surprise *toward* working, so
the null below is safe and a positive would have needed vintage data to confirm.

### A1 — release SURPRISE (the value in the event)

Pre-run power target (Rule 2): target `d = 0.15` SR, sd 0.13 → `S_required` = 6 raw / **14
QNT-94-corrected**; `S` available = **15 events** → adequate for d = 0.15, `d_min = 0.143`, and
adequate for nothing smaller. Raw `S` is optimistic (CPI/PPI/PCE and NFP/claims/JOLTS correlate).

813 cells = 15 events x 17 fields x {k=12,36} x {hold 1,3,10 d} x {z, sign} x 4 variants.

```
QNT-99 event surprise: 813 cells = 15 source series x 54 transform-windows, n_eff ~ 233
(ICC(event) = 0.05). Per-series median SR_net +0.044, 11/15 positive, Wilcoxon p = 0.064.
sd(SR_net) across cells = 0.221 vs SE(SR | 15y) = 0.279 (ratio 0.79)
  -> dispersion BELOW the noise floor - ranking cells ranks noise.
  best cell SR_net = +0.565; expected max of 233 independent draws ~ +0.604
  -> the best cell is what a null grid this size produces.
```

IS-frozen sign → OOS: median **SR_IS +0.137 → SR_OOS −0.040**, 45.6% of cells positive OOS,
corr(IS, OOS) = **−0.078**. Per-event OOS Wilcoxon p = 0.454.
19 cells "pass" the four gates but hold a position on 0.1–31% of days (median |exec_w| 0.054);
they are near-flat books, not signals.

**Verdict: null.** Per Rule 1 the ratio is 0.79, so no best cell may be quoted at all.

### A2 — release TIMING (the calendar, no value attached)

Scheduled releases are known months ahead, so this is PIT by construction — no revisions, no
publication lag. 18 event types incl. **FOMC** (which carries no value at all), offsets −2…+2 TW
trading days, windows `c2c`/`o2o`/`day`/`ongap`/`night`.

**Selection-free test first** — 320 (event × offset × window) Welch t-tests of event-day vs
non-event-day return, 2010–:

| window | n | mean t | sd t | \|t\|>1.96 | expected | KS vs N(0,1) |
|---|---|---|---|---|---|---|
| c2c | 80 | +0.169 | 0.94 | 3 | 4.0 | p=0.14 |
| day | 80 | −0.014 | 1.10 | 6 | 4.0 | p=0.64 |
| night | 80 | +0.038 | 1.08 | 5 | 4.0 | p=0.87 |
| ongap | 80 | +0.228 | 0.99 | 3 | 4.0 | p=0.12 |
| **all** | **320** | **+0.105** | **1.03** | **17** | **16.0** | **p=0.36** |

The t's are N(0,1) to the eye and to a KS test. Bonferroni at 320 tests needs p < 0.00016; **0
cells reach it.** There is no MTX event-day effect for any US release, FOMC included.

### A3 — and the calendar sweep's 70 "gate passers" are a GATE LEAK, not an effect

The 800-cell calendar sweep produces 70 four-gate passers (8.8%). **All 70 have sign +1**, 25 of
them on the `night` leg. The reason:

```
always-long, 2010-, real costs:   c2c +0.916   o2o +0.921   day +0.153   ongap +0.508
                                  night SR_net +1.146, SR_of_SR +1.764, positive_years 1.00, beta 1.00
```

The MTX **night session carries an unconditional +1.15 SR drift**. Buy-and-hold fails the
`|beta| < 0.15` gate at beta 1.00 — but **beta shrinks with exposure and `SR_of_SR` /
`positive_years` do not**, so a long-only mask over 5–10% of nights slides under the beta gate
while keeping the drift. Measured on **purely random masks with no information whatsoever**:

| fraction of nights held long | SR_net | SR_of_SR | beta | **4-gate pass rate** |
|---|---|---|---|---|
| 5% | 0.021 | 0.149 | 0.047 | **10%** |
| 10% | 0.285 | 0.277 | 0.093 | **15%** |
| 20% | 0.238 | 0.291 | 0.196 | 0% |
| 40% | 0.541 | 0.749 | 0.392 | 0% |
| 100% | 1.146 | 1.764 | 1.000 | 0% |

A random 5–10%-exposure long-night mask passes all four house gates **10–15% of the time**. The
calendar sweep's rate is **8.8%** — at or below the no-information rate. And the passers'
**beta per unit of exposure is +0.69** (q25 0.56, q75 0.92): they are ~70% a long index bet,
scaled down until the beta gate stops seeing it. (The A1 surprise passers read +0.06, i.e. they
are genuinely market-neutral — they just have no edge.)

> **This is a defect in the gate set, not in this ticket.** `|beta| < 0.15` is scale-evadable.
> The fix is to gate on **`beta / mean|exec_w|`**, which reads ~1.0 for any index bet at any
> exposure. Ten of thirty earlier "survivors" died of a disguised long-index bet at full
> exposure; this is the same failure hiding at 6% exposure. Filed as a follow-up.

---

## Part B — put OI as a standalone signal

`opt_put_mo_oi_selftanh_w60` (total monthly put OI → tanh z60) was **disabled 2026-08-24**: on
the corrected full-strike data its IS-honest sign is +1, the live sign is −1, IS SR is +0.16
either way. The ticket asks whether put OI can stand alone. Questions before signals.

### Q1 — is the source complete and comparable across the sample? **No.**

| year | median monthly put OI | median strikes |
|---|---|---|
| 2011 | 570,046 | 151 |
| 2017 | 537,318 | 166 |
| 2021 | 167,533 | 305 |
| 2026 | **63,493** | **864** |

Put OI fell ~9x while the strike grid grew ~6x. Any feature that counts strikes (HHI, strike
counts) reads TAIFEX product decisions. Worse, `corr(log put OI, log spot)` is **+0.06 in
2013–18 and −0.69 in 2019–26** — *the sign of the relation to price flips between eras*. That,
not bad luck, is why the total-OI signal could not identify a sign.

Every feature below is therefore **OI-weighted over strikes** (adding empty strikes changes
nothing) or a **ratio/growth** (scale-free): `put_cog` (OI-weighted K/S−1), `put_disp`,
`put_otm_share`, `put_far_share` (>5% OTM = crash hedging), `put_wall` (moneyness of the
largest-OI strike), `put_front_share`, `put_churn` (vol/OI), `put_oi_growth`, `put_cog_chg`,
`put_far_chg`, and `put_oi_total` as the control. Two panels: monthly-expiry only, and all
expiries. 2009-01 → 2026-09, TAIEX close as the moneyness reference.

### Q2 — is the "feature" actually a mirror of today's price? **For three of them, yes.**

`corr(feature_t, return_t)`: `put_cog_chg` **−0.69**, `put_far_chg` **+0.55**,
`put_otm_share` **+0.45**, `put_far_share` +0.30. Moneyness measured against today's spot moves
mechanically when spot moves — those series are mostly `−1 × today's return` wearing an
options-data costume. `put_disp` (+0.01), `put_front_share` (+0.02) and `put_oi_growth` are clean.

### Q3 — is the relation stable across eras? **Mostly not, and it is tiny where it is.**

Spearman IC vs the `c2c` return at t+2, by era: every feature is inside |IC| < 0.08 and most are
inside 0.05. Six of eleven flip sign across the four era blocks. Only `put_cog`, `put_disp` and
(all-expiry) `put_far_share` keep one sign in all four — and `put_disp`'s IC is +0.009 / +0.017
/ +0.063 / +0.053.

### Q4 — the gate sweep: **0 of 1,584 cells pass all four gates**

11 features x 2 panels x 6 operators x {20,60,120,252} x 5 variants, sign frozen on IS
(≤2016-12-31). 394 cells pass 1 gate, 1,117 pass 2, **73 pass 3, 0 pass 4.**

```
QNT-99 put-OI: 1584 cells = 11 source series x 144 transform-windows, n_eff ~ 30
(ICC(feature) = 0.36). Per-series median SR_net -0.011, 4/11 positive, Wilcoxon p = 0.465.
sd(SR_net) across cells = 0.218 vs SE(SR | 18y) = 0.239 (ratio 0.91)
  -> dispersion BELOW the noise floor - do NOT quote a best cell as a result.
IS -> OOS: med IS +0.002 -> med OOS -0.110; 34.7% of cells positive OOS.
```

**Every gate is failed on the beta gate, and the near-misses show why.** The best-scoring family
is `put_front_share` (share of put OI in the front expiry) — `robustz w60`, sign −1, SR_IS
**+0.97**, SR_OOS **+0.72**, SR_net +0.82, `SR_of_SR` 1.32, positive_years 0.94 — four numbers
that look like a signal. Its **beta is +1.27 at mean|exec_w| 1.93**: front-share sawtooths with
the monthly expiry cycle, so `robustz` of it is a persistent long tilt. The same holds for
`put_disp`'s best cells (beta +0.37…+0.42). The one cell that is genuinely beta-free —
`monthly|put_disp|dev|w60|c2c`, beta **+0.003**, SR_net 0.54, SR_OOS 0.79 — has `SR_of_SR` 0.354
and sits in a grid whose dispersion is under the noise floor, so under Rule 1 it is not
quotable as a result.

**Verdict: there is no standalone put-OI signal in this data.** The one lead worth keeping on
file is `put_disp` — the OI-weighted dispersion of put strikes — which is the only feature that
is simultaneously (a) not a mechanical mirror of price, (b) sign-stable across all four era
blocks, and (c) beta-free in its `dev` form. Its effect size is ~IC 0.03, i.e. below what this
grid can resolve.

---

## Artefacts (`signal_zoo/qnt99/`)

`event_inputs.py`, `event_sweep.py`, `event_timing.py`, `event_calendar_null.py`,
`gate_leak.py`, `put_oi_panel.py`, `put_oi_research.py`, `figures.py`, `explore_put_oi.py`;
`event_surprise_full.csv` (813), `event_calendar_sweep.csv` (800), `event_study.csv` (320),
`gate_leak_random_masks.csv`, `put_oi_features_{monthly,all}.csv`,
`put_oi_diagnostics.csv`, `put_oi_sweep_full.csv` (1,584);
`qnt99_summary.png`, `qnt99_gate_leak.png`.
