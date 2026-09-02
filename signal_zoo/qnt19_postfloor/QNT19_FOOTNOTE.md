# QNT-19 footnote — the `load_macro_tw` +1 PIT floor, and what it moved

**Approved by Reggie on Linear QNT-19, 2026-09-01.** This is the canonical note
referenced from `cta/global_macro.py`, `cta/signals/_base.py`,
`tools/check_macro_shift_overrides.py`, and from the footnote comments on
QNT-10 / QNT-14 / QNT-15 / QNT-16 / QNT-18 / QNT-25.

## What changed

`cta.load_macro_tw` (and `load_macro_yoy_tw`) now resolve
`max(macro_series_meta.pub_lag_days, cta.MACRO_MIN_PUB_LAG_DAYS)` with
`MACRO_MIN_PUB_LAG_DAYS = 1`, so a macro observation stamped with reference
date `D` first becomes visible on TW trading index `D+1`.

| loader | convention | US obs of D lands on |
|---|---|---|
| `cta.load_us_index_tw` | `pit_lag_days = 1` (hard floor, raises below 1) | TW index D+1 |
| `cta.load_macro_tw` **(before)** | `available_from = D + pub_lag_days`, all 12 daily series carry 0 | TW index **D** |
| `cta.load_macro_tw` **(now)** | `max(pub_lag_days, 1)` | TW index D+1 |

The floor binds on exactly the **12 daily FRED series** (all `pub_lag_days = 0`).
The other 31 series carry lags of 7–150 days, so it is a no-op for them —
verified: 0 of 1512 slow-input cells moved.

`enforce_floor=False` reproduces the pre-QNT-19 alignment; an *explicit*
`pub_lag_days` below the floor raises. Guard: `tools/check_macro_shift_overrides.py`.

The floor is a **calendar-day** floor, not a one-row shift, so it is close to but
not identical to running the old loader one variant-shift later — the two
disagree on ~2.5% of rows, all around TW holidays and long weekends. Every number
below is against the real loader, not that proxy.

## Method for the re-run

Each published script was executed **verbatim** with only its `OUT` path
redirected into `signal_zoo/qnt19_postfloor/`. No pre-floor directory was
overwritten; both sides are on disk. Both grids end `2026-08-31`, `n_bars` max
6192 on each side, so the comparison is apples-to-apples.

Diff script: `compare_prefloor.py` → `qnt19_prefloor_vs_postfloor.csv` (1,530
cells that moved). Effective-n recheck: `qnt25_recheck.py` →
`qnt25_recheck_grid2.csv`.

## What moved

### QNT-10 / QNT-12 — 522-cell standalone macro sweep — **conclusion unchanged**

144 of 522 cells moved (8 of 29 source series are daily US-close: `krw_usd`,
`twd_usd`, `us_breakeven_5y5y`, `us_dgs5`, `us_dxy_broad`, `us_real_10y`,
`us_term_premium_10y`, `wti`). Median ΔSR_full **+0.000**, mean +0.005,
max |Δ| 0.242, 15 IS-sign flips.

Five-gate passers **0 → 0**. Best `SR_of_SR` **0.577 → 0.577**
(`igrea_level|bdtanh|w252` — monthly, untouched). The headline stands.

### QNT-14 — 198-cell daily-macro window grid — **headline collapses**

This is the grid the floor was built from, and the one it changes.

| statistic (regime=full) | pre-floor | post-floor |
|---|---|---|
| dSR `day@1 − c2c@2`, per **cell** | +0.0708, 64% win, binom p<0.001 | **+0.0130, 53% win, p=0.523** |
| dSR `day@1 − c2c@2`, per **series** | +0.089, 8/11, p=0.227 | +0.038, 6/11, p=1.000 |
| dSR `o2o@2 − c2c@2`, per **series** | +0.041, 9/11, wilcoxon p=0.024 | **−0.008, 4/11** |
| 4-gate passers `day` | 2 | **0** |
| corr(SR_c2c, SR_day) across cells | 0.415 | 0.176 |

QNT-18 already retracted the day-window effect as a per-cell-n artefact. The
floor removes the *cell-level* statistic that motivated it too: the gain was
riding on the day leg reading a US close from the same TW date. Nothing on this
grid passes four gates on `day` or `o2o` post-floor.

Per-variant, regime=full: c2c ΔSR median +0.022 (61% win, max|Δ| 0.523),
day −0.020 (43%, 0.317), o2o −0.027 (31%, 0.533). Regime=night moves more —
`day` 4-gate 37 → 10, `ongap` 11 → 6 — but that regime is a 10-year subsample.

### QNT-15 — regime overlay — **bit-identical**

Re-ran `regime_overlay.py` post-floor; `regime_cells.csv`, `regime_deltas.csv`
and both PNGs are byte-identical to the pre-floor artifacts (same md5). Its
regime inputs are `igrea`, `us_stlfsi`, `epu_global`, `nfci` — all weekly or
monthly, all `pub_lag_days >= 7`. Conclusion untouched.

### QNT-16 — EW3 macro combination — **unchanged to 3 dp**

`igrea`, `epu_global`, `kr_kospi` are all monthly. EW3 `SR_full` 0.605,
`SR_of_SR` 0.542, positive years 0.760, beta −0.065 — identical on both sides.

### QNT-18 — 378-cell slow-input grid — **unchanged**

0 of 1512 cells moved. All 21 slow inputs carry `pub_lag_days >= 7`.

### QNT-25 — effective-n re-report — **conclusions hold, one strengthens**

Grid 1: `ICC(series)` on `SR_full` 0.615 → 0.622 (n_eff 45.6 → 45.1); noise-floor
ratio `sd(SR)/SE(SR)` 0.69 → 0.68 on `SR_full` and 0.73 → 0.75 on `SR_of_SR` —
still below 1, so ranking cells still ranks noise. Per-series median `SR_IS`
+0.164 (29/29 positive) vs `SR_OOS` −0.056 (11/29, p=0.265): the sign-selection
reading is unchanged.

Grid 2: `ICC` on the paired dSR rises 0.495 → 0.681 (n_eff 21.0 → 15.7), and the
per-series median falls from +0.41 SE to **+0.15 SE**. QNT-25 said the QNT-14
headline was noise at honest n; post-floor it is noise at cell n as well.

## Net read

Aggregate cost of the floor is ~zero — no candidate crosses or falls off a house
gate in the standalone sweep, the combination, the slow grid or the regime
overlay. Its one real consequence is that the QNT-14 window-timing result, already
retracted on effective-n grounds by QNT-18/QNT-25, is now also gone at face value.
That is the correct direction for a PIT fix to push a result.
