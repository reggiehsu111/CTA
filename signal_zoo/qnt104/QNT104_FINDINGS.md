# QNT-104 — CALL open interest and CALL/PUT-COMBINED OI as MTX signals

**Run 2026-09-02, lane `taiex-futures`, `/home/ubuntu/mtx`.** Nothing was published, no sign or
`recommended_variants` was chosen, no `*_signal_config` row was read or written. Read-only DB.

QNT-99 swept **put** OI alone and came back null. This ticket adds the two axes it did not
touch — the **call** side, and features that only exist when both sides are read together —
and adds a **second held-out block** so a cell has to survive twice.

---

## Design, stated before the run

**Power target (QNT-78 Rule 2 with the QNT-94 correction).** Target `d = 0.12` SR, `sd = 0.13`
→ `S_required = 2.33·(2.80·0.13/0.12)² = 22` source series. **S available = 26 NEW features**
(11 call-side + 15 combined) → `d_min = 4.26·0.13/√26 = 0.109`. Adequate for `d ≥ 0.12`, and
for nothing smaller. Raw `S` is optimistic and the realised redundancy was **measured, not
assumed**: eigenvalue `n_eff` of the per-feature PnL correlation matrix is **9.32 of 26**
(0.36·S, mean pairwise corr +0.074), close to QNT-94's 0.43·S rule of thumb. At that `n_eff`
the honest floor is `d_min = 0.169`.

**Feature construction.** Every constraint QNT-99 established carries over. Monthly **call** OI
fell 148M (2011) → 7.5M (2026 ytd) while the strike grid grew 55 → 539, so every feature is
either **OI-weighted over strikes** (adding empty strikes changes nothing) or a **ratio /
growth** (scale-free). Moneyness `m = K/S − 1` against the same-day TAIEX close; `far` = 5% OTM
on each side's own OTM direction.

* **call (11)** — `oi_total, cog, disp, otm_share, far_share, wall, front_share, churn,
  oi_growth, cog_chg, far_chg`
* **combined (15)** — `pcr_oi, pcr_vol, pcr_oi_chg, pcr_far` (tail-hedge PCR), `pcr_atm,
  cog_gap, cog_mid, far_asym, wall_gap, wall_mid, churn_ratio, oi_growth_diff, disp_ratio,
  front_diff, max_pain` (writer-payout minimiser, front monthly, ±15% band)

Two panels (`monthly`, `all`), regular session only (the after-hours rows carry no OI at all),
2009-01 → 2026-09 so the QNT-99 put grid stays comparable.

**Out-of-sample design.** Sign frozen on IS and never re-chosen.

| regime | variants | IS | OOS1 | OOS2 |
|---|---|---|---|---|
| `full` | c2c / o2o / day | 2009-01…2016-12 | 2017…2021 | **2022…2026 (second held-out block)** |
| `night` | + ongap / night (needs `night_close`, from 2017-05-16) | 2017-05…2022-12 | 2023…2026 | — |

Real costs (70 TWD + 4e-5 per side, priced off each window's own entry), roll-adjusted legs from
`_base.py`, gates via `cta.house_gates` (QNT-100 `beta_per_w`) with the raw rule kept alongside.

---

## Step 2 — is the feature a price mirror, and is it era-stable?

`corr(feature_t, return_t)` flags **8 of 26** as mechanical mirrors of today's return
(`|r| > 0.35`): `call_cog_chg` (−0.82), `call_far_chg` (−0.67), `pcr_oi_chg` / `oi_growth_diff`
(±0.59, and note these two are the same series with opposite sign — the 26 features are really
25), `pcr_atm` (−0.46), `max_pain` (−0.43), `call_otm_share` (−0.42), `far_asym` (+0.38).
Moneyness measured against today's spot moves when spot moves. Same failure as the put side.

Non-stationarity is on the call side too: `corr(call_oi_total, log spot)` is **+0.11 in
2009-17 and −0.64 in 2018-26** — the sign of the price relation flips, exactly as it does for
put OI. Nine features keep one IC sign across all four era blocks; every IC is inside |0.09|.

## Step 3 — the gate sweep: 9,984 cells, 18 four-gate passers, and the null says that is noise

`26 features × 2 panels × 6 transforms × {20,60,120,252} × variants` = **3,744 (full) + 6,240
(night)**.

```
QNT-104 call+comb OI [full]: 3744 cells = 26 source series x 144 transform-windows, n_eff ~ 65
(ICC(feature) = 0.39). Per-series median SR_net -0.019, 10/26 positive, Wilcoxon p = 0.980.
sd(SR_net) across cells = 0.212 vs SE(SR | 18y) = 0.239 (ratio 0.89)
  -> dispersion BELOW the noise floor - ranking cells ranks noise.
```

Per QNT-78 Rule 1 the ratio is 0.89 < 1, so **no best cell may be quoted at all.** The night
regime reads the same (ratio 0.83, Wilcoxon p = 0.980), and so does each family separately
(call 0.86, comb 0.90).

**IS does not carry.** `corr(SR_IS, SR_OOS1) = +0.354`, but `corr(SR_IS, SR_OOS2) = +0.018` and
`corr(SR_OOS1, SR_OOS2) = −0.214`. Per-feature medians: IS +0.065 → OOS1 −0.047 → OOS2 −0.074;
the OOS2 per-feature median is **significantly negative** (8/26 positive, Wilcoxon p = 0.041).
Cells positive in IS *and* OOS1 *and* OOS2: **1.15%, against 0.99% expected** if the three
blocks were independent coin flips at the observed marginal rates.

**The 18 passers are at the no-information rate.** 40 circular-shift reps (each feature rolled
by a random offset — preserves its autocorrelation, distribution and expiry sawtooth, destroys
only the alignment with returns), identical grid:

| | observed | shift null | p |
|---|---|---|---|
| four-gate passers (`beta_per_w`) | **9** / 3,744 | mean 4.5, sd 5.0, range 0–21 | **0.200** |
| four-gate passers (raw `\|beta\|`) | 23 | mean 5.8, sd 6.2 | 0.025 |
| best cell `SR_net` | +1.184 | mean +0.700, max +0.935 | 0.000 |

The best cell being above the null is not a signal — it is `max_pain|rankc`, one of the flagged
price mirrors, with **beta 0.52 / `beta_per_w` 0.79**: a scaled-down long-index bet. Same for
the `call_front_share|robustz|w252` family (`beta_per_w` +1.19…+1.30 at exposure 1.85) — the
call-side twin of the `put_front_share` trap QNT-99 found, front-share sawtooths with the expiry
cycle so its z-score is a persistent long tilt. **The QNT-100 gate catches both**; the
pre-QNT-100 raw-beta rule would have let 23 cells through against a null of 5.8.

**Best feature, search-corrected.** `pcr_oi` is the best of the 26 (per-feature median SR_net
+0.284, and positive in all three blocks: IS +0.391 / OOS1 +0.081 / OOS2 +0.313). But
`SE` of a per-feature median is `0.239/√(1/ICC) = 0.149`, so the expected best of 26 draws under
the null is **+0.309** — above what was observed. `pcr_oi` is what a null grid this size produces.

## Steps 5–8 — combining the two sides, and the one thing that did not die immediately

Cell ranking is noise, so the ticket's "combine both" was tested as **equal-weight baskets with
no member selection**: every feature that is not a price mirror, `robust_z(w60)` → `tanh`, sign
frozen on IS, four baskets (`callEW`, `combEW`, `putEW`, `allEW`) × 2 panels × 3 variants = 24.

`combEW|o2o` passes all four house gates in both panels — `all` panel: SR_IS 1.151 → OOS1 0.705
→ OOS2 0.409, `SR_net` 0.794, `SR_of_SR` 1.063, positive_years 0.94, `beta_per_w` −0.010. And
across all 24 baskets the **held-out** median is `SR_OOS(net) = +0.352` over 9.3 years with the
sign frozen in 2016, 19 of 24 positive, `n_eff` 6.05 → **t = 2.64, p ≈ 0.008** (t = 2.11 with a
1.25× median-vs-mean penalty). A 200-rep basket shift null agrees on the aggregate.

**It is the index drift.** Buy-and-hold over the same held-out window is `SR c2c +1.148`,
`o2o +1.182`; the baskets run `beta_OOS` +0.01…+0.18 and exposures 0.08–0.17 — *below* the 0.31
floor at which `beta_per_w` can distinguish a directional book from a neutral one, so the beta
gate cannot see it. Hedging each basket's held-out PnL against the same-variant buy-and-hold and
re-scoring the residual:

| | median across 24 baskets | n_eff | t | p |
|---|---|---|---|---|
| `SR_OOS` (net) | **+0.352** | 6.05 | 2.64 | 0.008 |
| `SR_OOS` hedged (alpha) | **−0.071** | 6.71 | −0.57 | 0.572 |

11 of 24 have positive hedged alpha. The best single hedged basket (`all|combEW|o2o`, +0.554)
sits at the expected best of 6.7 effective draws (+0.57). **The entire out-of-sample basket
result is MTX's unconditional long drift, collected at 10% exposure** — QNT-99 Part A3's leak,
found again at the basket level and killed by a hedge rather than by a gate. Note the corollary:
no basket-level result on this instrument should be reported without the hedged alpha beside it,
because at these exposures no beta gate can reject it.

## Verdict

**Null.** Call OI adds nothing that put OI did not, and the two sides combined add nothing
either. 0 of 26 features clears the noise floor; the 18 four-gate passers are at the shift-null
rate (p = 0.20); the aggregate held-out positive is entirely index beta. Everything with a
headline SR is a moneyness mirror of today's price, or a front-share/max-pain expiry sawtooth
wearing a long-index bet.

Open leads, both below what this grid can resolve: `pcr_oi` (only feature positive in all three
blocks, but at the expected best-of-26) and QNT-99's `put_disp`. Both are IC ≈ 0.03. Per QNT-78
Rule 2 the lever is **more source series, not more transforms** — TXO OI is now exhausted on
both sides, so the next series must come from elsewhere (volume/settlement-based positioning,
`tw_large_trader` option legs, or the after-hours session once TAIFEX populates its OI).

## Artefacts (`signal_zoo/qnt104/`)

`oi_panel.py` (features) · `oi_sweep.py` (diagnostics + 9,984-cell grid) · `oi_analyse.py`
(headline, n_eff, OOS decay) · `oi_null.py` (40-rep circular-shift null) · `oi_combine.py` +
`oi_combine_null.py` (baskets, 200-rep null) · `oi_combine_honest.py` (error bar + hedged alpha)
· `figures.py` → `qnt104_summary.png`. CSVs: `oi_diagnostics`, `oi_sweep_full`, `oi_sweep_gated`,
`survivors_{full,night}`, `null_control`, `null_real_vectorised`, `oi_combos`, `combine_null`,
`basket_oos`, `basket_alpha`.
