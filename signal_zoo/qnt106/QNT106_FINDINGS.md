# QNT-106 — 台股 market internals as MTX inputs

**Verdict: null.** A new source-series axis (29 series, 4 families, 16-17 years) fails
every test. Gate passers are at the circular-shift no-information rate.

## What was tried first, and why it was dropped

The ticket's first option was "download more data". The obvious target was the TAIFEX
**三大法人-區分各期貨契約** history: `tw_three_majors` starts **2023-06-26**, so the
whole 外資 台指期未平倉 axis is stuck at 3.2 years and can never clear the `n_years >= 5`
gate. It cannot be extended. Both TAIFEX endpoints — `futContractsDateExcel` (per-day) and
`futContractsDateDown` (range CSV) — serve a **rolling 3-year window** and return the
631-byte 日期時間錯誤 page for older dates (probed 2015-01-05 and 2015-01-05..09, both
631 bytes). Buying the history or a third-party mirror is the only route; that spends money
and is Reggie's call.

## The axis that was swept

Everything MTX has been swept on so far is either exogenous macro (QNT-12/14/25) or the
TXO options book (QNT-99/104). The **cash market's own internals** had never been used.
29 daily series, aggregated **in SQL** (so NULL propagates — CLAUDE.md rule 1), TSE common
shares only, returns adjusted with `tw_spot_adj` because TW ex-dividend dates cluster in
Jun-Aug and would otherwise punch a seasonal hole in every breadth series:

| family | n | series |
| -- | -- | -- |
| breadth | 6 | `adv_share` `upvol_share` `above_ma20` `above_ma60` `ext_net` `n_traded` |
| xsec | 3 | `xs_disp` `xs_skew` `ew_minus_aw` |
| liquidity | 6 | `log_amount` `amt_hhi` `log_avgtrade` `mean_range` `mean_spread` `close_loc` |
| flow | 7 | 外資/投信/自營 net + gross + breadth (`tw_institutional`) |
| leverage | 7 | 融資餘額/週轉/券資比/融券 + 借券賣出 (`tw_margin_summary`, `tw_margin_short`) |

Panel validated against the obvious prior: `adv_share` has `corr_same_day = +0.756`,
`upvol_share = +0.820` — the breadth aggregation is doing what breadth should.

**PIT.** Every series is a Taiwan EOD print (spot/margin 15:00-21:00 TPE, 三大法人
15:00-16:00). `c2c`/`o2o` shift(2), `day`/`ongap` shift(1) — all safe. **`night` is
excluded**: `night_open[t]` is 15:00 of t-1, the same afternoon the print lands, so
shift(1) into the night session is not safely after publication.

**Grid.** 29 series x 6 transforms x 4 windows x 3-4 variants x 2 regimes = **4,872 cells**.
`full` IS 2009-17 | OOS1 2018-21 | OOS2 2022-26; `night` (2017-05-16..) IS ..2022 | OOS 2023-26.
Sign frozen on IS, never re-chosen. Signals tanh-normalised to [-1,1] (canonical harness),
real costs (fixed 70/side + 4e-5), roll-adjusted legs from `_base.py`.

**Pre-run power target (QNT-78 rule 2, stated in the script before the run):** target
d = 0.12 SR, sd = 0.13, `S_required = 2.33*(2.80*0.13/0.12)^2 = 22`; S = 29 gives
d_min = 0.103.

## Results

**Reporting line (QNT-78 rule 1).** All four slices are below the noise floor, so
**no best cell may be quoted**:

```
[all]         4872 cells = 29 series x 168 tw, n_eff ~ 128 (ICC .22). median SR_net -0.049,
              11/29 positive, Wilcoxon p = 0.455. sd .254 vs SE(SR|10y) .288 -> ratio 0.88
[full]        2088 cells, ICC .39. median SR_net +0.004, 15/29 positive, p = 0.949.
              sd .226 vs SE(SR|18y) .242 -> ratio 0.93
[night]       2784 cells. median -0.064, 11/29, p = 0.230. ratio 0.85
[non-mirror]  3360 cells, 20 series. median -0.048, 8/20, p = 0.956. ratio 0.89
```

**The two held-out blocks disagree in sign.** With the sign frozen on IS, the per-series
median SR_net is **-0.211 in OOS1 (2018-21, p = 0.000)** and **+0.151 in OOS2 (2022-26,
p = 0.005)**. Two significant results pointing opposite ways is the signature of era luck,
not of a relation. `corr(SR_IS, SR_OOS2)` across series is only +0.228.

**Four-gate passers are noise.** 66 of 4,872 pass all four house gates. 40-rep
circular-shift null (source series shifted, sign re-chosen on IS, re-scored):

| regime | observed | null mean | null sd | p |
| -- | -- | -- | -- | -- |
| full | 3 | 6.1 | 9.9 | 0.390 |
| night | 63 | 70.2 | 83.0 | 0.415 |

Observed is *below* the null mean in both. The vectorised null scorer reproduces the
sweep's counts exactly (3 / 63), so the two code paths agree.

**Baskets are negative before and after hedging.** 15 selection-free equal-weight family x
variant baskets, held out 2018-2026: median SR_OOS **-0.246**, and after hedging each
against the same-variant buy-and-hold, median alpha SR **-0.410, median t = -1.19**. Betas
are 0.00-0.14, so unlike QNT-104 this is not a disguised index bet — it is just negative.

## Two reusable facts

1. **QNT-94's `n_eff ~ 0.43*S` replicates on a completely unrelated data family.** Measured
   from the 29 per-feature PnLs: **n_eff = 13.60 = 0.47*S**, mean |corr| 0.144. Macro
   (0.43), TXO OI (0.36) and now cash internals (0.47) all land near the same coefficient.
2. **The pre-run power claim was optimistic by 1.45x.** Realised
   `d_min = 4.26*0.13/sqrt(13.60) = 0.150` against the 0.103 promised on raw S = 29. The
   run was powered for d >= 0.15 only. Rule 2 should probably quote *both* numbers: the raw-S
   d_min and a `sqrt(1/0.45)` = 1.49x-inflated honest one.

## Data-hygiene finding

`CLAUDE.md` says `tw_institutional` **"stops 2026-01-09, known gap"**. That is **stale**:
the table is complete to **2026-09-01**, 160 of 160 spot trading days in 2026 have
institutional rows, no interior gap. The table that actually stops near that date is
**`tw_trading_flags` (max 2026-01-12)** — the note looks mis-attributed. Not fixed here
(CLAUDE.md is Reggie's).

## Artefacts

`build_panel.py` (SQL aggregation) - `features.py` (29 series) - `sweep.py` (diagnostics +
grid) - `analyse.py` (headline, per-series, n_eff, baskets) - `null.py` (circular-shift
control) - `figures.py`. Data: `raw_*.csv`, `features.csv`, `diagnostics.csv`,
`sweep_full.csv`, `sweep_gated.csv`, `sweep_pnl.pkl`, `per_series.csv`, `baskets.csv`,
`null_control.csv`, `null_draws.csv`, `qnt106_summary.png`.
