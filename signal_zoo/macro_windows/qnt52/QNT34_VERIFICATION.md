# QNT-34 verification (2026-09-01)

QNT-34's scope was executed by QNT-52 (filed as its redo after QNT-34 died 3x on quota).
This file records an independent re-verification, not a second re-run.

## Preconditions
- asset: 6,256 rows to 2026-09-01, `back_open` present, volume 121,934, open != night_open.
- `macro_window_sweep.py` imports `_o2o_ret` / `_prev_open_cc` from `cta/signals/_base.py`;
  the local `_prev_o_adj` close-measured approximation is gone. registered/slow/floor_cost
  exec that header and inherit it.

## B&H o2o reference (reproduced)
raw 0.4841 -> exact 0.6669 ; approximation 0.6829 (close on B&H, poor per-day).

## Full grids, approx (qnt52/approx/) vs exact (current CSVs)
macro `window_sweep_full.csv`, 1,386 cells, 23 numeric cols:
  cells changed -- c2c 0/396, day 0/396, ongap 0/198, **o2o 396/396**
  SR_net mean  c2c +0.0842->+0.0842 | day +0.0887->+0.0887 | ongap +0.1208->+0.1208
               o2o +0.0918->+0.0899 (mean|d| 0.0230, best cell +0.9374->+0.9552)
  all-4-gate passes 31 -> 27 ; by variant c2c 9->9, day 6->6, ongap 6->6, **o2o 10->6**

registered `registered_window_sweep.csv`, 99 cells:
  cells changed -- **o2o 22/22**, c2c/day/night/noonpause/ongap all 0
  SR_net mean o2o +0.5629->+0.5640 (mean|d| 0.0146); gate passes 26 -> 26

The 990 unchanged macro cells and 77 unchanged registered cells are bit-identical
(max abs diff 0.0) -- that identity is the check that the patch is isolated to o2o.

## noonpause
Bit-identical before/after. Its `back_close` denominator was already exact and equals
`_base._next_front_close`, so the 21%-of-variance spread QNT-21 measured on that leg was
never carried by these sweeps. No fix needed.

## Does any QNT-14 conclusion move? No.
Per-series mean dSR(o2o - c2c), full regime: -0.0103 (6/11 positive, wilcoxon p=0.577)
approx -> -0.0125 (5/11, p=0.638) exact. Night regime +0.0254 (8/11) -> +0.0238 (7/11).
The o2o edge was already ~zero after the QNT-19 PIT floor and stays ~zero after the roll
fix. The only material movement is the 4 macro o2o cells that lose their gate pass, i.e.
~40% of the gated o2o population were roll-approximation artefacts -- which reinforces
rather than revises QNT-18's retraction of the day-window effect.

Caveat: the CSVs were produced at 10:07-10:11 UTC on an asset ending 2026-08-31; the asset
now carries one further bar (2026-09-01). One bar out of ~6,250 moves no conclusion here,
so the grids were not regenerated for it.
