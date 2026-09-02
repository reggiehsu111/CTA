"""Month-turn calendar drift: LONG MTX for the last 5 + first 10 TW days of each month.

Discovered in the CPI/NFP macro-event research (see nfci_research.ipynb §11):
after diagnosing that "CPI × vol at offset 10" and "NFP × ret_5d at offset 10"
signals were secretly capturing a month-end/beginning MTX drift, we tested the
pure calendar signal directly and found:

  - Best config: LONG MTX from d-5 (before month-end) to d+10 (from month-start)
  - SR all-days:  +0.94   (vs MTX buy-and-hold +0.80)
  - ann_ret:      +14.82%
  - vol:          15.7%
  - max_dd:       -35.9%
  - Coverage:     73%

Why it works (well-documented in equity-market literature):
  1. Retirement plan / mutual fund inflows land near month-boundaries
  2. Institutional window-dressing rebalancing into month-end
  3. Retail flows biased toward paycheck cycles (~1st/15th)
  4. Options expiry effects (3rd Friday clusters near mid-month)

Robustness checks (§10 of the research):
  - 8-neighbor day-parameter sweep: all cells SR 0.5-1.0 (no cliff-edge)
  - Every conditional variant (momentum, contrarian, VIX regime, volume regime)
    scored WORSE than the unconditional base
  - d5-from-end is statistically significant standalone (t = +2.64)
  - Near-zero correlation with all 9 other live signals (max |ρ| < 0.15)

No external data dependency — pure calendar computation.
"""
import pandas as pd

from ._base import Signal, register


# ── Tunables ─────────────────────────────────────────────────────────────
_LAST_N_FROM_END   = 5       # LONG on last N TW days before month-end
_FIRST_N_FROM_START = 10     # LONG on first N TW days from month-start


@register
class MonthTurnDrift(Signal):
    name           = "month_turn_drift_d5_d10"
    cn_name           = "月底月初日曆漂移"
    cn_short           = "月轉"

    description    = ("Pure calendar drift: LONG MTX for last 5 + first 10 TW "
                      "days of each month. No external data.")
    cn_description = ("月底月初日曆效應：每月最後 5 個交易日 + 下月前 10 個交易日"
                      "做多 MTX，捕捉法人結算、月結金流、退休金流等季節性行情")
    sources        = ("mtx_1d",)                       # only needs TW trading calendar
    cadence        = "daily_15_45_tpe"
    live_date      = "2026-08-09"
    sign           = +1                                 # LONG during window
    variants       = ("c2c", "o2o", "day", "noonpause", "night", "ongap")
    recommended_variants = ("c2c",)
    pre_normalized = True                               # {0, +1} binary; skip tanh(rolling_z)

    def compute_raw(self, ctx) -> pd.Series:
        # Get TW calendar (from asset index)
        tw = pd.DatetimeIndex(ctx.tw_index).sort_values()

        # For each date, compute TW-day position within its calendar month
        s = pd.DataFrame({'date': tw})
        s['ym'] = s.date.dt.to_period('M')
        s['td_from_start'] = s.groupby('ym').cumcount() + 1
        s['td_from_end']   = s.groupby('ym').cumcount(ascending=False) + 1

        # LONG on last N days of month OR first M days of month
        in_window = ((s.td_from_end   <= _LAST_N_FROM_END) |
                     (s.td_from_start <= _FIRST_N_FROM_START))

        return pd.Series(in_window.astype(float).values, index=tw, name=self.name)
