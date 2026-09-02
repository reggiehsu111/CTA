"""LT top-10 buy-minus-sell → tanh of 15-day self-z-score.

Discovered in the regime-factor sweep (2026-08-08). While the sweep was hunting
for regime-conditional PnL, the LT top-10 (前十大交易人) net-flow features
surfaced as strong STANDALONE signals — outperforming their existing gated
cousin `lt_top10npct_signth_w60` by ~2× SR.

The existing #7 signal `lt_top10npct_signth_w60` uses:
  sign_thresh(z_60(top10_net_pct), 0.5)  → discrete {-1, 0, +1}, ~40% coverage.

This new signal is a denser/faster parameterization of the same data source:
  tanh(z_15(top10_buy - top10_sell))  → continuous, always-on.

Robustness (Simulate on MTX c2c, net of 20 TWD + 2bp costs):
  - Full sample SR = +0.79  (n = 4,058, 2010-2026)
  - H1 SR = +0.52,  H2 SR = +1.03  (both halves positive, monotonic)
  - w-window sweep (10-60): every window/feature combo positive in both halves
  - Peak: bs_diff at w=15; adjacent w=10, w=20 both SR > 0.68
  - tanh sensitivity 0.5→1.5 monotone: SR 0.75 → 0.79 → 0.83

Independence vs existing #7 `lt_top10npct_signth_w60`:
  - Position corr = 0.68,  PnL corr = 0.69
  - Residual SR after regressing out #7 = +0.73 (89% of raw +0.81)
  - Genuinely additive — different parameterization captures different flow horizons

Data path: TAIFEX 前十大交易人 daily → tw_large_trader RDS table
(ingested by tw_taifex_lt Lambda, post-market same day → PIT safe with shift(1)).
"""
import numpy as np
import pandas as pd

from ._base import Signal, register
from ._operators import selfz


@register
class LtTop10BsdiffSelftanhW15(Signal):
    name           = "lt_top10_bsdiff_selftanh_w15"
    cn_name           = "十大交易人買賣差（連續式）"
    cn_short           = "十大B"

    description    = ("LT top-10 buy-minus-sell → tanh(z-window 15). "
                      "Continuous always-on companion to the discrete #7 "
                      "sign_thresh variant.")
    cn_description = ("台指期十大交易人買賣淨部位 15 日 Z 分數 → tanh 連續版本，"
                      "為第 7 號離散版本的密集版")
    sources        = ("large_trader", "mtx_1d")
    cadence        = "daily_15_45_tpe"
    live_date      = "2026-08-08"
    sign           = +1                                    # top10 net long → LONG MTX
    variants       = ("c2c", "o2o", "day", "noonpause", "night", "ongap")
    recommended_variants = ("c2c",)

    def compute_raw(self, ctx) -> pd.Series:
        buy  = ctx.lt("top10_buy")
        sell = ctx.lt("top10_sell")
        raw  = buy - sell
        return np.tanh(selfz(raw, 15))
