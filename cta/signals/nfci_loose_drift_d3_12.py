"""NFCI loose-shock delayed drift: LONG MTX for 10 days starting day+3 after a
weekly NFCI Δ_1w < −σ_52w loose shock.

Discovered in `mtx/nfci_research.ipynb` § 10.7:
  - Post-loosening TAIEX rallies for 10-15 days (delayed drift, not reversal).
  - Best d_start × d_end config: (3, 12) — SR 1.62 on active days, SR 0.72 all-days,
    ann_ret 27.4% at 16.8% active-day vol, max_dd −14.6%, ~19% coverage.
  - PIT-safe, and NOT dependent on any chain-trigger (QNT-46). The Chicago Fed
    publishes the Friday-dated value the FOLLOWING Wed 08:30 ET; _PIT_LAG_DAYS=6
    maps event Friday F to the first TW trading day >= F+6 — the Thursday after
    publication. _FF_OFFSET=1 turns the signal on the next TW day (Fri) and c2c
    shift(2) enters the position on Tue, so every shift>=1 variant reads a value
    that was public at least 3 TW days before fill.
  - The chain-trigger the ORIGINAL docstring cited was dead for 8 days and the
    claim that the signal "lands by Wed 20:35 TPE" was false in that window.
    us-nfci-ingest was retired 2026-08-24 and folded into us-macro-ingest, which
    kept invoking mtx-signal-runner with source="us_macro" — a key no signal
    declares and which is absent from ALL_SOURCES, so the runner raised
    ValueError on every fire (async, so it never surfaced).
    FIXED AND DEPLOYED 2026-09-01 15:50 UTC: us-macro-ingest now routes one invoke
    per table whose max(date) advanced (source="nfci" for us_nfci, nothing for the
    other 8) — QuantResearch/tools/lambda/us_macro_ingest/{lambda_function.py,
    test_chain_plan.py,deploy_code_only.sh}. First live proof lands on the next
    Wednesday NFCI publication; until one is observed, treat the 15:31 TPE
    "mtx_1d" run as the trigger you can count on.
    Either way this is a LATENCY question, not a correctness one: the c2c fill is
    Tue and _PIT_LAG_DAYS=6 already puts publication 3 TW days ahead of it.
  - Recommended variant remains c2c for the tightest fit to the backtest that
    discovered the signal.
"""
import numpy as np
import pandas as pd

import cta
from ._base import Signal, register


# ── Tunables (matched to § 10.7 winner) ──────────────────────────────────────
_SIGMA_WINDOW      = 52      # weeks of rolling std for shock threshold
_SHOCK_MULTIPLIER  = 1.0     # Δ_1w < -1.0 × σ_52w = loose shock
_PIT_LAG_DAYS      = 6       # calendar-day lag from Fri value → first usable TW date
_FF_OFFSET         = 1       # signal turns on TW day (event + 1) — first day after event
_FF_LIMIT          = 10      # forward-fill for 10 more TW days (11 total)


@register
class NfciLooseDriftD3D12(Signal):
    name           = "nfci_loose_drift_d3_12"
    cn_name           = "金融條件寬鬆後漂移"
    cn_short           = "NFCI"

    description    = ("NFCI loose shock (Δ_1w < -σ_52w) → LONG MTX for 10 days "
                      "starting event+3 (delayed post-loose drift)")
    cn_description = ("Chicago Fed NFCI 週變化跌破 −σ_52w 時，事件後 3–13 個交易日做多 MTX，"
                      "捕捉美國金融條件放鬆後 TAIEX 延續 2 週的上漲慣性")
    sources        = ("nfci", "mtx_1d")
    cadence        = "weekly_wed_2035_tpe"   # when NFCI LANDS in us_nfci;
                                             # recompute is the 15:31 TPE mtx_1d run (QNT-46)
    live_date      = "2026-08-08"
    sign           = +1                       # LONG MTX during window
    variants       = ("c2c", "o2o", "day", "noonpause", "night", "ongap")
    recommended_variants = ("c2c",)
    pre_normalized = True                     # {0, +1} binary; skip tanh(rolling_z)

    def compute_raw(self, ctx) -> pd.Series:
        # 1) Weekly NFCI Δ_1w and its rolling 52w std → loose-shock Friday dates
        nfci_wk  = ctx.nfci_weekly("nfci")
        d1w      = nfci_wk.diff()
        sigma_52 = d1w.rolling(_SIGMA_WINDOW, min_periods=13).std()
        loose_fri = d1w[(d1w < -_SHOCK_MULTIPLIER * sigma_52) & sigma_52.notna()].index

        # 2) Map each Fri event to its first-usable TW trading date (Fri + 6d)
        tw = pd.DatetimeIndex(ctx.tw_index).sort_values()
        loose_mask = pd.Series(0, index=tw, dtype=int)
        for fri in loose_fri:
            pos = tw.searchsorted(fri + pd.Timedelta(days=_PIT_LAG_DAYS), side="left")
            if 0 <= pos < len(tw):
                loose_mask.iloc[pos] = 1

        # 3) EventFFill: feature = constant +1, offset=+1 day, limit=10 more days
        cta.set_active_asset(ctx.asset)
        feat   = pd.Series(1.0, index=tw)
        raw    = cta.EventFFill(feat, loose_mask, offset=_FF_OFFSET, limit=_FF_LIMIT)

        # 4) NaN outside event windows → 0 (no position); +1 during window
        return raw.fillna(0.0)
