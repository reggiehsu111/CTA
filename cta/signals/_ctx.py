"""
Compute context handed to every signal's `compute_raw`. Provides lazy,
cached loaders so multiple signals sharing a source don't refetch.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import cta


class Context:
    """Per-run cache of loaded data. Instantiate once, pass to every signal."""

    def __init__(self, asset: pd.DataFrame):
        self.asset = asset            # e.g. cta.load_asset('mtx', '1d')
        self.tw_index = asset.index   # tw trading calendar

    # ── large-trader (TX) ───────────────────────────────────────────────
    @lru_cache(maxsize=None)
    def lt(self, series: str) -> pd.Series:
        """LT metric for commodity TX. Cached per (series)."""
        return cta.load_large_trader("TX", series).astype(float)

    @property
    def lt_log_ratio(self) -> pd.Series:
        b = self.lt("top10_buy"); s = self.lt("top10_sell")
        return np.log(b.replace(0, np.nan) / s.replace(0, np.nan))

    # ── options (TXO) ───────────────────────────────────────────────────
    @lru_cache(maxsize=None)
    def opt_total(self, metric: str, side: str, group: str) -> pd.Series:
        """cta.load_option_daily_total wrapper."""
        return cta.load_option_daily_total(metric, side, group).astype(float)

    # ── US indexes, TW-aligned ─────────────────────────────────────────
    @lru_cache(maxsize=None)
    def us(self, ticker: str, field: str = "close") -> pd.Series:
        return cta.load_us_index_tw(ticker, self.tw_index, field)

    # ── TAIEX spot (from us_index_pv ^TWII) ────────────────────────────
    @lru_cache(maxsize=None)
    def taiex_close(self) -> pd.Series:
        return self.us("^TWII", "close").rename("taiex")

    # ── Chicago Fed NFCI (weekly, TW-aligned, PIT-safe forward-fill) ───
    @lru_cache(maxsize=None)
    def nfci_tw(self, field: str = "nfci") -> pd.Series:
        """PIT-safe NFCI on TW calendar. `field` ∈ nfci/anfci/credit/risk/leverage/nonfin_lev."""
        return cta.load_nfci_tw(field, self.tw_index)

    @lru_cache(maxsize=None)
    def nfci_weekly(self, field: str = "nfci") -> pd.Series:
        """Raw weekly NFCI (Friday-dated). For event-mask construction."""
        return cta.load_nfci(field)

    # ── Macro (QNT-10 tidy layer), PIT-safe on the TW calendar ─────────
    # Reference-dated monthlies carry up to 150 days of look-ahead if joined
    # naively; `load_macro_tw` shifts by the documented publication lag first.
    # Never build a signal off `cta.load_macro` directly.
    @lru_cache(maxsize=None)
    def macro(self, series_id: str) -> pd.Series:
        """PIT-safe macro level on the TW calendar. See cta.macro_catalog()."""
        return cta.load_macro_tw(series_id, self.tw_index)

    @lru_cache(maxsize=None)
    def macro_yoy(self, series_id: str, periods: int = 12) -> pd.Series:
        """YoY % change, differenced on the RAW series then aligned PIT.
        `periods` is observations: 12 monthly, 4 quarterly — check the catalog."""
        return cta.load_macro_yoy_tw(series_id, self.tw_index, periods=periods)

    @lru_cache(maxsize=None)
    def tw_macro(self, field: str) -> pd.Series:
        """PIT-safe field from `tw_macro_monthly` on the TW calendar."""
        return cta.load_tw_macro_tw(field, self.tw_index)


def build_context() -> Context:
    asset = cta.load_asset("mtx", "1d")
    return Context(asset)
