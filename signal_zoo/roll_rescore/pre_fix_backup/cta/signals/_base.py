"""
Signal registry base class.

Each signal is a subclass of `Signal` in its own file under `mtx/cta/signals/`.
On import of the package, every subclass registers itself into `SIGNAL_REGISTRY`
via the `@register` decorator, so the runner picks it up with no config edits.

Convention
----------
    from cta.signals._base import Signal, register

    @register
    class LtTop10NetSignth(Signal):
        name        = "lt_top10npct_signth_w60"
        description = "LT top-10 net-position pct → sign-threshold (z-window 60)"
        sources     = ("large_trader", "mtx_1d")
        cadence     = "daily_15_45_tpe"
        live_date   = "2026-07-27"
        sign        = -1
        variants    = ("c2c",)          # add "ongap" to also trade the 05:00→08:45 gap

        def compute_raw(self, ctx) -> pd.Series:
            # ctx exposes lazy loaders + tw index. Return an unnormalized Series.
            tx_net10 = ctx.load_large_trader("TX", "top10_net_pct").astype(float)
            return _sign_thresh(tx_net10, 60)

The runner:
  1. Calls `compute_raw`
  2. Applies `normalize_signal(raw, method='tanh', window=252)`
  3. Multiplies by `self.sign` (frozen at declaration time — never re-discovered)
  4. Iterates `variants`; for each, calls the shared PnL function
  5. Upserts (date, signal_name, variant) → mtx_signal_values

`sign` MUST be committed by the human when the signal goes live. The runner
never auto-discovers a sign at compute time — that would introduce look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar

import pandas as pd


SIGNAL_REGISTRY: dict[str, "Signal"] = {}


class Signal:
    """Base class. Subclass and set the class attributes; define compute_raw."""
    # ── declared per subclass ─────────────────────────────────────────────
    name:           ClassVar[str]              = ""
    cn_name:        ClassVar[str]              = ""    # Chinese display name for UI (dropdown + card title)
    cn_short:       ClassVar[str]              = ""    # 2-4 char Chinese code for correlation matrix cells
    description:    ClassVar[str]              = ""
    cn_description: ClassVar[str]              = ""    # Chinese one-line summary for UI
    sources:        ClassVar[tuple[str, ...]]  = ()
    cadence:     ClassVar[str]              = "daily_15_45_tpe"
    live_date:   ClassVar[str]              = ""      # ISO date; before this = backtest
    sign:        ClassVar[int]              = +1      # +1 or -1, frozen at commit time
    variants:    ClassVar[tuple[str, ...]]  = ("c2c",)  # subset of VARIANT_REGISTRY keys
    # Optional override: variant_key → shift_days. Used when a signal's data
    # source publishes early enough to allow a tighter shift than the variant
    # default. Example: US-only signals can safely use shift(1) for o2o
    # because US data lands 06:00 TPE, before open[t-1] at 08:45 TPE.
    #
    # ── TRAP (QNT-19): that justification is LOADER-SPECIFIC. Do not copy ──
    # `shift_override = {"o2o": 1}` onto a signal built from `ctx.macro()`.
    #
    # The two US-data loaders align onto the TW calendar differently:
    #
    #   loader                 convention                    US obs of D lands on
    #   --------------------   --------------------------    --------------------
    #   cta.load_us_index_tw   pit_lag_days=1 (hard floor,   TW index D+1
    #                          raises below 1)
    #   cta.load_macro_tw      available_from =              TW index D
    #                          D + pub_lag_days; all 12
    #                          daily FRED series carry
    #                          pub_lag_days = 0
    #
    # So `load_macro_tw` is one full day more aggressive. The five `us_*`
    # signals below read `ctx.us_index(...)` and their o2o:1 override is
    # legitimate. A signal reading `ctx.macro(...)` / `ctx.macro_yoy(...)`
    # with the same override has roughly a full day of look-ahead.
    #
    # PIT-legal lags for a pub_lag_days=0 US-close daily series read through
    # `ctx.macro`:  c2c 2 · o2o 2 · day 1 · ongap is a RACE (window opens
    # 05:00 TPE, at or before the H.15 post) · night/noonpause not evaluated.
    #
    # Cost of the mistake, measured on QNT-14's 198-cell macro grid (median
    # paired ΔSR lag1−lag2, 25y, net): o2o +0.271 (win-rate 76.8%), c2c
    # +0.145 (66.7%). At the legal lag it manufactures gate passes from
    # nothing: 0 of 198 cells pass all four house gates at lag 2, versus 15
    # (o2o) and 1 (c2c) at lag 1.
    # Evidence: signal_zoo/macro_windows/lag_decomposition.csv
    # Guard: tools/check_macro_shift_overrides.py
    shift_override: ClassVar[dict[str, int]] = {}
    # Variants this signal is *included in* when building the aggregate. If
    # empty, the aggregate falls back to picking the auto-best-by-SR variant.
    # Set to include ONE or MANY variants — each becomes its own equally-
    # weighted sleeve in the aggregate.
    recommended_variants: ClassVar[tuple[str, ...]] = ()
    enabled:     ClassVar[bool]             = True
    # When True, the runner SKIPS the default `tanh(rolling_z(252))`
    # normalization and uses `compute_raw`'s output as-is (× sign). Use for
    # event-anchored binary/discrete signals whose values are already in
    # [-1, +1] and would be corrupted by rolling-z (which biases them
    # toward the historical mean and creates unintended baseline positions
    # during quiet periods).
    pre_normalized: ClassVar[bool] = False

    def compute_raw(self, ctx) -> pd.Series:  # pragma: no cover - subclass overrides
        raise NotImplementedError


def register(cls):
    """Decorator: register the class into SIGNAL_REGISTRY under its .name."""
    if not cls.name:
        raise ValueError(f"{cls.__name__}.name is empty")
    if cls.name in SIGNAL_REGISTRY:
        raise ValueError(f"duplicate signal name: {cls.name}")
    SIGNAL_REGISTRY[cls.name] = cls()
    return cls


# ── Variant definitions ────────────────────────────────────────────────────
# Each variant describes HOW the signal trades: which return window and
# which shift lag. `shift_days` is the PIT lag applied to the signal before
# multiplying by the return.
#
# TAIFEX night-session labelling (verified against 1-minute MXF bars to the
# tick): the night columns describe the session that ENDED on day t —
#     night_open[t]  = 15:00 of day t-1
#     night_close[t] = 05:00 of day t
# So a window anchored to day t needs shift(-1) on the night columns when it
# STARTS on day t (noonpause, night), and no shift when it ENDS on day t
# (ongap). Getting this backwards is what broke noonpause and ongap.
#
# The four intraday windows of day t, all traded on signal[t-1] (shift 1),
# which is published 14:00-15:00 TPE of t-1:
#     ongap     05:00 t → 08:45 t    open[t] / night_close[t]              14.0h margin
#     day       08:45 t → 13:45 t    close[t] / open[t]                    17.8h margin
#     noonpause 13:45 t → 15:00 t    night_open[t+1] / close[t]            22.8h margin
#     night     15:00 t → 05:00 t+1  night_close[t+1] / night_open[t+1]    24.0h margin
# They chain: day[t]·noonpause[t]·night[t]·ongap[t+1] = open[t+1]/open[t].
#
# The two daily windows use shift(2) because they open a day earlier:
#     c2c   close[t] / close[t-1]    entry 13:45 t-1, signal[t-2]
#     o2o   open[t]  / open[t-1]     entry 08:45 t-1, signal[t-2]

@dataclass(frozen=True)
class Variant:
    key:         str            # short identifier stored in mtx_signal_values.variant
    description: str
    shift_days:  int
    return_of:   Callable[[pd.DataFrame], pd.Series]
    cost_of:     Callable[[pd.DataFrame], pd.Series]


def _c2c_ret(asset: pd.DataFrame) -> pd.Series:
    return asset["close"].astype(float).pct_change()


def _c2c_cost(asset: pd.DataFrame) -> pd.Series:
    # 20 TWD round-trip fee per contract; MTX big-point value 50 TWD/point;
    # 0.002% slippage/impact tacked on. Matches notebook convention.
    c = asset["close"].astype(float)
    return 20.0 / (c * 50.0) + 0.00002


def _ongap_ret(asset: pd.DataFrame) -> pd.Series:
    """隔夜跳空 (05:00 t → 08:45 t).

    night_close[t] is 05:00 of day t (TAIFEX labels a night session by the
    day it ENDED), so this is same-index — no shift. The old
    `night_close.shift(1)` reached back to 05:00 of t-1 and therefore spanned
    ~28h including the whole day session of t-1, which the position (set from
    signal[t-1], published 14:00-15:00 of t-1) could not have known.
    """
    return asset["open"].astype(float) / asset["night_close"].astype(float) - 1


def _ongap_cost(asset: pd.DataFrame) -> pd.Series:
    nc = asset["night_close"].astype(float)
    return 20.0 / (nc * 50.0) + 0.00002


def _day_ret(asset: pd.DataFrame) -> pd.Series:
    """日盤 (day session): open→close of TAIFEX 08:45-13:45 session."""
    return asset["close"].astype(float) / asset["open"].astype(float) - 1


def _day_cost(asset: pd.DataFrame) -> pd.Series:
    o = asset["open"].astype(float)
    return 20.0 / (o * 50.0) + 0.00002


def _night_ret(asset: pd.DataFrame) -> pd.Series:
    """夜盤 (15:00 t → 05:00 t+1).

    The night session that STARTS on day t is labelled t+1 by TAIFEX, hence
    shift(-1). Anchoring it to day t (rather than using the same-index
    columns, which describe 15:00 t-1 → 05:00 t) is what gives the position
    a full day of margin: entry is 15:00 of t against signal[t-1], published
    14:00-15:00 of t-1. The same-index form entered at 15:00 of t-1 —
    simultaneous with the 15:00 large_trader / three_majors publication.
    """
    no = asset["night_open"].astype(float).shift(-1)
    nc = asset["night_close"].astype(float).shift(-1)
    return nc / no - 1


def _night_cost(asset: pd.DataFrame) -> pd.Series:
    o = asset["night_open"].astype(float).shift(-1)
    return 20.0 / (o * 50.0) + 0.00002


def _noonpause_ret(asset: pd.DataFrame) -> pd.Series:
    """日夜盤空檔 (13:45 t → 15:00 t) — the 75-minute lunch gap.

    night_open[t] is 15:00 of t-1, so the 15:00 print that CLOSES day t's
    gap is night_open[t+1]. The old same-index form divided 15:00 of t-1 by
    13:45 of t: a 22-hour window running BACKWARDS, i.e. the reciprocal of a
    real return. It was ~4.2x too volatile for a 75-minute gap and correlated
    -0.06 with the true gap, which is what produced a uniformly negative
    Sharpe across every signal.
    """
    return (asset["night_open"].astype(float).shift(-1)
            / asset["close"].astype(float) - 1)


def _noonpause_cost(asset: pd.DataFrame) -> pd.Series:
    c = asset["close"].astype(float)
    return 20.0 / (c * 50.0) + 0.00002


def _o2o_ret(asset: pd.DataFrame) -> pd.Series:
    """開盤對開盤 (open-to-open, 08:45 t-1 → 08:45 t)."""
    return asset["open"].astype(float).pct_change()


def _o2o_cost(asset: pd.DataFrame) -> pd.Series:
    o = asset["open"].astype(float)
    return 20.0 / (o * 50.0) + 0.00002


VARIANT_REGISTRY: dict[str, Variant] = {
    "c2c":       Variant("c2c",       "close[t]/close[t-1]-1 with shift(2)",             2, _c2c_ret,       _c2c_cost),
    "o2o":       Variant("o2o",       "開盤對開盤 open[t]/open[t-1]-1; shift 2 default (1 for US-only)", 2, _o2o_ret, _o2o_cost),
    "day":       Variant("day",       "日盤 close[t]/open[t]-1 with shift(1)",           1, _day_ret,       _day_cost),
    "noonpause": Variant("noonpause", "日夜盤空檔 night_open[t]/close[t]-1 with shift(1)", 1, _noonpause_ret, _noonpause_cost),
    "night":     Variant("night",     "夜盤 night_close[t]/night_open[t]-1 with shift(1)", 1, _night_ret,     _night_cost),
    "ongap":     Variant("ongap",     "隔夜跳空 open[t]/night_close[t-1]-1 with shift(1)",  1, _ongap_ret,     _ongap_cost),
}

# Chinese labels for UI
VARIANT_CN: dict[str, str] = {
    "c2c":       "日對日 (c2c)",
    "o2o":       "開對開 (o2o)",
    "day":       "日盤",
    "noonpause": "日夜盤空檔",
    "night":     "夜盤",
    "ongap":     "隔夜跳空",
}


def compute_variant_pnl(signed_signal: pd.Series, asset: pd.DataFrame,
                         variant: Variant, shift_days: int | None = None
                         ) -> tuple[pd.Series, pd.Series]:
    """Return (position_series, pnl_series) for one variant.

    ``shift_days`` overrides ``variant.shift_days`` when given (used by
    signals whose data source lets them use a tighter shift PIT-safely).
    """
    lag = shift_days if shift_days is not None else variant.shift_days
    position = signed_signal.reindex(asset.index).shift(lag)
    ret  = variant.return_of(asset)
    cost = variant.cost_of(asset)
    pnl  = position * ret - position.fillna(0).diff().abs() * cost
    return position, pnl
