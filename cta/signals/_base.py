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
    # Optional override: variant_key → shift_days.
    #
    # ── QNT-60: in practice this must stay EMPTY. Read before using it. ────
    # The old rationale was "this signal's source publishes early enough to
    # allow a tighter shift than the variant default" — e.g. US data lands
    # 06:00 TPE, before o2o's 08:45 t-1 entry, so `{"o2o": 1}` looked safe.
    # Five us_*/tv_* signals carried exactly that. It was LOOK-AHEAD, because
    # data-arrival time is not the test. The test is whether the VALUE existed
    # at fill time, and the value is whatever the runner wrote.
    #
    # Measured over all of `mtx_signal_values` (QNT-60): the only clock time
    # that ever stamps a row with its own date is 15:31 TPE (n=1,620 lag-0
    # rows; no other computed_at bucket reaches lag 0). It cannot be earlier —
    # a signal's value for D is indexed on `ctx.tw_index`, so it needs the MTX
    # bar for D, which lands 14:00 TPE. So `signed[t-k]` first exists at 15:31
    # of t-k, and the PIT-legal minimum shift is exactly the variant default:
    #
    #   c2c 2 · o2o 2 · day 1 · ongap 1 · night 1 · noonpause 1
    #
    # `{"o2o": 1}` put the fill 6h45m before the value existed. Removed on
    # QNT-60; it bought nothing (mean paired ΔSR -0.072, worse on 4 of 5).
    # Deploy of that removal is QNT-72.
    #
    # An override is therefore only ever justified by a NEW, EARLIER RUNNER
    # INVOCATION (e.g. a US-only 06:30 TPE pass), never by an upstream
    # publication time. Re-derive against that runner's write time; do not
    # relax the table by hand.
    #
    # Separately (QNT-19, loader-side): never set `enforce_floor=False` or pass
    # `pub_lag_days=` in a registered signal. Before the +1 floor,
    # `cta.load_macro_tw` landed a US observation stamped D on TW index D while
    # `load_us_index_tw` landed it on D+1 — worth a median paired ΔSR of +0.271
    # on o2o (76.8% win-rate) and +0.145 on c2c across QNT-14's 198-cell grid,
    # and 15 fake four-gate passers on o2o.
    # Evidence: signal_zoo/macro_windows/lag_decomposition.csv
    #           signal_zoo/qnt19_postfloor/QNT19_FOOTNOTE.md
    # Guard: tools/check_macro_shift_overrides.py  (enforces both rules)
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


def _is_rollover(asset: pd.DataFrame) -> pd.Series:
    """True on days whose front-month contract differs from yesterday's.

    Self-contained (does not require the Asset subclass) so the variant legs
    below work on a plain DataFrame too. Falls back to all-False when the
    loader did not supply `front_expiry`.
    """
    if "front_expiry" not in asset.columns:
        return pd.Series(False, index=asset.index)
    fe = asset["front_expiry"]
    return (fe.ne(fe.shift(1)) & fe.shift(1).notna()).fillna(False).astype(bool)


def _prev_close_cc(asset: pd.DataFrame) -> pd.Series:
    """Yesterday's close ON THE CONTRACT HELD TODAY.

    On rollover day t today's front IS yesterday's back, so `close.shift(1)`
    (the OLD contract) books the calendar spread as P&L: 305 roll days, mean
    57.0 bps, max 351.8 bps. Same definition as Asset.continuous_prev_close.
    """
    prev = asset["close"].astype(float).shift(1)
    if "back_close" in asset.columns:
        prev = prev.where(~_is_rollover(asset),
                          asset["back_close"].astype(float).shift(1))
    return prev


def _prev_open_cc(asset: pd.DataFrame) -> pd.Series:
    """Yesterday's open on the contract held today (o2o counterpart).

    Exact when the loader supplies `back_open`; otherwise falls back to
    yesterday's open scaled by the calendar spread measured at yesterday's
    close. The fallback is a poor substitute - measured against the exact
    back_open it removes only 32% of the roll error (mean |exact - approx|
    31.7 bps vs mean |raw - exact| 47.0 bps) - so prefer back_open.
    """
    o = asset["open"].astype(float)
    prev = o.shift(1)
    roll = _is_rollover(asset)
    if "back_open" in asset.columns:
        return prev.where(~roll, asset["back_open"].astype(float).shift(1))
    if "back_close" in asset.columns:
        ratio = asset["back_close"].astype(float) / asset["close"].astype(float)
        return prev.where(~roll, (o * ratio).shift(1))
    return prev


def _next_front_close(asset: pd.DataFrame) -> pd.Series:
    """Today's 13:45 close on TOMORROW's front contract (noonpause leg).

    `night_open[t+1]` is the 15:00 print of day t but on front(t+1) - the
    loader merges the night session on (date, front_expiry) - so on the eve of
    a roll the denominator must be today's BACK close, not today's close.
    Verified: front_expiry[t+1] == back_expiry[t] on 305/305 roll days.
    """
    c = asset["close"].astype(float)
    if "back_close" not in asset.columns:
        return c
    roll_next = _is_rollover(asset).shift(-1).fillna(False).astype(bool)
    return c.where(~roll_next, asset["back_close"].astype(float))


def _c2c_ret(asset: pd.DataFrame) -> pd.Series:
    """close[t] / close[t-1] - 1, roll-adjusted (QNT-21).

    Was `close.pct_change()`, which booked the calendar spread on all 305 roll
    days. B&H c2c 0.493 raw -> 0.700 roll-adjusted.
    """
    return asset["close"].astype(float) / _prev_close_cc(asset) - 1


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

    Roll-adjusted (QNT-21): night_open[t+1] is already on the NEW front on the
    eve of a roll, so the denominator is _next_front_close, not close[t]. The
    unadjusted form booked 47.7 bps mean (370.7 max) on 112 roll-eve days -
    21% of this 75-minute window's total variance, the worst of the six legs.
    B&H noonpause -0.253 raw -> +0.611 roll-adjusted.
    """
    return (asset["night_open"].astype(float).shift(-1)
            / _next_front_close(asset) - 1)


def _noonpause_cost(asset: pd.DataFrame) -> pd.Series:
    c = asset["close"].astype(float)
    return 20.0 / (c * 50.0) + 0.00002


def _o2o_ret(asset: pd.DataFrame) -> pd.Series:
    """開盤對開盤 (open-to-open, 08:45 t-1 → 08:45 t), roll-adjusted (QNT-21).

    Was `open.pct_change()`: 305 roll days, mean 47.0 bps, max 289.7 bps.
    B&H o2o 0.483 raw -> 0.666 roll-adjusted (exact back_open).
    """
    return asset["open"].astype(float) / _prev_open_cc(asset) - 1


def _o2o_cost(asset: pd.DataFrame) -> pd.Series:
    o = asset["open"].astype(float)
    return 20.0 / (o * 50.0) + 0.00002


VARIANT_REGISTRY: dict[str, Variant] = {
    "c2c":       Variant("c2c",       "close[t]/close[t-1]-1 with shift(2)",             2, _c2c_ret,       _c2c_cost),
    "o2o":       Variant("o2o",       "開盤對開盤 open[t]/open[t-1]-1; shift 2 default (1 for US-only)", 2, _o2o_ret, _o2o_cost),
    "day":       Variant("day",       "日盤 close[t]/open[t]-1 with shift(1)",           1, _day_ret,       _day_cost),
    "noonpause": Variant("noonpause", "日夜盤空檔 night_open[t]/close[t]-1 with shift(1)", 1, _noonpause_ret, _noonpause_cost),
    "night":     Variant("night",     "夜盤 night_close[t]/night_open[t]-1 with shift(1)", 1, _night_ret,     _night_cost),
    "ongap":     Variant("ongap",     "隔夜跳空 open[t]/night_close[t]-1 with shift(1)",  1, _ongap_ret,     _ongap_cost),
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
