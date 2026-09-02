"""SOX 20d return relative to SPY 20d return, "agreement amplifier", robust-z(20)."""
from ._base import Signal, register
from ._operators import robust_z
import numpy as np


@register
class UsSoxAgreeAmpN20RobustW20(Signal):
    name        = "us_sox_agree_amp_n20_robustw20"
    cn_name        = "費半連續同向強度"
    cn_short        = "費續強"

    description    = "sign(SPY_20d) * SOX_20d / SPY_20d (clipped ±10) → robust_z(20)"
    cn_description = "費半20日報酬相對SPY 20日報酬的放大倍數（同向時保留、反向時反轉），再取20日穩健Z分數"
    sources     = ("us_indexes", "mtx_1d")
    live_date   = "2026-07-29"
    sign        = +1
    variants    = ("c2c", "o2o", "day", "noonpause", "night", "ongap")
    # QNT-60: shift_override REMOVED (was {"o2o": 1}).
    #
    # The information IS available in time — the US close of t-2 lands ~05:00
    # TPE of t-1, before o2o's 08:45 t-1 entry. But information-availability is
    # not the test; the test is whether the VALUE EXISTED at fill time. It did
    # not. `mtx_signal_values` is written by the runner, and measured over the
    # whole table the ONLY clock time that ever stamps a row with its own date
    # is 15:31 TPE (the daily mtx_1d cron; n=1,620 rows at lag 0, and no other
    # bucket reaches lag 0). Nothing earlier can: a signal's value for date D
    # is indexed on `ctx.tw_index`, so it cannot be computed until the MTX bar
    # for D exists at 14:00 TPE. The 06:00 us_indexes chain-trigger therefore
    # only ever rewrites rows dated D-1 and older with unchanged inputs.
    #
    # So signed[t-1] first exists at 15:31 TPE of t-1 — 6h45m AFTER o2o's
    # 08:45 t-1 entry. shift(1) on o2o read a number that had not been
    # computed yet. The variant default shift(2) is the PIT-legal minimum
    # (entry 08:45 t-1 against signed[t-2], computed 15:31 t-2: 17.2h margin).
    #
    # It bought nothing: SR on o2o with shift(1) vs shift(2) is
    # 0.302/0.409, 0.311/0.324, 0.316/0.379, -0.006/0.185, 0.177/0.162
    # across the five signals that carried it — mean dSR -0.072, i.e. the
    # look-ahead version is WORSE. No live sleeve uses o2o, so this only ever
    # affected the displayed variant-comparison table (which is what a future
    # `recommended_variants` choice would be read off).

    def compute_raw(self, ctx):
        sox = ctx.us("^SOX", "close")
        spy = ctx.us("SPY",  "close")
        r_sox_20 = sox.pct_change(20)
        r_spy_20 = spy.pct_change(20)
        denom = r_spy_20.where(r_spy_20.abs() > 0.003, np.nan)
        amp = (np.sign(r_spy_20) * (r_sox_20 / denom)).clip(-10, 10)
        return robust_z(amp, 20)
