"""US log(SOX/SPY) 120-day change (momentum on relative log-ratio)."""
from ._base import Signal, register
import numpy as np


@register
class UsLogRatioPctChgW120(Signal):
    name        = "us_log_ratio_pct_chg_w120"
    cn_name        = "費半與美股比值變動"
    cn_short        = "費美比"

    description    = "US log(SOX/SPY) − log(SOX/SPY).shift(120)  (120d momentum)"
    cn_description = "美股 log(費半/SPY) 相對強弱的120日動能：半導體相對大盤走強時看多台指"
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
        spy = ctx.us("SPY", "close")
        lr = np.log(sox / spy)
        return lr - lr.shift(120)
