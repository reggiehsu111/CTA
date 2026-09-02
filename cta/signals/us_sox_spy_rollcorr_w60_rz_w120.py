"""SOX×SPY 1d-return rolling-60 correlation → robust_z(120)."""
from ._base import Signal, register
from ._operators import robust_z


@register
class UsSoxSpyRollcorrW60RzW120(Signal):
    name        = "us_sox_spy_rollcorr_w60_rz_w120"
    cn_name        = "費半與美股相關性 Z"
    cn_short        = "費美相"

    description    = "SOX/SPY 1d-return rolling-60 corr → robust_z(120)"
    cn_description = "費半與SPY日報酬60日滾動相關係數，再取120日穩健Z分數：兩者連動性異常時作為進場訊號"
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
        sox_r1 = ctx.us("^SOX", "close").pct_change(1)
        spy_r1 = ctx.us("SPY",  "close").pct_change(1)
        rollcorr = sox_r1.rolling(60, min_periods=20).corr(spy_r1)
        return robust_z(rollcorr, 120)
