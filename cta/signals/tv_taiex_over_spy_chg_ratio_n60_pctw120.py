"""TAIEX/SPY 60d change-ratio − same value 120d ago."""
from ._base import Signal, register
from ._operators import pair_chg_ratio


@register
class TvTaiexOverSpyChgRatioN60PctW120(Signal):
    name        = "tv_taiex_over_spy_chg_ratio_n60_pctw120"
    cn_name        = "台指相對美股變動比"
    cn_short        = "台美比"

    description    = "TAIEX/SPY 60d change-ratio (pair_chg_ratio) − same 120d ago"
    cn_description = "台股加權/SPY 60日漲跌比率（除數含最低幅度保護）的120日變化：台股相對美股走強幅度加速時進場"
    sources     = ("us_indexes", "mtx_1d")
    live_date   = "2026-07-29"
    sign        = -1
    variants    = ("c2c", "o2o", "day", "noonpause", "night", "ongap")
    # Re-derived 2026-08-22 after the ongap look-ahead fix:
    # ongap SR was +0.68 on the look-ahead formula, +0.22 (2nd-half-negative)
    # once corrected; c2c is stable at 0.41 over 6048 bars
    recommended_variants = ("c2c",)
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
        taiex = ctx.taiex_close()
        spy   = ctx.us("SPY", "close")
        r = pair_chg_ratio(taiex, spy, 60)
        return r - r.shift(120)
