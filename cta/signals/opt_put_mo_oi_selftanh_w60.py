"""Options put monthly-expiry OI → tanh(z(60))."""
from ._base import Signal, register
from ._operators import bd_selftanh


@register
class OptPutMoOiSelftanhW60(Signal):
    name        = "opt_put_mo_oi_selftanh_w60"
    cn_name        = "近月賣權未平倉（連續式）"
    cn_short        = "賣權"

    description    = "TXO put OI (monthly expiry only) → tanh(z(60))"
    cn_description = "TXO月合約賣權未平倉量的60日Z分數（tanh平滑）：Put OI異常高時反映避險需求，用作反向指標"
    sources     = ("options", "mtx_1d")
    live_date   = "2026-07-29"
    sign        = -1
    variants    = ("c2c", "o2o", "day", "noonpause", "night", "ongap")
    # Re-derived 2026-08-22 after the ongap look-ahead fix:
    # ongap SR was +1.97 on the look-ahead formula, -0.04 once corrected.
    # o2o is the ONLY variant positive in both halves (0.35, 0.12/0.73);
    # noonpause scores 0.81 but its 2nd half is -0.12, so it is rejected
    recommended_variants = ("o2o",)

    # DISABLED 2026-08-24. Re-derived on the corrected full-strike option
    # data (the old 簡表 source captured 7.9% of true monthly OI by 2026)
    # with roll-adjusted returns and an honest IS/OOS split at 2016:
    #
    #   IS-chosen sign is +1 -- the OPPOSITE of the frozen -1 below.
    #   IS SR is +0.16 either way, i.e. indistinguishable from noise.
    #   With the IS-honest sign  -> OOS -0.51
    #   With the live sign (-1)  -> OOS +0.48
    #
    # The sign flips between halves, so it is not identifiable from data;
    # the -1 was fitted on full contaminated history and the +0.48 is only
    # reachable with a direction nothing in-sample supported. Left
    # registered (values keep computing for monitoring) but out of the book.
    enabled = False

    def compute_raw(self, ctx):
        return bd_selftanh(ctx.opt_total("oi", "put", "monthly"), 60)
