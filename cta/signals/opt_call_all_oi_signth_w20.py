"""Options call all-expiry OI → sign-threshold on 20-day z-score."""
from ._base import Signal, register
from ._operators import sign_thresh


@register
class OptCallAllOiSignthW20(Signal):
    name        = "opt_call_all_oi_signth_w20"
    cn_name        = "買權未平倉（門檻式）"
    cn_short        = "買權"

    description    = "TXO call OI (all expiries) → sign-threshold(z(20))"
    cn_description = "TXO全部合約買權未平倉量的20日Z分數超過±0.5門檻時進場：Call OI上升代表看多意願"
    sources     = ("options", "mtx_1d")
    live_date   = "2026-07-29"
    sign        = -1
    variants    = ("c2c", "o2o", "day", "noonpause", "night", "ongap")

    # DISABLED 2026-08-24, same failure mode as opt_put_mo_oi_selftanh_w60.
    # Re-derived on the corrected full-strike option data (the old 簡表
    # source captured 7.9% of true OI by 2026) with roll-adjusted returns
    # and an honest IS/OOS split at 2016:
    #
    #   IS-chosen sign is +1 -- the OPPOSITE of the frozen -1 above.
    #   IS SR is +0.06, i.e. noise.
    #   With the IS-honest sign -> OOS -0.52
    #   With the live sign (-1) -> OOS +0.43
    #
    # As with the put signal, the direction is not identifiable in-sample and
    # the live sign is contradicted by the only data you could have fitted on.
    # Left registered so values keep computing for monitoring, but out of the
    # equal-weight book. See signal_zoo/options_rederive_scoreboard.csv.
    enabled = False

    def compute_raw(self, ctx):
        return sign_thresh(ctx.opt_total("oi", "call", "all"), 20)
