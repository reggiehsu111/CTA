"""LT top-10 net-position pct, sign-threshold on 60-day z-score."""
from ._base import Signal, register
from ._operators import sign_thresh


@register
class LtTop10npctSignthW60(Signal):
    name        = "lt_top10npct_signth_w60"
    cn_name        = "十大交易人淨部位（門檻式）"
    cn_short        = "十大A"

    description    = "LT top-10 net-position pct → sign-threshold(z(60), 0.5)"
    cn_description = "台指期十大交易人淨部位%的60日Z分數超過±0.5門檻時進場（順勢做多/空）"
    sources     = ("large_trader", "mtx_1d")
    live_date   = "2026-07-29"
    sign        = +1
    variants    = ("c2c", "o2o", "day", "noonpause", "night", "ongap")
    # Re-derived 2026-08-22 after the ongap look-ahead fix:
    # ongap SR was +1.34 on the look-ahead formula, +0.12 once corrected;
    # day is the best stable window (SR 0.80, halves 0.86/0.75, n=4257)
    recommended_variants = ("day",)

    def compute_raw(self, ctx):
        return sign_thresh(ctx.lt("top10_net_pct"), 60)
