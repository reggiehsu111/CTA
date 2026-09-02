"""QNT-99 Part B step 2 — build PUT-OI-ONLY features that survive the two things
that break naive put-OI series:

  1. The TOTAL is non-stationary. Median monthly put OI fell 570k (2011) -> 63k
     (2026) while the strike grid grew 147 -> 864 strikes. corr(log put OI, log
     spot) is +0.06 in 2013-18 and -0.69 in 2019-26 — the SIGN OF THE RELATION
     FLIPS. That is why opt_put_mo_oi_selftanh_w60 could not identify a sign and
     was disabled 2026-08-24.
  2. The strike grid keeps changing, so any feature that counts strikes (HHI,
     strike counts) reads TAIFEX's product decisions, not positioning.

Every feature below is therefore either OI-WEIGHTED over strikes (adding empty
strikes changes nothing) or a RATIO/GROWTH (scale-free). Moneyness is measured
against the TAIEX close of the SAME day — the options print is end-of-day, so the
feature is known after the 13:45 close, which is what the _SHIFT convention on
top of it assumes.
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
from db_utils import engine

OUT = "/home/ubuntu/mtx/signal_zoo/qnt99"

def build(expiry_filter="monthly"):
    where = "side='put' and market_code=0"
    if expiry_filter == "monthly":
        where += " and expiry ~ '^[0-9]{6}$'"
    q = (f"select date, expiry, strike, oi, volume from tw_options_daily "
         f"where {where} and oi is not null and date >= '2009-01-01'")
    df = pd.read_sql(q, engine, parse_dates=["date"])
    tx = pd.read_sql("select date, close from tw_taiex", engine,
                     parse_dates=["date"]).set_index("date")["close"].astype(float)
    df["spot"] = df["date"].map(tx)
    df = df.dropna(subset=["spot"])
    df["oi"] = df["oi"].astype(float); df["volume"] = df["volume"].astype(float)
    df["m"] = df["strike"] / df["spot"] - 1.0          # +ve strike = ITM put
    df["w"] = df["oi"]
    df["front"] = (df["expiry"] == df["date"].dt.strftime("%Y%m")).astype(float)

    g = df.groupby("date")
    tot = g["oi"].sum().replace(0, np.nan)
    wsum = lambda col: g.apply(lambda x: np.nansum(x["w"] * x[col]))
    cog  = wsum("m") / tot                                        # OI-weighted moneyness
    m2   = wsum("m2") if False else g.apply(lambda x: np.nansum(x["w"] * x["m"]**2)) / tot
    disp = np.sqrt((m2 - cog**2).clip(lower=0))
    otm  = g.apply(lambda x: np.nansum(x["w"] * (x["m"] < 0))) / tot     # strike below spot
    far  = g.apply(lambda x: np.nansum(x["w"] * (x["m"] < -0.05))) / tot # >5% OTM = crash hedge
    wall = g.apply(lambda x: x.loc[x["oi"].idxmax(), "m"] if x["oi"].max() > 0 else np.nan)
    fsh  = g.apply(lambda x: np.nansum(x["w"] * x["front"])) / tot
    vol  = g["volume"].sum()
    churn = (vol / tot)

    out = pd.DataFrame({
        "put_oi_total":  tot,
        "put_cog":       cog,      # OI-weighted (K/S - 1)
        "put_disp":      disp,     # OI-weighted sd of moneyness
        "put_otm_share": otm,      # share of put OI struck BELOW spot
        "put_far_share": far,      # share struck >5% below spot
        "put_wall":      wall,     # moneyness of the single biggest-OI strike
        "put_front_share": fsh,
        "put_churn":     churn,    # put volume / put OI
    }).sort_index()
    out["put_oi_growth"] = np.log(out["put_oi_total"]).diff()      # scale-free flow
    out["put_cog_chg"]   = out["put_cog"].diff()
    out["put_far_chg"]   = out["put_far_share"].diff()
    return out

if __name__ == "__main__":
    for ef in ("monthly", "all"):
        p = build(ef)
        p.to_csv(f"{OUT}/put_oi_features_{ef}.csv")
        print(f"=== {ef}: {p.shape} {p.index.min().date()} -> {p.index.max().date()}")
        print(p.groupby(p.index.year).mean().round(4).to_string())
