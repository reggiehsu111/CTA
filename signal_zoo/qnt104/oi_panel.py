"""QNT-104 step 1 — CALL-OI and CALL/PUT-COMBINED features for the TXO OI sweep.

QNT-99 swept PUT OI alone and came back null (0/1584 four-gate passers). This
ticket adds the two axes it did not touch: the CALL side, and features that
only exist when both sides are read together (PCR, tail-hedge asymmetry, the
call/put wall gap, max pain).

Every design constraint QNT-99 established is carried over unchanged:

  * The TOTALS are non-stationary in BOTH sides. Monthly call OI fell
    148M (2011) -> 7.5M (2026 ytd) while the strike grid grew 55 -> 539.
    So every feature here is either OI-WEIGHTED over strikes (adding empty
    strikes changes nothing) or a RATIO / GROWTH (scale-free).
  * Moneyness m = K/S - 1 against the TAIEX close of the SAME day. For calls
    m > 0 is OTM (mirror of puts, where m < 0 is OTM). `far` = 5% OTM on the
    side's own OTM direction, i.e. the crash-hedge / upside-chase tail.
  * Anything measured against today's spot is a candidate PRICE MIRROR; step 2
    tests that explicitly before anything is scored.

Panels: `monthly` (expiry ~ '^[0-9]{6}$') and `all` (weeklies included, which
only exist from 2012 — stated, not hidden). Regular session only (market_code=0;
the after-hours rows carry no OI at all).
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
from db_utils import engine

OUT   = "/home/ubuntu/mtx/signal_zoo/qnt104"
START = "2009-01-01"          # same window as QNT-99, so the put grid is comparable


def _side_features(df, side):
    """OI-weighted / scale-free features for one side. df has date, m, oi, volume, front."""
    pre  = side[0]                       # 'c' / 'p'
    otm_dir = 1.0 if side == "call" else -1.0     # sign of m that is OUT of the money
    g = df.groupby("date")
    tot = g["oi"].sum().replace(0, np.nan)
    w   = df["oi"].values
    def wagg(v):
        s = pd.Series(w * v, index=df["date"].values).groupby(level=0).sum()
        return s / tot
    m = df["m"].values
    cog  = wagg(m)
    m2   = wagg(m ** 2)
    disp = np.sqrt((m2 - cog ** 2).clip(lower=0))
    otm  = wagg((otm_dir * m > 0).astype(float))          # share of OI out of the money
    far  = wagg((otm_dir * m > 0.05).astype(float))       # >5% OTM = the tail
    fsh  = wagg(df["front"].values)
    idx  = g["oi"].idxmax()
    wall = df.loc[idx, "m"].values
    wall = pd.Series(wall, index=idx.index)
    churn = g["volume"].sum() / tot
    out = pd.DataFrame({
        f"{side}_oi_total": tot, f"{side}_cog": cog, f"{side}_disp": disp,
        f"{side}_otm_share": otm, f"{side}_far_share": far, f"{side}_wall": wall,
        f"{side}_front_share": fsh, f"{side}_churn": churn,
    }).sort_index()
    out[f"{side}_oi_growth"] = np.log(out[f"{side}_oi_total"]).diff()
    out[f"{side}_cog_chg"]   = out[f"{side}_cog"].diff()
    out[f"{side}_far_chg"]   = out[f"{side}_far_share"].diff()
    # extras needed by the combined block, dropped before the call-only panel ships
    out[f"_{pre}_far_oi"] = wagg((otm_dir * m > 0.05).astype(float)) * tot
    out[f"_{pre}_atm_oi"] = wagg((np.abs(m) < 0.025).astype(float)) * tot
    out[f"_{pre}_vol"]    = g["volume"].sum()
    return out


def _max_pain(df_c, df_p):
    """Moneyness of the max-pain strike (writer payout minimiser), front monthly only.

    Restricted to strikes within +-15% of spot: outside that band OI is ~0 and the
    payout curve is monotone, so the argmin cannot live there. O(K^2) per date on
    ~60 strikes.
    """
    out = {}
    both = pd.concat([df_c.assign(s=1.0), df_p.assign(s=-1.0)])
    both = both[(both["front"] == 1) & (both["m"].abs() < 0.15)]
    for d, x in both.groupby("date"):
        K = np.sort(x["strike"].unique())
        if len(K) < 5: continue
        kc, oc = x.loc[x.s > 0, "strike"].values, x.loc[x.s > 0, "oi"].values
        kp, op = x.loc[x.s < 0, "strike"].values, x.loc[x.s < 0, "oi"].values
        pain = (np.maximum(K[:, None] - kc[None, :], 0) * oc).sum(1) + \
               (np.maximum(kp[None, :] - K[:, None], 0) * op).sum(1)
        out[d] = K[int(np.argmin(pain))] / x["spot"].iloc[0] - 1.0
    return pd.Series(out).sort_index()


def build(expiry_filter="monthly"):
    where = "market_code=0 and oi is not null and date >= %(s)s"
    if expiry_filter == "monthly":
        where += " and expiry ~ '^[0-9]{6}$'"
    df = pd.read_sql(f"select date, side, expiry, strike, oi, volume from tw_options_daily "
                     f"where {where}", engine, params={"s": START}, parse_dates=["date"])
    tx = pd.read_sql("select date, close from tw_taiex", engine,
                     parse_dates=["date"]).set_index("date")["close"].astype(float)
    df["spot"] = df["date"].map(tx)
    df = df.dropna(subset=["spot"]).copy()
    for c in ("oi", "volume", "strike"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].fillna(0.0)
    df["m"]     = df["strike"] / df["spot"] - 1.0
    df["front"] = (df["expiry"] == df["date"].dt.strftime("%Y%m")).astype(float)
    df = df.sort_values("date")

    dc = df[df.side == "call"]
    dp = df[df.side == "put"]
    C = _side_features(dc, "call")
    P = _side_features(dp, "put")
    J = C.join(P, how="outer")

    # ── COMBINED features: only exist when both sides are read together ──
    ct, pt = J["call_oi_total"], J["put_oi_total"]
    comb = pd.DataFrame(index=J.index)
    comb["pcr_oi"]        = pt / ct
    comb["pcr_vol"]       = J["_p_vol"] / J["_c_vol"].replace(0, np.nan)
    comb["pcr_oi_chg"]    = np.log(comb["pcr_oi"]).diff()
    comb["pcr_far"]       = J["_p_far_oi"] / J["_c_far_oi"].replace(0, np.nan)   # tail-hedge PCR
    comb["pcr_atm"]       = J["_p_atm_oi"] / J["_c_atm_oi"].replace(0, np.nan)
    comb["cog_gap"]       = J["call_cog"] - J["put_cog"]        # width of the positioned range
    comb["cog_mid"]       = (J["call_cog"] + J["put_cog"]) / 2  # its centre vs spot: skew
    comb["far_asym"]      = J["put_far_share"] - J["call_far_share"]
    comb["wall_gap"]      = J["call_wall"] - J["put_wall"]
    comb["wall_mid"]      = (J["call_wall"] + J["put_wall"]) / 2
    comb["churn_ratio"]   = J["put_churn"] / J["call_churn"].replace(0, np.nan)
    comb["oi_growth_diff"]= J["call_oi_growth"] - J["put_oi_growth"]
    comb["disp_ratio"]    = J["put_disp"] / J["call_disp"].replace(0, np.nan)
    comb["front_diff"]    = J["call_front_share"] - J["put_front_share"]
    comb["max_pain"]      = _max_pain(dc, dp).reindex(J.index)

    calls = C[[c for c in C.columns if not c.startswith("_")]]
    puts  = P[[c for c in P.columns if not c.startswith("_")]]
    return calls, comb, puts


if __name__ == "__main__":
    for ef in ("monthly", "all"):
        calls, comb, puts = build(ef)
        calls.to_csv(f"{OUT}/call_features_{ef}.csv")
        comb.to_csv(f"{OUT}/comb_features_{ef}.csv")
        puts.to_csv(f"{OUT}/put_features_{ef}.csv")
        print(f"=== {ef}: call {calls.shape} comb {comb.shape} "
              f"{calls.index.min().date()} -> {calls.index.max().date()}")
        print(pd.concat([calls, comb], axis=1).groupby(calls.index.year).mean().round(4).to_string())
