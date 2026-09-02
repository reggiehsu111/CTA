"""QNT-106 step 2 — 29 market-internals SOURCE SERIES from the raw aggregates."""
import pandas as pd, numpy as np
OUT = "/home/ubuntu/mtx/signal_zoo/qnt106"

def load():
    r = {n: pd.read_csv(f"{OUT}/raw_{n}.csv", index_col=0, parse_dates=True)
         for n in ("spot", "inst", "margin", "sbl")}
    s, i, m, b = r["spot"], r["inst"], r["margin"], r["sbl"]
    amt_k = s["tot_amount"] / 1e3                      # NTD -> thousands, to match *_k
    f = pd.DataFrame(index=s.index)

    # breadth (6)
    f["adv_share"]    = s["adv_share"]
    f["upvol_share"]  = s["upvol_share"]
    f["above_ma20"]   = s["above_ma20"]
    f["above_ma60"]   = s["above_ma60"]
    f["ext_net"]      = s["ext_up_share"] - s["ext_dn_share"]
    f["n_traded"]     = s["n_stocks"]
    # cross-sectional shape (3)
    f["xs_disp"]      = s["xs_disp"]
    f["xs_skew"]      = s["xs_skew"]
    f["ew_minus_aw"]  = s["ew_minus_aw"]
    # liquidity / microstructure (6)
    f["log_amount"]   = np.log(s["tot_amount"])
    f["amt_hhi"]      = s["amt_hhi"]
    f["log_avgtrade"] = np.log(s["avg_trade_ntd"])
    f["mean_range"]   = s["mean_range"]
    f["mean_spread"]  = s["mean_spread"]
    f["close_loc"]    = s["close_loc"]
    # institutional cash flow (7)
    f["fgn_net"]      = (i["fgn_net_k"]    / amt_k).reindex(s.index)
    f["trust_net"]    = (i["trust_net_k"]  / amt_k).reindex(s.index)
    f["dealer_net"]   = (i["dealer_net_k"] / amt_k).reindex(s.index)
    f["fgn_gross"]    = (i["fgn_gross_k"]  / amt_k).reindex(s.index)
    f["trust_gross"]  = (i["trust_gross_k"]/ amt_k).reindex(s.index)
    f["fgn_breadth"]  = i["fgn_breadth"].reindex(s.index)
    f["trust_breadth"]= i["trust_breadth"].reindex(s.index)
    # leverage: 融資 / 融券 / 借券 (7)
    f["fin_intens"]   = (m["fin_bal_k"] / amt_k).reindex(s.index)
    f["fin_dlog"]     = np.log(m["fin_bal_k"]).diff().reindex(s.index)
    f["short_fin"]    = (m["short_bal_lots"] / m["fin_bal_lots"]).reindex(s.index)
    f["short_dlog"]   = np.log(m["short_bal_lots"]).diff().reindex(s.index)
    f["fin_turn"]     = ((m["fin_buy_lots"] + m["fin_sell_lots"])
                         / m["fin_prev_lots"]).reindex(s.index)
    f["sbl_vs_fin"]   = (b["sbl_bal_k"] / m["fin_bal_k"]).reindex(s.index)
    f["sbl_sell"]     = (b["sbl_sell_k"] / amt_k).reindex(s.index)
    return f.replace([np.inf, -np.inf], np.nan)

FAMILY = {**{k: "breadth" for k in
             ["adv_share","upvol_share","above_ma20","above_ma60","ext_net","n_traded"]},
          **{k: "xsec" for k in ["xs_disp","xs_skew","ew_minus_aw"]},
          **{k: "liquidity" for k in
             ["log_amount","amt_hhi","log_avgtrade","mean_range","mean_spread","close_loc"]},
          **{k: "flow" for k in
             ["fgn_net","trust_net","dealer_net","fgn_gross","trust_gross",
              "fgn_breadth","trust_breadth"]},
          **{k: "leverage" for k in
             ["fin_intens","fin_dlog","short_fin","short_dlog","fin_turn",
              "sbl_vs_fin","sbl_sell"]}}

if __name__ == "__main__":
    f = load(); f.to_csv(f"{OUT}/features.csv")
    print(f"{f.shape[1]} source series x {len(f)} days  "
          f"{f.index.min().date()} .. {f.index.max().date()}")
    print(pd.DataFrame({"n": f.notna().sum(), "first": f.apply(lambda c: c.first_valid_index()),
                        "family": pd.Series(FAMILY)}).to_string())
