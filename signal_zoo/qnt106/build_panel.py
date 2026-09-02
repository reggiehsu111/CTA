"""QNT-106 step 1 — build the 台股 MARKET-INTERNALS panel for MTX.

A new SOURCE-SERIES axis (QNT-78 rule 2: add sources, never transforms).
Nothing here has appeared in the macro grids (QNT-12/14/25) or the TXO OI
grids (QNT-99/104): those were exogenous-macro and options-book. This is the
CASH MARKET's own internals — breadth, cross-sectional dispersion, liquidity,
institutional flow, and leverage — aggregated to one daily number per series.

All aggregation happens IN SQL so NULL propagates (CLAUDE.md rule 1: pandas
sums an all-NaN group to 0.0).

Sources, all long-history and already local:
  tw_spot_pv + tw_spot_adj  2009-01 ..  breadth / dispersion / liquidity
  tw_institutional          2005-01 ..  外資 / 投信 / 自營 cash flow
  tw_margin_summary         2010-01 ..  融資 / 融券 market-wide balances
  tw_margin_short           2005-01 ..  借券賣出 (SBL) balances
"""
import sys, time
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
import pandas as pd, numpy as np
from db_utils import engine

OUT = "/home/ubuntu/mtx/signal_zoo/qnt106"
START = "2009-01-01"

# ── 1. cash-market internals from tw_spot_pv ──────────────────────────────
# TSE common shares only (4-digit ticker). Returns are ADJUSTED (adj_factor)
# because TW ex-dividend dates cluster in Jun-Aug and would otherwise put a
# seasonal hole in every breadth series.
SPOT_SQL = f"""
WITH base AS (
  SELECT p.date, p.ticker,
         p.close * COALESCE(a.adj_factor, 1.0) AS adjc,
         p.open, p.high, p.low, p.close, p.volume, p.amount, p.trades,
         p.bid, p.ask
  FROM tw_spot_pv p
  LEFT JOIN tw_spot_adj a ON a.date = p.date AND a.ticker = p.ticker::text
  WHERE p.board = 'TSE' AND p.ticker BETWEEN 1000 AND 9999
    AND p.date >= '{START}' AND p.close > 0 AND p.volume > 0
),
r AS (
  SELECT b.*,
    adjc / NULLIF(LAG(adjc) OVER w, 0) - 1                           AS ret,
    AVG(adjc) OVER (PARTITION BY ticker ORDER BY date ROWS 19 PRECEDING) AS ma20,
    AVG(adjc) OVER (PARTITION BY ticker ORDER BY date ROWS 59 PRECEDING) AS ma60,
    COUNT(*)  OVER (PARTITION BY ticker ORDER BY date ROWS 59 PRECEDING) AS nback
  FROM base b WINDOW w AS (PARTITION BY ticker ORDER BY date)
)
SELECT date,
  COUNT(*)                                                    AS n_stocks,
  SUM(CASE WHEN ret > 0 THEN 1 ELSE 0 END)::float
    / NULLIF(SUM(CASE WHEN ret <> 0 THEN 1 ELSE 0 END), 0)     AS adv_share,
  SUM(CASE WHEN ret > 0 THEN amount ELSE 0 END)
    / NULLIF(SUM(CASE WHEN ret <> 0 THEN amount ELSE 0 END), 0) AS upvol_share,
  STDDEV_SAMP(ret)                                            AS xs_disp,
  (percentile_cont(0.9) WITHIN GROUP (ORDER BY ret)
   + percentile_cont(0.1) WITHIN GROUP (ORDER BY ret)
   - 2 * percentile_cont(0.5) WITHIN GROUP (ORDER BY ret))
   / NULLIF(percentile_cont(0.9) WITHIN GROUP (ORDER BY ret)
            - percentile_cont(0.1) WITHIN GROUP (ORDER BY ret), 0) AS xs_skew,
  AVG(ret) - SUM(ret * amount) / NULLIF(SUM(amount), 0)        AS ew_minus_aw,
  AVG(CASE WHEN ret >= CASE WHEN date >= DATE '2015-06-01' THEN 0.095 ELSE 0.065 END
           THEN 1.0 ELSE 0.0 END)                              AS ext_up_share,
  AVG(CASE WHEN ret <= CASE WHEN date >= DATE '2015-06-01' THEN -0.095 ELSE -0.065 END
           THEN 1.0 ELSE 0.0 END)                              AS ext_dn_share,
  AVG(CASE WHEN nback >= 20 AND adjc > ma20 THEN 1.0
           WHEN nback >= 20 THEN 0.0 END)                      AS above_ma20,
  AVG(CASE WHEN nback >= 60 AND adjc > ma60 THEN 1.0
           WHEN nback >= 60 THEN 0.0 END)                      AS above_ma60,
  SUM(amount)                                                  AS tot_amount,
  SUM(POWER(amount::float8,2)) / NULLIF(POWER(SUM(amount::float8), 2), 0)       AS amt_hhi,
  SUM(amount) / NULLIF(SUM(trades), 0)                          AS avg_trade_ntd,
  AVG((high - low) / NULLIF(close, 0))                          AS mean_range,
  AVG(CASE WHEN bid > 0 AND ask > bid
           THEN (ask - bid) / ((ask + bid) / 2) END)            AS mean_spread,
  AVG((close - low) / NULLIF(high - low, 0))                    AS close_loc
FROM r
WHERE ret IS NOT NULL
GROUP BY date ORDER BY date
"""

INST_SQL = f"""
SELECT date,
  SUM("外資買賣超金額_千元")                                    AS fgn_net_k,
  SUM("投信買賣超金額_千元")                                    AS trust_net_k,
  SUM("自營買賣超金額_自行") + SUM("自營買賣超金額_避險")        AS dealer_net_k,
  SUM("外資買進金額_千元") + SUM("外資賣出金額_千元")            AS fgn_gross_k,
  SUM("投信買進金額_千元") + SUM("投信賣出金額_千元")            AS trust_gross_k,
  AVG(CASE WHEN "外資買賣超張數" > 0 THEN 1.0
           WHEN "外資買賣超張數" < 0 THEN 0.0 END)              AS fgn_breadth,
  AVG(CASE WHEN "投信買賣超張數" > 0 THEN 1.0
           WHEN "投信買賣超張數" < 0 THEN 0.0 END)              AS trust_breadth
FROM tw_institutional WHERE date >= '{START}' GROUP BY date ORDER BY date
"""

MARGIN_SQL = f"""
SELECT date,
  SUM(fin_today_ntd_k)  AS fin_bal_k,
  SUM(fin_today_lots)   AS fin_bal_lots,
  SUM(short_today_lots) AS short_bal_lots,
  SUM(fin_buy_lots)     AS fin_buy_lots,
  SUM(fin_sell_lots)    AS fin_sell_lots,
  SUM(fin_prev_lots)    AS fin_prev_lots,
  SUM(short_sell_lots)  AS short_sell_lots
FROM tw_margin_summary WHERE date >= '{START}' GROUP BY date ORDER BY date
"""

SBL_SQL = f"""
SELECT date, SUM(bal_short_k) AS sbl_bal_k, SUM(sell_short_k) AS sbl_sell_k
FROM tw_margin_short WHERE date >= '{START}' GROUP BY date ORDER BY date
"""

frames = {}
for name, sql in [("spot", SPOT_SQL), ("inst", INST_SQL),
                  ("margin", MARGIN_SQL), ("sbl", SBL_SQL)]:
    t0 = time.time()
    df = pd.read_sql(sql, engine)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index().astype(float)
    frames[name] = df
    print(f"{name:7s} {df.shape}  {df.index.min().date()} .. {df.index.max().date()}"
          f"  {time.time()-t0:.0f}s", flush=True)
    df.to_csv(f"{OUT}/raw_{name}.csv")

print("\nNULL rate per raw column:")
for n, df in frames.items():
    s = df.isna().mean().round(3)
    print(f"-- {n}\n{s.to_string()}")
