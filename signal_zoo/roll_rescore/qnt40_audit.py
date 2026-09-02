#!/usr/bin/env python3
"""QNT-40 read-only audit: is the approved QNT-21 backfill still intact in RDS?

For each of the 6,277 (date, signal_name, variant) keys the backfill touched,
classify the CURRENT stored pnl_1d as
    NEW      -> roll-adjusted (what the backfill wrote)
    OLD      -> raw-return    (what the stale deployed Lambda writes)
    NEITHER  -> something else rewrote it
Also reports which of those keys sit inside the Lambda's LOOKBACK_DAYS=30
recompute window, i.e. which ones a daily pass can still revert.
No writes.
"""
import sys, datetime as dt
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
import pandas as pd
from db_utils import engine

HERE = "/home/ubuntu/mtx/signal_zoo/roll_rescore"
exp = pd.read_csv(f"{HERE}/backfill_rows.csv", parse_dates=["date"])

cur = pd.read_sql(
    "SELECT date, signal_name, variant, pnl_1d, computed_at "
    "FROM mtx_signal_values WHERE variant IN ('c2c','o2o','noonpause')",
    engine, parse_dates=["date"])

m = exp.merge(cur, on=["date", "signal_name", "variant"], how="left", validate="1:1")
print("keys expected :", len(exp), " missing in DB:", int(m["pnl_1d"].isna().sum()))

tol = 1e-12
d_new = (m["pnl_1d"] - m["new_pnl_1d"]).abs()
d_old = (m["pnl_1d"] - m["old_pnl_1d"]).abs()
is_new = d_new <= tol
is_old = (~is_new) & (d_old <= tol)
other  = ~(is_new | is_old)
print(f"NEW (roll-adjusted): {int(is_new.sum())}")
print(f"OLD (raw-return)   : {int(is_old.sum())}")
print(f"NEITHER            : {int(other.sum())}")
if other.any():
    print(m.loc[other, ["date","signal_name","variant","old_pnl_1d","new_pnl_1d","pnl_1d"]].head(20).to_string())

# how many affected keys are still inside a 30-calendar-day recompute window
today = pd.Timestamp(dt.date.today())
for lb in (30,):
    cut = today - pd.Timedelta(days=lb)
    inw = m[m["date"] >= cut]
    print(f"\naffected keys with date >= {cut.date()} (LOOKBACK_DAYS={lb}): {len(inw)}")
    if len(inw):
        print(inw.groupby(inw["date"].dt.date).size().to_string())
        print("max |delta| in window:", inw["delta"].abs().max())
        print("latest computed_at in window:", inw["computed_at"].max())

# when does the window clear?
if len(m[m["date"] >= today - pd.Timedelta(days=60)]):
    last = m["date"].max()
    print("\nlatest affected date overall:", last.date(),
          "-> leaves a 30-day window on", (last + pd.Timedelta(days=30)).date())
