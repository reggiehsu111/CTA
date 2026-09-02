#!/usr/bin/env python3
"""QNT-21 backfill: rewrite mtx_signal_values.pnl_1d on the roll-adjusted legs.

Rewrites ONLY the pnl_1d column, ONLY on the 6,277 (date, signal_name, variant)
rows listed in backfill_rows.csv. Nothing else is touched: raw_value, position,
metadata and computed_at are left as-is, and mtx_signal_config is never opened.

    new_pnl_1d = old_pnl_1d + position * (roll_adjusted_ret - raw_ret)

which is exact because the runner's pnl is

    pnl = position * ret - |Δposition| * cost

and the cost leg is unchanged by the fix. Verified: reconstructing every stored
pnl_1d from the stored position under the OLD legs reproduces the DB to
max|Δ| = 0.000e+00 across all 66 signal x variant series.

One transaction, with a pre-flight key check and a post-verify read-back.
Pre-image of every affected row is on disk at backfill_preimage.csv.

Usage:  python3 apply_backfill.py --execute      (omit --execute for a dry run)
"""
import sys, argparse, datetime as dt
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
import pandas as pd
from sqlalchemy import text
from db_utils import engine

HERE     = "/home/ubuntu/mtx/signal_zoo/roll_rescore"
EXPECTED = 6277

SQL = text(
    "UPDATE mtx_signal_values SET pnl_1d = :new_pnl "
    "WHERE date = :d AND signal_name = :s AND variant = :v"
)

ap = argparse.ArgumentParser()
ap.add_argument("--execute", action="store_true")
args = ap.parse_args()

d = pd.read_csv(f"{HERE}/backfill_rows.csv", parse_dates=["date"])
assert len(d) == EXPECTED,                        f"row count {len(d)} != expected {EXPECTED}"
assert d.isna().sum().sum() == 0,                 "NaN in backfill set"
assert set(d["variant"]) == {"c2c", "o2o", "noonpause"}, "unexpected variant"
assert not d.duplicated(["date", "signal_name", "variant"]).any(), "duplicate key"

params = [
    {"d": r.date.date(), "s": r.signal_name, "v": r.variant, "new_pnl": float(r.new_pnl_1d)}
    for r in d.itertuples()
]

print(f"rows to update : {len(params)}")
print(f"variants       : {d['variant'].value_counts().to_dict()}")
print(f"date range     : {d['date'].min().date()} .. {d['date'].max().date()}")
print(f"statement      : {SQL.text}")

# ---- pre-flight: every key must already exist, and hold the old value we expect
cur = pd.read_sql(
    "SELECT date, signal_name, variant, pnl_1d FROM mtx_signal_values "
    "WHERE variant IN ('c2c','o2o','noonpause')", engine)
cur["date"] = pd.to_datetime(cur["date"])
chk = d.merge(cur, on=["date", "signal_name", "variant"], how="left")
assert chk["pnl_1d"].notna().all(), "some target rows are missing from the table"
drift = (chk["pnl_1d"] - chk["old_pnl_1d"]).abs().max()
print(f"pre-flight     : all {len(chk)} keys present, max|db - expected old| = {drift:.3e}")
assert drift < 1e-12, "the table moved since backfill_rows.csv was built - rebuild it"

n_before = pd.read_sql("SELECT count(*) n FROM mtx_signal_values", engine)["n"][0]
print(f"row count now  : {n_before}")

if not args.execute:
    print("\nDRY RUN - nothing written. Re-run with --execute.")
    sys.exit(0)

with engine.begin() as conn:
    conn.execute(SQL, params)

# ---- post-verify against the DB
back = pd.read_sql(
    "SELECT date, signal_name, variant, pnl_1d FROM mtx_signal_values "
    "WHERE variant IN ('c2c','o2o','noonpause')", engine)
back["date"] = pd.to_datetime(back["date"])
m = d.merge(back, on=["date", "signal_name", "variant"], how="left")
err = (m["pnl_1d"] - m["new_pnl_1d"]).abs().max()
n_tot = pd.read_sql("SELECT count(*) n FROM mtx_signal_values", engine)["n"][0]
print(f"post-verify    : max|db - intended| = {err:.3e}; table row count = {n_tot}")
assert err < 1e-12, "post-verify failed"
assert n_tot == n_before, f"row count moved {n_before} -> {n_tot}: nothing should have been inserted or deleted"

with open("/home/ubuntu/mtx/audit_log.txt", "a") as fh:
    fh.write(
        f"\n{dt.datetime.utcnow().isoformat()}Z  QNT-21 roll-fix backfill\n"
        f"  approved_by : Reggie (Linear QNT-21) - 'Apply + backfill' 2026-09-01 06:34,\n"
        f"                exact-SQL approval 'Approve' 2026-09-01 07:13\n"
        f"  statement   : {SQL.text}\n"
        f"  rows        : {EXPECTED} (c2c 2643, o2o 2562, noonpause 1072) of {n_tot}\n"
        f"  date range  : {d['date'].min().date()} .. {d['date'].max().date()}\n"
        f"  columns     : pnl_1d only\n"
        f"  pre-image   : {HERE}/backfill_preimage.csv\n"
        f"  rollback    : restore pnl_1d from the pre-image CSV on (date, signal_name, variant)\n")
print("audit_log.txt appended. DONE.")
