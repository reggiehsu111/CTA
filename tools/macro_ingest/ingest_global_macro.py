#!/usr/bin/env python3
"""
QNT-10 part 2 -- ingest global / regional macro history into `quant_data`.

Creates TWO NEW tables and upserts into them. It never ALTERs, DROPs, TRUNCATEs
or writes to any pre-existing table, and it never deletes a row.

    macro_series_meta   one row per series: source, units, frequency, pub lag
    macro_series        (series_id, date, value)

See `global_macro_sources.py` for the registry, the publication-lag convention,
and the list of sources that were probed and REJECTED.

Usage:
    python3 ingest_global_macro.py            # dry run: fetch, validate, report
    python3 ingest_global_macro.py --commit   # create tables and upsert
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx/tools/macro_ingest")

from db_utils import engine            # noqa: E402
import fetch_global_macro as F         # noqa: E402
import global_macro_sources as S       # noqa: E402

AUDIT = "/home/ubuntu/mtx/audit_log.txt"

DDL = [
"""
CREATE TABLE IF NOT EXISTS macro_series_meta (
    series_id     text PRIMARY KEY,
    label         text NOT NULL,
    country       text NOT NULL,       -- TW / JP / KR / CN / US / GL
    category      text NOT NULL,       -- fx / trade / rates / semis / cycle / ...
    freq          text NOT NULL,       -- D / W / M / Q
    units         text NOT NULL,
    source        text NOT NULL,
    source_id     text NOT NULL,
    source_url    text NOT NULL,
    -- CONSERVATIVE calendar-day lag from the first day of the reference period
    -- to first public availability. Not a scraped release date. See the module
    -- docstring in global_macro_sources.py before using this in a backtest.
    pub_lag_days  integer NOT NULL,
    first_obs     date,
    last_obs      date,
    n_obs         integer,
    updated_at    timestamptz NOT NULL DEFAULT now()
)
""",
"""
CREATE TABLE IF NOT EXISTS macro_series (
    series_id  text NOT NULL,
    date       date NOT NULL,
    value      double precision NOT NULL,
    PRIMARY KEY (series_id, date)
)
""",
"CREATE INDEX IF NOT EXISTS macro_series_date_idx ON macro_series (date)",
]

UPSERT_META = text("""
INSERT INTO macro_series_meta
    (series_id, label, country, category, freq, units, source, source_id,
     source_url, pub_lag_days, first_obs, last_obs, n_obs, updated_at)
VALUES
    (:series_id, :label, :country, :category, :freq, :units, :source, :source_id,
     :source_url, :pub_lag_days, :first_obs, :last_obs, :n_obs, now())
ON CONFLICT (series_id) DO UPDATE SET
    label = EXCLUDED.label, country = EXCLUDED.country,
    category = EXCLUDED.category, freq = EXCLUDED.freq, units = EXCLUDED.units,
    source = EXCLUDED.source, source_id = EXCLUDED.source_id,
    source_url = EXCLUDED.source_url, pub_lag_days = EXCLUDED.pub_lag_days,
    first_obs = EXCLUDED.first_obs, last_obs = EXCLUDED.last_obs,
    n_obs = EXCLUDED.n_obs, updated_at = now()
""")

UPSERT_VAL = text("""
INSERT INTO macro_series (series_id, date, value)
VALUES (:series_id, :date, :value)
ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value
""")


def audit(msg: str) -> None:
    with open(AUDIT, "a") as fh:
        fh.write(f"[{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}] QNT-10 {msg}\n")


def before_counts(conn) -> tuple[int, int]:
    def one(tbl):
        try:
            return conn.execute(text(f"SELECT count(*) FROM {tbl}")).scalar()
        except Exception:
            return 0
    return one("macro_series"), one("macro_series_meta")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="actually write to the DB")
    ap.add_argument("--only", help="comma-separated series_ids, for re-running a subset")
    args = ap.parse_args()

    series = dict(S.FRED_SERIES)
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        missing = want - set(series)
        if missing:
            print(f"unknown series_id(s): {sorted(missing)}", file=sys.stderr)
            return 2
        series = {k: v for k, v in series.items() if k in want}

    print(f"fetching {len(series)} series from FRED ...")
    frames, problems = F.fetch_all(series)

    stale = [p for p in problems if "STALE" in p or "EMPTY" in p]
    for p in problems:
        print("  ! " + p)
    if stale:
        # A frozen source is the exact failure this ticket was written around.
        # Refuse the whole run rather than quietly landing a dead series.
        print(f"\nABORT: {len(stale)} series failed the freshness check. "
              "Fix or move them to REJECTED in global_macro_sources.py.", file=sys.stderr)
        return 1

    rows = sum(len(d) for d in frames.values())
    print(f"\nfetched {len(frames)} series, {rows:,} observations")
    print(f"{'series_id':<22}{'source_id':<22}{'freq':<6}{'first':<12}{'last':<12}{'n':>8}  lag")
    for sid, spec in series.items():
        d = frames[sid]
        print(f"{sid:<22}{spec[0]:<22}{spec[4]:<6}{str(d.date.iloc[0]):<12}"
              f"{str(d.date.iloc[-1]):<12}{len(d):>8}  {spec[6]}d")

    if not args.commit:
        print("\nDRY RUN -- nothing written. Re-run with --commit to load.")
        return 0

    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
        v0, m0 = before_counts(conn)

        for sid, spec in series.items():
            fred_id, label, country, cat, freq, units, lag = spec
            d = frames[sid]
            conn.execute(UPSERT_META, dict(
                series_id=sid, label=label, country=country, category=cat,
                freq=freq, units=units, source="FRED (Federal Reserve Bank of St. Louis)",
                source_id=fred_id, source_url=S.FRED_CSV.format(id=fred_id),
                pub_lag_days=lag, first_obs=d.date.iloc[0], last_obs=d.date.iloc[-1],
                n_obs=len(d)))
            payload = [dict(series_id=sid, date=r.date, value=float(r.value))
                       for r in d.itertuples()]
            for i in range(0, len(payload), 5000):
                conn.execute(UPSERT_VAL, payload[i:i + 5000])
            print(f"  upserted {sid:<22} {len(payload):>8,} rows")

        v1, m1 = before_counts(conn)

    print(f"\nmacro_series      {v0:,} -> {v1:,} rows  (+{v1 - v0:,})")
    print(f"macro_series_meta {m0:,} -> {m1:,} rows  (+{m1 - m0:,})")
    audit(f"ingest_global_macro --commit: {len(series)} series; "
          f"macro_series {v0}->{v1}, macro_series_meta {m0}->{m1}; "
          f"source FRED CSV; CREATE TABLE IF NOT EXISTS + INSERT ON CONFLICT DO UPDATE only; "
          f"no existing table read or written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
