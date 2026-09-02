#!/usr/bin/env python3
"""
QNT-10 — ingest Taiwan and Japan macro history into `quant_data`.

Creates three NEW tables (it never ALTERs, DROPs, TRUNCATEs or touches any
existing table) and upserts into them:

    tw_macro_monthly   CPI, unemployment, M1A/M1B/M2, PMI/NMI, NDC 景氣指標
    jp_macro_monthly   CPI (all/core/energy/food), unemployment, BoJ assets, 10y JGB
    jp_markets_daily   Nikkei 225, USD/JPY

Writes are INSERT ... ON CONFLICT DO UPDATE keyed on `date`, so re-running is
idempotent and no row is ever deleted.

Usage:
    python3 ingest_asia_macro.py             # dry run: fetch, validate, print DDL + counts
    python3 ingest_asia_macro.py --commit    # actually create tables and upsert
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx/tools/macro_ingest")

from db_utils import engine          # noqa: E402
import fetch_asia_macro as F         # noqa: E402
import asia_macro_sources as S       # noqa: E402

AUDIT = "/home/ubuntu/mtx/audit_log.txt"

# ── DDL — new tables only ───────────────────────────────────────────────────
DDL = {
"tw_macro_monthly": """
CREATE TABLE IF NOT EXISTS tw_macro_monthly (
    date                  date PRIMARY KEY,
    -- 主計總處 CPI, 民國110年(2021)=100; *_yoy in percent
    cpi                   numeric, cpi_yoy               numeric,
    cpi_food              numeric, cpi_food_yoy          numeric,
    cpi_apparel           numeric, cpi_apparel_yoy       numeric,
    cpi_housing           numeric, cpi_housing_yoy       numeric,
    cpi_transport         numeric, cpi_transport_yoy     numeric,
    cpi_health            numeric, cpi_health_yoy        numeric,
    cpi_education         numeric, cpi_education_yoy     numeric,
    cpi_misc              numeric, cpi_misc_yoy          numeric,
    -- 主計總處 人力資源調查, percent NSA
    unemployment_rate     numeric,
    -- 中央銀行 貨幣總計數, 百萬元 (millions TWD), daily-average basis
    m1a numeric, m1a_yoy numeric,
    m1b numeric, m1b_yoy numeric,
    m2  numeric, m2_yoy  numeric,
    -- 中華經濟研究院 PMI / NMI, diffusion index (50 = no change)
    pmi                   numeric, nmi                   numeric,
    -- 國發會 景氣指標
    leading_idx           numeric, leading_idx_nt        numeric,
    coincident_idx        numeric, coincident_idx_nt     numeric,
    lagging_idx           numeric, lagging_idx_nt        numeric,
    monitor_score         numeric,      -- 景氣對策信號綜合分數, 9-45
    monitor_signal        text,         -- 藍 / 黃藍 / 綠 / 黃紅 / 紅
    export_orders_dci     numeric,      -- 外銷訂單動向指數 (以家數計)
    semi_equip_imports    numeric,      -- 名目半導體設備進口, 新臺幣百萬元
    industrial_production numeric,      -- 工業生產指數, 2021=100
    mfg_sales_idx         numeric,      -- 製造業銷售量指數, 2021=100
    customs_exports       numeric,      -- 海關出口值, 十億元 (billions TWD)
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
)""",

"jp_macro_monthly": """
CREATE TABLE IF NOT EXISTS jp_macro_monthly (
    date              date PRIMARY KEY,
    -- OECD SDMX COICOP-2018; index 2020=100 (NSA preferred), *_yoy in percent
    cpi               numeric, cpi_yoy        numeric,
    cpi_core          numeric, cpi_core_yoy   numeric,   -- ex food & energy
    cpi_energy        numeric, cpi_energy_yoy numeric,
    cpi_food          numeric, cpi_food_yoy   numeric,
    unemployment_rate numeric,   -- percent, SA (FRED LRHUTTTTJPM156S)
    boj_assets        numeric,   -- 100 million JPY (FRED JPNASSETS)
    jgb10y            numeric,   -- percent p.a., monthly avg (FRED IRLTLT01JPM156N)
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
)""",

"jp_markets_daily": """
CREATE TABLE IF NOT EXISTS jp_markets_daily (
    date       date PRIMARY KEY,
    nikkei225  numeric,   -- index points, close (FRED NIKKEI225)
    usdjpy     numeric,   -- JPY per USD, NY noon (FRED DEXJPUS)
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
)""",
}


def _sanity(name: str, df: pd.DataFrame, key_col: str, min_last: str) -> None:
    """Refuse to write a frame that is empty, stale, or duplicated.

    Deliberately strict: a silently stale or half-empty macro table is the
    failure mode that costs weeks, not an exception at ingest time.
    """
    if df.empty:
        raise RuntimeError(f"{name}: empty frame — refusing to write")
    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()][:5].tolist()
        raise RuntimeError(f"{name}: duplicate dates {dupes}")
    if key_col not in df.columns:
        raise RuntimeError(f"{name}: missing key column {key_col}")
    last = df[key_col].last_valid_index()
    if last is None or last < pd.Timestamp(min_last):
        raise RuntimeError(
            f"{name}: {key_col} last valid {last} < required {min_last} — "
            f"source looks stale, refusing to write")


def _upsert(conn, table: str, df: pd.DataFrame) -> int:
    df = df.copy()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Drop all-NaN rows so we never write a row of pure NULLs.
    value_cols = [c for c in df.columns if c != "date"]
    df = df.dropna(subset=value_cols, how="all")
    df = df.astype(object).where(pd.notna(df), None)

    tmp = f"_tmp_{table}"
    conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
    df.to_sql(tmp, conn, if_exists="replace", index=False, method="multi", chunksize=500)
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in value_cols)
    conn.execute(text(f"""
        INSERT INTO {table} ({', '.join(['date'] + value_cols)})
        SELECT {', '.join(['date'] + value_cols)} FROM {tmp}
        ON CONFLICT (date) DO UPDATE SET {sets}, updated_at = now()
    """))
    conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))
    return len(df)


def _stats(conn, table: str) -> str:
    exists = conn.execute(text(
        "SELECT to_regclass(:t)"), {"t": f"public.{table}"}).scalar()
    if not exists:
        return "does not exist"
    r = conn.execute(text(
        f"SELECT count(*), min(date), max(date) FROM {table}")).fetchone()
    return f"{r[0]} rows, {r[1]} → {r[2]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="create tables and write; without it this is a dry run")
    args = ap.parse_args()

    print("Fetching Taiwan sources …")
    tw = pd.concat([F.fetch_tw_cpi(), F.fetch_tw_unemployment(), F.fetch_tw_money(),
                    F.fetch_tw_pmi(), F.fetch_tw_cycle()], axis=1).sort_index()
    tw.index.name = "date"
    print("Fetching Japan sources …")
    jp_m = F.fetch_jp_monthly()
    jp_d = F.fetch_jp_daily()

    _sanity("tw_macro_monthly", tw,   "cpi",       "2026-06-01")
    _sanity("jp_macro_monthly", jp_m, "cpi",       "2026-05-01")
    _sanity("jp_markets_daily", jp_d, "nikkei225", "2026-08-25")

    frames = {"tw_macro_monthly": tw, "jp_macro_monthly": jp_m, "jp_markets_daily": jp_d}
    for name, df in frames.items():
        print(f"\n=== {name}: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")
        print(df.notna().sum().to_string())

    if not args.commit:
        print("\n--- DRY RUN. DDL that would run: ---")
        for d in DDL.values():
            print(d.strip(), ";\n")
        with engine.connect() as conn:
            for name in frames:
                print(f"  BEFORE {name}: {_stats(conn, name)}")
        print("\nRe-run with --commit to apply.")
        return 0

    with engine.connect() as conn:
        before = {n: _stats(conn, n) for n in frames}
        for name, ddl in DDL.items():
            conn.execute(text(ddl))
        conn.commit()
        written = {}
        for name, df in frames.items():
            written[name] = _upsert(conn, name, df)
            conn.commit()
        after = {n: _stats(conn, n) for n in frames}

    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines = [f"\n[{ts}] QNT-10 asia macro ingest (agent: Claw-EC2, unattended)"]
    for n in frames:
        lines.append(f"  {n}: upserted {written[n]} | before: {before[n]} | after: {after[n]}")
    lines.append("  DDL: CREATE TABLE IF NOT EXISTS on 3 NEW tables; no ALTER/DROP/TRUNCATE;")
    lines.append("       writes are INSERT ... ON CONFLICT DO UPDATE on `date` only.")
    lines.append("  sources: " + "; ".join(v["dataset"] for v in S.TW_SOURCES.values())
                 + "; OECD SDMX COICOP2018 (JP CPI); FRED "
                 + ",".join(c["id"] for c in
                            list(S.JP_FRED_MONTHLY.values()) + list(S.JP_FRED_DAILY.values())))
    with open(AUDIT, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
