"""Tick-level MXF/TXF data from S3, with a local cache.

Written by the `shioaji-ingest` repo (fetch_shioaji_ticks.py) as one snappy parquet per
trading day, mirrored to s3://reggie-f-cache-tw/shioaji/ticks/<SYMBOL>/YYYYMMDD.parquet.
S3 is the source of truth; this module keeps a local mirror under ~/.f_cache/tw/shioaji/
so repeat reads cost nothing, matching the s3_cache pattern used elsewhere in the repo.

    import cta
    df   = cta.load_ticks('MXFR1', '2026-08-01', '2026-08-31')
    days = cta.available_tick_days('MXFR1')
    q    = cta.tick_quotes('MXFR1', '2026-08-26')     # spread / mid / imbalance added

Columns as fetched: ts, close, volume, bid_price, bid_volume, ask_price, ask_volume,
tick_type. Roughly 193k rows per MXF trading day, so a month is ~4M rows — load a range
you actually need rather than the lot.

WHAT THIS IS FOR
----------------
Bid/ask is the only way to *measure* execution cost instead of assuming it. The stop-loss
work concluded the ACE loss was execution-layer rather than signal; effective spread and
realised slippage by time of day are computable from here and were not from daily bars.

CONTRACT NAMES
--------------
TXF = 臺股期貨 (大台, multiplier 200), MXF = 小型臺指期貨 (小台, multiplier 50),
TMF = 微型臺指期貨 (multiplier 10, no history before 2025). R1 is the front-month
continuous series and rolls one session BEFORE the daily tw_index_futures_pv front_expiry
changes — harmless for within-day microstructure, wrong if you compute returns across a
roll boundary.
"""
from __future__ import annotations

import os
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

S3_ROOT = os.environ.get("SHIOAJI_S3_ROOT", "s3://reggie-f-cache-tw/shioaji/ticks")
CACHE = Path(os.environ.get("SHIOAJI_TICK_CACHE",
                            Path.home() / ".f_cache" / "tw" / "shioaji" / "ticks"))
TICK_TYPE = {0: "neutral", 1: "sell", 2: "buy"}   # 內盤/外盤


def _aws(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env.setdefault("AWS_PROFILE", "crypto_project")
    return subprocess.run(["aws", *args], capture_output=True, text=True, env=env)


def available_tick_days(symbol: str = "MXFR1") -> list[str]:
    """YYYYMMDD stems present in S3 (the authoritative list)."""
    r = _aws("s3", "ls", f"{S3_ROOT}/{symbol}/")
    if r.returncode != 0:
        raise RuntimeError(f"cannot list {S3_ROOT}/{symbol}/: {r.stderr.strip()[:200]}")
    return sorted(ln.split()[-1].replace(".parquet", "")
                  for ln in r.stdout.splitlines() if ln.strip().endswith(".parquet"))


def _local(symbol: str, stem: str) -> Path:
    return CACHE / symbol / f"{stem}.parquet"


def _ensure(symbol: str, stem: str) -> Path | None:
    """Local path for one day, pulling from S3 on a miss. None if absent upstream."""
    p = _local(symbol, stem)
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    r = _aws("s3", "cp", f"{S3_ROOT}/{symbol}/{stem}.parquet", str(p), "--only-show-errors")
    if r.returncode != 0 or not p.exists():
        if p.exists():
            p.unlink()      # never leave a truncated cache file behind
        return None
    return p


def load_ticks(symbol: str = "MXFR1", start=None, end=None,
               columns: list[str] | None = None, verbose: bool = False) -> pd.DataFrame:
    """Ticks for [start, end] inclusive, concatenated and sorted by ts.

    Missing days are skipped silently — the backfill runs a bounded number of days per
    session, so gaps are expected while it catches up. Compare len(available_tick_days())
    against your range if completeness matters.
    """
    days = available_tick_days(symbol)
    if start:
        s = pd.Timestamp(start).strftime("%Y%m%d")
        days = [d for d in days if d >= s]
    if end:
        e = pd.Timestamp(end).strftime("%Y%m%d")
        days = [d for d in days if d <= e]
    if not days:
        raise ValueError(f"no {symbol} tick days in range {start}..{end}; "
                         f"available: {available_tick_days(symbol)[:1]} .. "
                         f"{available_tick_days(symbol)[-1:]}")

    frames, missing = [], 0
    for stem in days:
        p = _ensure(symbol, stem)
        if p is None:
            missing += 1
            continue
        frames.append(pd.read_parquet(p, columns=columns))
    if not frames:
        raise RuntimeError(f"every day in range failed to download for {symbol}")
    df = pd.concat(frames, ignore_index=True)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
    if verbose:
        print(f"{symbol}: {len(df):,} ticks over {len(frames)} days "
              f"({days[0]}..{days[-1]})" + (f", {missing} unavailable" if missing else ""))
    return df


def tick_quotes(symbol: str = "MXFR1", start=None, end=None,
                verbose: bool = False) -> pd.DataFrame:
    """load_ticks plus the derived microstructure columns.

    mid           (bid+ask)/2, the reference price a fill should be measured against
    spread        ask - bid, in index points
    spread_bps    spread / mid * 1e4
    depth_imb     (bid_volume - ask_volume) / (bid_volume + ask_volume), in [-1, 1]
    signed_vol    volume signed by tick_type (buy +, sell -), for order-flow imbalance
    eff_spread    2 * |close - mid|, the standard effective-spread measure of what a
                  marketable order actually paid relative to the prevailing mid

    Rows with a zero/absent quote are dropped for the quote-derived columns: the first
    tick of a session carries bid=ask=0 and would otherwise produce a spread of the full
    index level.
    """
    df = load_ticks(symbol, start, end, verbose=verbose)
    ok = (df["bid_price"] > 0) & (df["ask_price"] > 0) & (df["ask_price"] >= df["bid_price"])
    df = df[ok].copy()
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_bps"] = df["spread"] / df["mid"] * 1e4
    tot = df["bid_volume"] + df["ask_volume"]
    df["depth_imb"] = ((df["bid_volume"] - df["ask_volume"]) / tot.where(tot > 0)).fillna(0.0)
    sign = df["tick_type"].map({1: -1, 2: 1}).fillna(0)
    df["signed_vol"] = df["volume"] * sign
    df["eff_spread"] = 2 * (df["close"] - df["mid"]).abs()
    return df.reset_index(drop=True)


def session_date(ts: pd.Series) -> pd.Series:
    """Map wall-clock timestamps to the TAIFEX trading day they belong to.

    The night session for trading day t opens at 15:00 on t-1 and runs to 05:00 on t, so
    ts.dt.normalize() splits one session across two calendar dates — asking for 2026-08-25
    onward returns rows stamped 2026-08-24. Everything at or after 15:00 belongs to the
    NEXT trading day.

    This is the same convention that produced the night-into-day-column bug fixed in
    62e5255; getting it wrong here silently mixes two sessions into one row.

    Note this yields calendar days, not trading days — a Friday 15:00 session belongs to
    Monday. Reindex onto the trading calendar if you need exact alignment.
    """
    ts = pd.to_datetime(ts)
    return (ts.dt.normalize() + pd.to_timedelta((ts.dt.hour >= 15).astype(int), unit="D"))


def tick_daily_summary(symbol: str = "MXFR1", start=None, end=None,
                       by_session: bool = True) -> pd.DataFrame:
    """One row per session — the shape worth persisting to Postgres.

    Raw ticks are ~122 bytes/row in Postgres against ~12 as parquet, so keep the ticks in
    S3 and store aggregates like this (a few hundred rows a year) in the database.

    by_session=True groups by TAIFEX trading day (night session attributed forward to the
    day it belongs to). Set False for raw calendar-date grouping, which splits sessions.
    """
    q = tick_quotes(symbol, start, end)
    q["date"] = session_date(q["ts"]) if by_session else q["ts"].dt.normalize()
    g = q.groupby("date")
    out = pd.DataFrame({
        "n_ticks": g.size(),
        "volume": g["volume"].sum(),
        "spread_mean": g["spread"].mean(),
        "spread_med": g["spread"].median(),
        "spread_bps_mean": g["spread_bps"].mean(),
        "eff_spread_mean": g["eff_spread"].mean(),
        "depth_imb_mean": g["depth_imb"].mean(),
        "signed_vol": g["signed_vol"].sum(),
        "close": g["close"].last(),
    })
    out["ofi"] = out["signed_vol"] / out["volume"].where(out["volume"] > 0)
    return out
