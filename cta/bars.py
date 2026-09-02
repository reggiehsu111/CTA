"""Assemble intraday bars for a symbol from every source we have.

`_load_intraday` used to read one hardcoded file, `MXFR1k.csv` — MXF only, ending
2026-05-22. That file is now one source among several:

  1. legacy CSV   MXFR1k.csv               MXF family only, 2023-03-13 .. 2026-05-22
  2. local parquet $SHIOAJI_DATA_DIR (or <repo>/../Cache/shioaji)/<SYMBOL>/YYYYMM.parquet
  3. S3 parquet   s3://reggie-f-cache-tw/shioaji/kbars/<SYMBOL>/YYYYMM.parquet

Written by the shioaji-ingest repo (fetch_shioaji_history.py). Sources are unioned and
de-duplicated on timestamp with **later sources winning**, so a re-fetch supersedes the
legacy file rather than colliding with it.

    from cta.bars import load_1min, bar_coverage
    df = load_1min("MXFR1", verbose=True)
    bar_coverage("MXFR1")          # provenance: which source supplied what

GAPS ARE REPORTED, NOT HIDDEN
-----------------------------
Splicing sources silently is dangerous: as of 2026-09 the legacy CSV ends 2026-05-22 and
the Shioaji parquet begins 2026-06-01, so a naive concat produces one "1-minute" bar whose
return actually spans nine days. `load_1min(verbose=True)` prints the largest gaps and
`bar_coverage` shows the seams. Weekend gaps (~2.1 days, Sat 05:00 -> Mon 08:45) are
normal; anything much larger is missing data or a holiday.

CONTRACT SERIES
---------------
MXFR1/TXFR1 are front-month *continuous* and roll one session before the daily series'
front_expiry changes. Fine within a session; wrong if you compute returns across a roll.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

_OHLCV = ["open", "high", "low", "close", "volume"]

# 'mtx' and 'mxf' both mean the 小台 front-continuous series
ALIASES = {"MTX": "MXFR1", "MXF": "MXFR1", "MXFR1": "MXFR1",
           "TX": "TXFR1", "TXF": "TXFR1", "TXFR1": "TXFR1",
           "TMF": "TMFR1", "TMFR1": "TMFR1"}

LEGACY_CSV_SYMBOLS = {"MXFR1"}          # MXFR1k.csv is MXF only
S3_ROOT = os.environ.get("SHIOAJI_S3_KBARS", "s3://reggie-f-cache-tw/shioaji/kbars")


def resolve_symbol(asset: str) -> str:
    code = asset.upper()
    if code not in ALIASES:
        raise ValueError(f"unknown intraday symbol {asset!r}; known: {sorted(set(ALIASES))}")
    return ALIASES[code]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _legacy_csv() -> Path:
    return _repo_root() / "MXFR1k.csv"


def _local_parquet_dir(symbol: str) -> Path:
    base = os.environ.get("SHIOAJI_DATA_DIR")
    root = Path(base) if base else _repo_root().parent / "Cache" / "shioaji"
    return root / symbol


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
    df.index = pd.to_datetime(df.index)
    keep = [c for c in _OHLCV if c in df.columns]
    return df[keep].sort_index()


def _read_legacy(symbol: str) -> pd.DataFrame | None:
    p = _legacy_csv()
    if symbol not in LEGACY_CSV_SYMBOLS or not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["ts"],
                     dtype={"Open": "float32", "High": "float32", "Low": "float32",
                            "Close": "float32", "Volume": "int32", "Amount": "float64"})
    return _norm(df)


def _read_local_parquet(symbol: str) -> pd.DataFrame | None:
    d = _local_parquet_dir(symbol)
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return None
    return _norm(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))


def _read_s3_parquet(symbol: str, cache_dir: Path | None = None) -> pd.DataFrame | None:
    """Pull months present in S3 but not locally. Silent no-op if aws is unavailable."""
    env = {**os.environ}
    env.setdefault("AWS_PROFILE", "crypto_project")
    r = subprocess.run(["aws", "s3", "ls", f"{S3_ROOT}/{symbol}/"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return None
    stems = [ln.split()[-1] for ln in r.stdout.splitlines() if ln.strip().endswith(".parquet")]
    if not stems:
        return None
    cache = cache_dir or (Path.home() / ".f_cache" / "tw" / "shioaji" / "kbars" / symbol)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in stems:
        local = cache / s
        if not local.exists():
            g = subprocess.run(["aws", "s3", "cp", f"{S3_ROOT}/{symbol}/{s}", str(local),
                                "--only-show-errors"], capture_output=True, text=True, env=env)
            if g.returncode != 0:
                continue
        frames.append(pd.read_parquet(local))
    return _norm(pd.concat(frames, ignore_index=True)) if frames else None


# later entries win on duplicate timestamps
_SOURCES = [("legacy_csv", _read_legacy),
            ("local_parquet", _read_local_parquet),
            ("s3_parquet", _read_s3_parquet)]


def bar_coverage(symbol: str = "MXFR1") -> pd.DataFrame:
    """What each source supplies — provenance, before anything is merged."""
    sym = resolve_symbol(symbol)
    rows = []
    for name, fn in _SOURCES:
        try:
            df = fn(sym)
        except Exception as e:
            rows.append({"source": name, "bars": 0, "start": None, "end": None,
                         "note": f"{type(e).__name__}: {str(e)[:60]}"})
            continue
        if df is None or df.empty:
            rows.append({"source": name, "bars": 0, "start": None, "end": None,
                         "note": "absent"})
        else:
            rows.append({"source": name, "bars": len(df), "start": df.index.min(),
                         "end": df.index.max(), "note": ""})
    return pd.DataFrame(rows)


def load_1min(symbol: str = "MXFR1", start=None, end=None,
              use_s3: bool = True, verbose: bool = False,
              gap_report: int = 5) -> pd.DataFrame:
    """1-minute OHLCV for `symbol`, unioned across all available sources.

    Duplicate timestamps resolve to the LAST source that has them (S3 > local > legacy),
    which is also what makes a re-fetch authoritative over the legacy CSV.
    """
    sym = resolve_symbol(symbol)
    frames, used = [], []
    for name, fn in _SOURCES:
        if name == "s3_parquet" and not use_s3:
            continue
        try:
            df = fn(sym)
        except Exception as e:
            if verbose:
                print(f"  {name}: skipped ({type(e).__name__}: {str(e)[:60]})")
            continue
        if df is not None and not df.empty:
            frames.append(df)
            used.append((name, len(df), df.index.min(), df.index.max()))
    if not frames:
        raise FileNotFoundError(
            f"no intraday bars for {sym}. Expected one of: {_legacy_csv()}, "
            f"{_local_parquet_dir(sym)}/*.parquet, or {S3_ROOT}/{sym}/")

    df = pd.concat(frames)
    # keep='last' also absorbs the ~20k duplicate timestamps the legacy CSV carries around
    # rollovers, where the same minute appears on both the old and new contract.
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end) + pd.Timedelta(days=1)]

    if verbose:
        print(f"{sym}: {len(df):,} bars  {df.index.min()} .. {df.index.max()}")
        for name, n, lo, hi in used:
            print(f"  {name:15} {n:8,d} bars  {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}")
        gaps = df.index.to_series().diff()
        big = gaps.nlargest(gap_report)
        big = big[big > pd.Timedelta(days=3)]
        if len(big):
            print(f"  gaps > 3 days ({len(big)} shown) — weekends are ~2.1d and normal:")
            for ts, g in big.items():
                print(f"    {g.days:3d}d {str(g).split(',')[-1].strip():8}  ending {ts:%Y-%m-%d %H:%M}")
        else:
            print("  no gaps > 3 days")
    return df
