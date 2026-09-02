"""
Taiwan market margin & short-selling loader — including 大盤融資維持率.

Data sources
------------
1. CSV files produced by ``QuantResearch/tools/scrape_twse_margin.py``:
     - ``tw_margin_summary.csv``   — one row per (date, board), aggregate totals
     - ``tw_margin_by_stock.csv``  — one row per (date, board, stock),
                                       per-stock margin & short balances

2. Per-stock close prices from RDS ``tw_spot_pv``  (already ingested by the
   ``tw_daily_ingest`` Lambda; loaded via ``Libs/db_loader.py``).

大盤融資維持率  (market margin maintenance ratio)
--------------------------------------------------
    大盤融資維持率 =
        Σᵢ  (close_i × 融資今日餘額股數_i)          ← collateral value (NTD)
        ────────────────────────────────────────
        Σᵢ  (融資今日餘額金額_i, in NTD)             ← margin debt outstanding
        × 100 %

The numerator sums across every stock in both boards (TSE + OTC).
The denominator is available directly from the daily summary CSV.

Typical range: 130 % – 170 %.  Individual-account tripwires — 追繳保證金 at
140 %, 斷頭 at 130 % — bracket the aggregate range.

Usage
-----
    >>> import cta
    >>> mr = cta.load_market_maintenance_ratio()
    >>> mr.tail(5)
    date
    2026-07-24    159.42
    2026-07-27    158.11
    2026-07-28    155.83
    Name: 大盤融資維持率, dtype: float64

    >>> # Align to an MTX trading calendar (fills forward like NFCI):
    >>> ASSET = cta.load_asset('mtx', '1d')
    >>> mr = cta.load_market_maintenance_ratio(trading_index=ASSET.index)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


# ── Default CSV location ─────────────────────────────────────────────────────
_DEFAULT_MARGIN_DIR = Path(__file__).resolve().parent.parent / "tw_margin"


# ── RDS spot-price loader (delegates to Libs/db_loader.py) ───────────────────
def _load_close_wide(td_index: pd.DatetimeIndex,
                     board: str) -> pd.DataFrame:
    """Return a (date × ticker) close-price matrix for one board.

    Delegates to ``Libs/db_loader.load_spot_from_db``. The board field name
    differs between the two boards — TSE uses 「收盤價」, OTC/TPEx uses 「收盤」.
    """
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path:
        sys.path.insert(0, _LIBS)
    from db_loader import load_spot_from_db

    spot = load_spot_from_db(td_index)
    fld_name = "收盤價" if board == "TSE" else "收盤"
    return spot.get(board, {}).get(fld_name, pd.DataFrame())


# ── Raw margin loaders — RDS preferred, CSV fallback ─────────────────────────
_SUMMARY_CACHE: pd.DataFrame | None = None
_STOCK_CACHE:   pd.DataFrame | None = None
_LAST_SOURCE:   str | None = None
_LAST_DIR:      Path | None = None


def _rds_table_exists(table: str) -> bool:
    """Return True if the given RDS table has ≥ 1 row (source of truth)."""
    try:
        import sys
        _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
        if _LIBS not in sys.path:
            sys.path.insert(0, _LIBS)
        from db_utils import engine
        from sqlalchemy import text
        with engine.connect() as c:
            n = c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        return (n or 0) > 0
    except Exception:
        return False


def _read_summary_from_rds(start=None, end=None) -> pd.DataFrame:
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path:
        sys.path.insert(0, _LIBS)
    from db_utils import engine
    where, params = [], {}
    if start is not None:
        where.append("date >= %(start)s"); params["start"] = pd.Timestamp(start).date()
    if end is not None:
        where.append("date <= %(end)s");   params["end"]   = pd.Timestamp(end).date()
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    df = pd.read_sql(
        f"SELECT date, board, "
        f"fin_prev_lots, fin_buy_lots, fin_sell_lots, fin_repay_lots, fin_today_lots, "
        f"short_prev_lots, short_buy_lots, short_sell_lots, short_repay_lots, short_today_lots, "
        f"fin_prev_ntd_k, fin_buy_ntd_k, fin_sell_ntd_k, fin_repay_ntd_k, fin_today_ntd_k "
        f"FROM tw_margin_summary {where_sql} ORDER BY date, board",
        engine, params=params,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _read_by_stock_from_rds(start=None, end=None) -> pd.DataFrame:
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path:
        sys.path.insert(0, _LIBS)
    from db_utils import engine
    where, params = [], {}
    if start is not None:
        where.append("date >= %(start)s"); params["start"] = pd.Timestamp(start).date()
    if end is not None:
        where.append("date <= %(end)s");   params["end"]   = pd.Timestamp(end).date()
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    df = pd.read_sql(
        f"SELECT date, board, stock_code, stock_name, "
        f"fin_prev_lots, fin_buy_lots, fin_sell_lots, fin_repay_lots, fin_today_lots, "
        f"short_prev_lots, short_buy_lots, short_sell_lots, short_repay_lots, short_today_lots, "
        f"offset_lots, note "
        f"FROM tw_margin_by_stock {where_sql} ORDER BY date, board, stock_code",
        engine, params=params,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["stock_code"] = df["stock_code"].astype(str)
    df["note"] = df["note"].fillna("").astype(str)
    return df


def _read_summary(margin_dir: Path, start=None, end=None) -> pd.DataFrame:
    """Whole-table cache only kicks in when no filters are set."""
    global _SUMMARY_CACHE, _LAST_SOURCE, _LAST_DIR
    src = "rds" if _rds_table_exists("tw_margin_summary") else "csv"
    if start is None and end is None:
        if _SUMMARY_CACHE is None or _LAST_SOURCE != src or _LAST_DIR != margin_dir:
            if src == "rds":
                _SUMMARY_CACHE = _read_summary_from_rds()
            else:
                _SUMMARY_CACHE = pd.read_csv(margin_dir / "tw_margin_summary.csv",
                                              parse_dates=["date"])
            _LAST_SOURCE = src
            _LAST_DIR = margin_dir
        return _SUMMARY_CACHE
    # Range query — never cache (small enough to just refetch)
    if src == "rds":
        return _read_summary_from_rds(start=start, end=end)
    df = pd.read_csv(margin_dir / "tw_margin_summary.csv", parse_dates=["date"])
    if start is not None: df = df[df["date"] >= pd.Timestamp(start)]
    if end   is not None: df = df[df["date"] <= pd.Timestamp(end)]
    return df


def _read_by_stock(margin_dir: Path, start=None, end=None) -> pd.DataFrame:
    global _STOCK_CACHE
    src = "rds" if _rds_table_exists("tw_margin_by_stock") else "csv"
    if start is None and end is None:
        if _STOCK_CACHE is None:
            if src == "rds":
                _STOCK_CACHE = _read_by_stock_from_rds()
            else:
                _STOCK_CACHE = pd.read_csv(
                    margin_dir / "tw_margin_by_stock.csv",
                    parse_dates=["date"],
                    dtype={"stock_code": str, "stock_name": str, "note": str},
                )
        return _STOCK_CACHE
    if src == "rds":
        return _read_by_stock_from_rds(start=start, end=end)
    df = pd.read_csv(margin_dir / "tw_margin_by_stock.csv",
                      parse_dates=["date"],
                      dtype={"stock_code": str, "stock_name": str, "note": str})
    if start is not None: df = df[df["date"] >= pd.Timestamp(start)]
    if end   is not None: df = df[df["date"] <= pd.Timestamp(end)]
    return df


def load_margin_summary(
    boards: Iterable[str] = ("TSE", "OTC"),
    start=None, end=None,
    margin_dir: str | Path = _DEFAULT_MARGIN_DIR,
) -> pd.DataFrame:
    """Daily summary rows across selected boards, sorted by date."""
    df = _read_summary(Path(margin_dir), start=start, end=end)
    return df[df["board"].isin(boards)].sort_values(["date", "board"]).reset_index(drop=True)


def load_margin_by_stock(
    boards: Iterable[str] = ("TSE", "OTC"),
    start=None, end=None,
    margin_dir: str | Path = _DEFAULT_MARGIN_DIR,
) -> pd.DataFrame:
    """Per-stock rows across selected boards, sorted by (date, board, stock)."""
    df = _read_by_stock(Path(margin_dir), start=start, end=end)
    return (df[df["board"].isin(boards)]
              .sort_values(["date", "board", "stock_code"])
              .reset_index(drop=True))


# ── 大盤融資維持率 loader ─────────────────────────────────────────────────────
def load_market_maintenance_ratio(
    trading_index: pd.DatetimeIndex | None = None,
    boards: Iterable[str] = ("TSE", "OTC"),
    start=None, end=None,
    margin_dir: str | Path = _DEFAULT_MARGIN_DIR,
) -> pd.Series:
    """
    Compute 大盤融資維持率 as a percentage time series.

    Parameters
    ----------
    trading_index : optional pd.DatetimeIndex
        If given, forward-fills onto this calendar (typical use — aligning to
        an MTX or TX trading calendar). NaN before the first available date.
        If None, returns the raw series on the union of dates present in the
        margin CSV.
    boards : subset of {"TSE", "OTC"}
        Which boards to include in the aggregate. Default: both (proper
        「大盤」 definition).
    margin_dir : path
        Directory holding ``tw_margin_summary.csv`` and
        ``tw_margin_by_stock.csv``. Defaults to
        ``mtx/tw_margin/`` next to this file.

    Returns
    -------
    pd.Series named "大盤融資維持率", index=date, values in %.
    """
    boards = tuple(b.upper() for b in boards)
    for b in boards:
        if b not in ("TSE", "OTC"):
            raise ValueError(f"unknown board {b!r}; must be TSE and/or OTC")

    # If trading_index is given but no explicit start/end, infer them so the
    # RDS pulls stay cheap.
    if trading_index is not None and start is None and end is None:
        ti = pd.DatetimeIndex(trading_index)
        start, end = ti.min(), ti.max()
    summary  = load_margin_summary(boards, start=start, end=end, margin_dir=margin_dir)
    by_stock = load_margin_by_stock(boards, start=start, end=end, margin_dir=margin_dir)

    # Aggregate margin debt across boards — direct from summary CSV, no join.
    # fin_today_ntd_k is in 仟元 → × 1000 to get NTD.
    debt = (summary.groupby("date")["fin_today_ntd_k"].sum() * 1000).astype(float)

    # Collateral value: per-stock close × per-stock 融資今日餘額 × 1000 shares/張.
    # Loaded per-board and summed.
    td_index = pd.DatetimeIndex(sorted(by_stock["date"].unique()))
    collateral_total = pd.Series(0.0, index=td_index)

    for b in boards:
        close_wide = _load_close_wide(td_index, b)          # date × ticker (int columns)
        if close_wide.empty:
            continue
        # Canonical string form on the price side ("1101", "50" from int 50).
        close_wide = close_wide.copy()
        close_wide.columns = close_wide.columns.map(str)

        # Wide (date × ticker) 融資今日餘額股數 for this board
        board_rows = by_stock[by_stock["board"] == b].copy()
        # tw_spot_pv only stores numeric-only tickers as integers, so ETFs
        # with letter suffixes (e.g. "00679B" bond ETFs) have no price row.
        # Canonicalise numeric codes to match ("0050" → "50"); leave the
        # rest as-is so the intersection cleanly drops them.
        board_rows["stock_code_key"] = board_rows["stock_code"].apply(
            lambda c: str(int(c)) if str(c).isdigit() else str(c))

        lots = board_rows.pivot_table(
            index="date", columns="stock_code_key",
            values="fin_today_lots", aggfunc="first")

        common = close_wide.columns.intersection(lots.columns)
        if len(common) == 0:
            continue

        # NTD: close(NTD/share) × lots(張) × 1000 shares/張
        board_collat = (close_wide[common] * lots[common] * 1000).sum(axis=1)
        collateral_total = collateral_total.add(board_collat, fill_value=0.0)

    # Ratio
    mr = (collateral_total / debt.reindex(collateral_total.index) * 100.0)
    mr = mr.rename("大盤融資維持率").sort_index()

    if trading_index is not None:
        mr = (mr.reindex(pd.DatetimeIndex(trading_index).sort_values(),
                          method="ffill"))
        mr = mr.reindex(trading_index)

    return mr


__all__ = [
    "load_margin_summary",
    "load_margin_by_stock",
    "load_market_maintenance_ratio",
]
