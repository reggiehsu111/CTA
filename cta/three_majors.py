"""
TAIFEX 三大法人-區分各期貨契約 loader.

Reads the daily institutional-flow data scraped by
`QuantResearch/tools/scrape_taifex_three_majors.py` (default location
`QuantResearch/data/taifex_three_majors.csv`) and exposes it through a few
small loaders that match the rest of the `cta` API.

CSV schema (one row per date × product × identity):

    date, product, identity,
    long_lots, long_amount, short_lots, short_amount, net_lots, net_amount,
    oi_long_lots, oi_long_amount, oi_short_lots, oi_short_amount,
    oi_net_lots, oi_net_amount

Where:
  * long_*, short_*, net_* = the day's trading volume
    (口數 / 契約金額)
  * oi_*  = end-of-day open interest (未平倉)

Identity values: 外資 (foreign), 投信 (investment trust), 自營商 (proprietary),
期貨合計 (grand total — only appears on the 期貨合計 / 期貨小計 product rows).

Typical use:

    import cta
    fi_oi_net = cta.load_three_majors(
        product="小型臺指期貨", identity="外資", metric="oi_net_lots",
    )                               # daily Series, date-indexed
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Default CSV path: repo-root/mtx/taifex_three_majors.csv
# (this file lives at repo-root/mtx/cta/three_majors.py)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "taifex_three_majors.csv"

_NUMERIC_COLS = [
    "long_lots", "long_amount",
    "short_lots", "short_amount",
    "net_lots", "net_amount",
    "oi_long_lots", "oi_long_amount",
    "oi_short_lots", "oi_short_amount",
    "oi_net_lots", "oi_net_amount",
]

# Symbol aliases — TAIFEX product codes & common English shortcuts → Chinese
# product name as it appears in the CSV. Keys are matched case-insensitively.
_PRODUCT_ALIASES: dict[str, str] = {
    # Taiwan index family
    "TXF":    "臺股期貨",     "TX":     "臺股期貨",     "TAIEX":  "臺股期貨",
    "MXF":    "小型臺指期貨", "MTX":    "小型臺指期貨", "MINI":   "小型臺指期貨",
    "TMF":    "微型臺指期貨", "MICRO":  "微型臺指期貨",
    # Sub-indices
    "TE":     "電子期貨",     "TEF":    "電子期貨",     "ELEC":   "電子期貨",
    "TF":     "金融期貨",     "TFF":    "金融期貨",     "FIN":    "金融期貨",
    "ZEF":    "小型電子期貨",
    "ZFF":    "小型金融期貨",
    "XIF":    "非金電期貨",   "NONFIN": "非金電期貨",
    "GTF":    "櫃買指數期貨",
    "SOX":    "半導體30期貨",
    "OTC200": "富櫃200期貨",
    # Sector / other domestic
    "ETF":    "ETF期貨",
    "STF":    "股票期貨",     "STOCK":  "股票期貨",
    "BTF":    "臺灣生技期貨", "BIO":    "臺灣生技期貨",
    "ESG":    "臺灣永續期貨",
    "SHF":    "航運期貨",     "SHIPPING": "航運期貨",
    # Foreign re-listings
    "SPF":    "美國標普500期貨",     "SP500":   "美國標普500期貨",
    "UDF":    "美國道瓊期貨",         "DJIA":    "美國道瓊期貨",
    "UNF":    "美國那斯達克100期貨",  "NASDAQ":  "美國那斯達克100期貨", "NDX": "美國那斯達克100期貨",
    "PHF":    "美國費城半導體期貨",   "PHLX":    "美國費城半導體期貨",
    "FUF":    "英國富時100期貨",      "FTSE":    "英國富時100期貨",
    "TJF":    "東證期貨",             "TOPIX":   "東證期貨",
    # Summary rows
    "TOTAL":  "期貨合計", "SUM": "期貨小計",
}

_IDENTITY_ALIASES: dict[str, str] = {
    "FOREIGN": "外資",  "FI": "外資",  "FOREIGN_INSTITUTIONS": "外資",
    "TRUST":   "投信",  "IT": "投信",  "INVESTMENT_TRUST": "投信",
    "PROP":    "自營商","DEALER": "自營商", "PROPRIETARY": "自營商",
    "TOTAL":   "期貨合計",
}


def _resolve_product(p: str, df: pd.DataFrame | None = None) -> str:
    """Map an alias / shortcut to the canonical Chinese product name."""
    key = str(p).strip().upper()
    if key in _PRODUCT_ALIASES:
        return _PRODUCT_ALIASES[key]
    # Already Chinese (or already canonical): try exact case-sensitive match
    if df is not None and p in df["product"].values:
        return p
    return p          # let the downstream KeyError happen with a clear message


def _resolve_identity(i: str) -> str:
    key = str(i).strip().upper()
    return _IDENTITY_ALIASES.get(key, i)

# Lazy cache so the CSV is only read once per process
_RAW_CACHE: pd.DataFrame | None = None
_RAW_CACHE_SOURCE: str | None = None


def _read_from_db() -> pd.DataFrame:
    """Query `tw_three_majors` into a DataFrame matching the CSV schema."""
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path: sys.path.insert(0, _LIBS)
    from db_utils import engine
    df = pd.read_sql(
        "SELECT date, product, identity, long_lots, long_amount, short_lots, short_amount, "
        "net_lots, net_amount, oi_long_lots, oi_long_amount, oi_short_lots, "
        "oi_short_amount, oi_net_lots, oi_net_amount "
        "FROM tw_three_majors ORDER BY date, product, identity",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["product"]  = df["product"].astype(str).str.strip()
    df["identity"] = df["identity"].astype(str).str.strip()
    return df


def _read_raw(csv_path: str | Path | None = None, use_db: bool = True) -> pd.DataFrame:
    """Load the raw long-format table, cached after first call.

    Source: DB (`tw_three_majors`) if ``use_db`` and no explicit ``csv_path``;
    otherwise the CSV.
    """
    global _RAW_CACHE, _RAW_CACHE_SOURCE
    source = "csv" if csv_path is not None else ("db" if use_db else "csv")
    if _RAW_CACHE is not None and csv_path is None and _RAW_CACHE_SOURCE == source:
        return _RAW_CACHE

    if source == "db":
        df = _read_from_db()
    else:
        path = Path(csv_path) if csv_path else _DEFAULT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"three-majors CSV not found at {path}. "
                "Scrape it first with "
                "QuantResearch/tools/scrape_taifex_three_majors.py."
            )
        df = pd.read_csv(path, parse_dates=["date"])
        df["product"]  = df["product"].astype(str).str.strip()
        df["identity"] = df["identity"].astype(str).str.strip()

    if csv_path is None:
        _RAW_CACHE = df
        _RAW_CACHE_SOURCE = source
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_three_majors(
    product:  str = "小型臺指期貨",
    identity: str = "外資",
    metric:   str = "oi_net_lots",
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """Return a single (product, identity, metric) daily series.

    Point-in-time (PIT) note
    ------------------------
    TAIFEX publishes the three-major-institutions report **after market
    close** of date D — typically around 15:00 TPE. So the row labeled
    ``date = D`` is information knowable at D's *close*, and the earliest
    a position can be taken using it is D+1's open.

    Two patterns are PIT-safe:

    * **Daily backtest** (`cta.load_asset(..., '1d')` + `cta.Simulate`) —
      `Simulate` applies `signal.shift(2)` internally, so signal[D] is acted
      on at D+1 open and earns PnL from D+1 → D+2 return. Use
      `pit_lag_days=0` (the default).
    * **Intraday backtest** (broadcasting to a 1m asset with
      `align_three_majors_to_asset`) — the daily value gets placed at every
      minute of date D, including *before* the 15:00 publication. Pass
      `pit_lag_days=1` so the value at date-index D is actually D-1's
      published number. The minute bars of D will then carry D-1's OI,
      which is genuinely public.

    Parameters
    ----------
    product : str
        Chinese product name. List with `available_products()`.
        Common: '臺股期貨', '小型臺指期貨', '電子期貨', '金融期貨',
                '股票期貨', '期貨合計'.
    identity : str
        '外資' | '投信' | '自營商' | '期貨合計'.
    metric : str
        One of the 12 numeric columns (e.g. 'net_lots' for the day's trading
        net, 'oi_net_lots' for end-of-day net open interest). List with
        `available_metrics()`.
    pit_lag_days : int, default 1
        Shift the series forward by N days so each date label has the data
        that was actually knowable at that date's open. Set 0 only for
        descriptive use.
    csv_path : str | Path | None
        Override the default file location.

    Returns
    -------
    pd.Series indexed by date (Timestamp at midnight), sorted ascending.
    Name is f"{product}_{identity}_{metric}" (with a `_pit` suffix when
    `pit_lag_days > 0`).

    Examples
    --------
    >>> fi_oi = cta.load_three_majors("小型臺指期貨", "外資", "oi_net_lots")
    >>> # The value at index 2026-06-26 is the EOD OI on 2026-06-25 — PIT-safe.
    """
    if metric not in _NUMERIC_COLS:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from: {_NUMERIC_COLS}"
        )
    df = _read_raw(csv_path, use_db=use_db)
    product  = _resolve_product(product, df)
    identity = _resolve_identity(identity)
    sub = df[(df["product"] == product) & (df["identity"] == identity)]
    if sub.empty:
        raise KeyError(
            f"No rows for product='{product}', identity='{identity}'. "
            f"Try `cta.show_three_majors_catalog()` for the full list."
        )
    name = f"{product}_{identity}_{metric}"
    if pit_lag_days > 0:
        name = name + "_pit"
    ser = (sub.set_index("date")[metric]
              .sort_index()
              .rename(name))
    if pit_lag_days > 0:
        ser = ser.shift(pit_lag_days)
    return ser


def load_three_majors_wide(
    product:  str = "小型臺指期貨",
    metric:   str = "oi_net_lots",
    identities: list[str] | None = None,
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.DataFrame:
    """All identities for one (product, metric) as a wide DataFrame.

    Columns are the identities ('外資', '投信', '自營商'); index is date.

    Examples
    --------
    >>> oi_by_identity = cta.load_three_majors_wide(
    ...     "小型臺指期貨", "oi_net_lots",
    ... )
    >>> oi_by_identity.tail(3)
                外資   投信  自營商
    date
    2026-06-24 -8421  120  3502
    ...
    """
    if metric not in _NUMERIC_COLS:
        raise ValueError(f"Unknown metric '{metric}'.")
    df = _read_raw(csv_path, use_db=use_db)
    product = _resolve_product(product, df)
    sub = df[df["product"] == product].copy()
    if identities is not None:
        identities = [_resolve_identity(i) for i in identities]
        sub = sub[sub["identity"].isin(identities)]
    if sub.empty:
        raise KeyError(f"No rows for product='{product}'.")
    wide = (sub.pivot_table(index="date", columns="identity",
                            values=metric, aggfunc="first")
              .sort_index())
    wide.index.name = "date"
    if pit_lag_days > 0:
        wide = wide.shift(pit_lag_days)
    return wide


def align_three_majors_to_asset(
    series: pd.Series,
    asset,
    fill_value: float | None = None,
) -> pd.Series:
    """Reindex a daily three-majors series to an intraday asset's bar index.

    Three-majors numbers update once per day (end-of-day OI / net trade
    figures). To use them as a factor on a 1-minute asset, forward-fill the
    daily value across every intraday bar of that calendar date. We snap by
    *calendar date* (not by trading session) so the day's value is in effect
    from 00:00 through 23:59 of that date — typical for a published OI series.

    Parameters
    ----------
    series : pd.Series
        Output of `load_three_majors(...)` (date-indexed).
    asset : cta.BaseAsset
        Active intraday asset whose index is the target.
    fill_value : float | None
        If given, bars before the first available date use this value (default
        leaves them NaN, which acts as 'no signal' downstream).

    Returns
    -------
    pd.Series on `asset.index`, name preserved.
    """
    daily = series.sort_index().copy()
    daily.index = pd.to_datetime(daily.index).normalize()
    # Build a per-bar-date lookup, then map
    bar_dates = pd.to_datetime(asset.index).normalize()
    out = pd.Series(daily.reindex(bar_dates).values,
                    index=asset.index, name=series.name)
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Introspection helpers
# ─────────────────────────────────────────────────────────────────────────────

def available_products(csv_path: str | Path | None = None, use_db: bool = True) -> list[str]:
    """List every product name present in the CSV."""
    return sorted(_read_raw(csv_path, use_db=use_db)["product"].unique())


def available_identities(csv_path: str | Path | None = None, use_db: bool = True) -> list[str]:
    """List the identity values (外資 / 投信 / 自營商 / 期貨合計)."""
    return sorted(_read_raw(csv_path, use_db=use_db)["identity"].unique())


def available_metrics() -> list[str]:
    """List the 12 numeric metric column names."""
    return list(_NUMERIC_COLS)


def show_three_majors_catalog(csv_path: str | Path | None = None, use_db: bool = True) -> pd.DataFrame:
    """Print + return the product catalog: each Chinese name with its aliases,
    so you can use either `'TXF'` or `'臺股期貨'` interchangeably.
    """
    df = _read_raw(csv_path, use_db=use_db)
    products = sorted(df["product"].unique())
    # Build reverse map: chinese_name -> list of aliases
    rev: dict[str, list[str]] = {p: [] for p in products}
    for alias, canonical in _PRODUCT_ALIASES.items():
        if canonical in rev:
            rev[canonical].append(alias)
    rows = [{"product": p, "aliases": ", ".join(sorted(rev[p])) or "(none)"}
            for p in products]
    cat = pd.DataFrame(rows)
    print(f"{len(cat)} products available  (csv: {_DEFAULT_CSV.name})")
    print(cat.to_string(index=False))
    print(f"\nidentities:  {available_identities(csv_path, use_db=use_db)}")
    print(f"metrics  :  {_NUMERIC_COLS}")
    return cat


__all__ = [
    "load_three_majors",
    "load_three_majors_wide",
    "align_three_majors_to_asset",
    "available_products",
    "available_identities",
    "available_metrics",
    "show_three_majors_catalog",
]
