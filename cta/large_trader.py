"""
TAIFEX 大額交易人未沖銷部位 loader (large-trader open interest structure).

Reads the daily large-trader concentration data scraped by
`QuantResearch/tools/scrape_taifex_large_trader.py` (default location
`mtx/taifex_large_trader.csv`) and exposes it through loaders that match
the rest of the `cta` API.

CSV schema
----------
    date, commodity, commodity_name, expiry, trader_type,
    top5_buy, top5_sell, top10_buy, top10_sell, total_oi

Key fields:
  * `commodity` — TAIFEX symbol, e.g. `TX`, `MTX`, `TMF`, `TE`, `TF`, `STF`.
  * `expiry`:
        - `999999` — aggregate across ALL expiries (usually what you want)
        - `666666` — aggregate for weekly contracts
        - `YYYYMM` — specific monthly expiry
  * `trader_type`:
        - `0` — all top-N traders (including non-institutional)
        - `1` — the 特定法人 (specific institutional) subset — usually
                foreign banks / prop shops / big insurers among the top-N.
  * `top5_buy` / `top5_sell` — number of contracts held long / short by
                the top-5 traders on that side.
  * `top10_buy` / `top10_sell` — same for top-10.
  * `total_oi` — total market open interest (contracts).

Derived metrics available through `metric=` on the loader:

    'top5_buy', 'top5_sell', 'top10_buy', 'top10_sell', 'total_oi'
                                                    (raw columns)

    'top5_net'         = top5_buy - top5_sell        (top-5 net position)
    'top10_net'        = top10_buy - top10_sell
    'top5_gross'       = top5_buy + top5_sell        (both sides combined)
    'top10_gross'      = top10_buy + top10_sell

    'top5_buy_pct'     = top5_buy    / total_oi      (share of OI)
    'top5_sell_pct'    = top5_sell   / total_oi
    'top10_buy_pct'    = top10_buy   / total_oi
    'top10_sell_pct'   = top10_sell  / total_oi
    'top5_net_pct'     = (top5_buy - top5_sell)  / total_oi
    'top10_net_pct'    = (top10_buy - top10_sell) / total_oi
    'top5_concentration'  = (top5_buy + top5_sell)  / (2 * total_oi)
    'top10_concentration' = (top10_buy + top10_sell) / (2 * total_oi)

PIT (point-in-time) note
------------------------
TAIFEX publishes this report **after each day's close** — the market closes
at 13:45 and the file is released ~15:00 TPE. So the row labelled ``date=D``
is available around 15:00 on D, well before D+1's open. `cta.Simulate` uses
`signal.shift(2)` internally (signal[D] → PnL[D+1..D+2] return), which is
PIT-safe with 22+ hours of margin.

Broadcasting a daily large-trader signal onto 1-minute intraday bars would
place D's value at bars starting 08:46 D, which is BEFORE the 15:00 D
publish. Pass `pit_lag_days=1` in that case so the 1m bars of D carry D-1's
published number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "taifex_large_trader.csv"

_RAW_METRICS = ("top5_buy", "top5_sell", "top10_buy", "top10_sell", "total_oi")
_DERIVED_METRICS = (
    "top5_net", "top10_net", "top5_gross", "top10_gross",
    "top5_buy_pct", "top5_sell_pct", "top10_buy_pct", "top10_sell_pct",
    "top5_net_pct", "top10_net_pct",
    "top5_concentration", "top10_concentration",
)
_ALL_METRICS = _RAW_METRICS + _DERIVED_METRICS

# Commodity code aliases — the large-trader report only publishes ROLLED-UP
# aggregate codes:
#   TX  = 臺股期貨 (TX + MTX/4 + TMF/20 in TX-equivalent contracts)
#   TE  = 電子期貨 (TE + ZEF/8)
#   TF  = 金融期貨 (TF + ZFF/4)
# So we route mini/micro/small aliases to their parent aggregate.
_COMMODITY_ALIASES: dict[str, str] = {
    # Taiwan index family — all resolve to the TX consolidated report
    "TXF":    "TX",   "TX":     "TX",   "TAIEX": "TX",
    "MXF":    "TX",   "MTX":    "TX",   "MINI":  "TX",     # rolled into TX
    "TMF":    "TX",   "MICRO":  "TX",                        # rolled into TX
    # Electronics — TE report includes ZEF/8
    "TE":     "TE",   "TEF":    "TE",   "ELEC":  "TE",
    "ZEF":    "TE",                                          # rolled into TE
    # Financials — TF report includes ZFF/4
    "TF":     "TF",   "TFF":    "TF",   "FIN":   "TF",
    "ZFF":    "TF",                                          # rolled into TF
    # Other
    "GTF":    "GTF",
    # Single-name stock futures — a few common ones by ticker
    "CD":     "CD",   "TSMC":   "CD",   "2330":  "CD",     # 台積電期貨
    "QF":     "QF",                                          # 小型台積電期貨
}

_TRADER_ALIASES: dict[int | str, int] = {
    0: 0, "0": 0, "all": 0, "ALL": 0, "TOTAL": 0,
    1: 1, "1": 1, "special": 1, "SPECIAL": 1, "special_legal": 1,
    "特定法人": 1, "institutional": 1, "INSTITUTIONAL": 1,
}

_RAW_CACHE: pd.DataFrame | None = None
_RAW_CACHE_SOURCE: str | None = None   # 'db' or 'csv'


def _read_from_db() -> pd.DataFrame:
    """Query full `tw_large_trader` table into a DataFrame in the exact CSV format."""
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path: sys.path.insert(0, _LIBS)
    from db_utils import engine
    df = pd.read_sql(
        "SELECT date, commodity, commodity_name, expiry, trader_type, "
        "top5_buy, top5_sell, top10_buy, top10_sell, total_oi "
        "FROM tw_large_trader ORDER BY date, commodity, expiry, trader_type",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])
    for c in ("commodity", "commodity_name", "expiry"):
        df[c] = df[c].astype(str).str.strip()
    # CSV numeric cols come back as float (may contain NaN); DB returns int / None.
    # Match CSV dtype so downstream code that expects float doesn't break.
    for c in ("trader_type", "top5_buy", "top5_sell", "top10_buy", "top10_sell", "total_oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df


def _read_raw(csv_path: str | Path | None = None, use_db: bool = True) -> pd.DataFrame:
    """Load the raw long-format table, cached after first call.

    Data source order:
      1. If ``csv_path`` explicitly given → read that CSV
      2. Elif ``use_db`` (default) → query `tw_large_trader` table
      3. Else → fall back to the default CSV path
    """
    global _RAW_CACHE, _RAW_CACHE_SOURCE
    if csv_path is not None:
        source = "csv"
    elif use_db:
        source = "db"
    else:
        source = "csv"

    if _RAW_CACHE is not None and csv_path is None and _RAW_CACHE_SOURCE == source:
        return _RAW_CACHE

    if source == "db":
        df = _read_from_db()
    else:
        path = Path(csv_path) if csv_path else _DEFAULT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"large-trader CSV not found at {path}. "
                "Scrape it first with "
                "QuantResearch/tools/scrape_taifex_large_trader.py."
            )
        df = pd.read_csv(
            path,
            dtype={"expiry": str, "commodity": str, "commodity_name": str},
            parse_dates=["date"],
        )
        for c in ("commodity", "commodity_name", "expiry"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

    if csv_path is None:
        _RAW_CACHE = df
        _RAW_CACHE_SOURCE = source
    return df


def _resolve_commodity(c: str) -> str:
    """Map alias / shortcut to canonical TAIFEX commodity code."""
    key = str(c).strip().upper()
    if key in _COMMODITY_ALIASES:
        return _COMMODITY_ALIASES[key]
    return key


def _resolve_trader_type(t) -> int:
    if t in _TRADER_ALIASES:
        return _TRADER_ALIASES[t]
    raise ValueError(
        f"Unknown trader_type {t!r}. Use 0/'all' or 1/'special'/'特定法人'."
    )


def _compute_metric(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return the requested metric as a Series, computing derived ones on the fly."""
    if metric in _RAW_METRICS:
        return df[metric].astype(float)
    tot = df["total_oi"].astype(float).replace(0, np.nan)
    b5, s5 = df["top5_buy"].astype(float), df["top5_sell"].astype(float)
    b10, s10 = df["top10_buy"].astype(float), df["top10_sell"].astype(float)
    if metric == "top5_net":            return b5 - s5
    if metric == "top10_net":           return b10 - s10
    if metric == "top5_gross":          return b5 + s5
    if metric == "top10_gross":         return b10 + s10
    if metric == "top5_buy_pct":        return b5 / tot
    if metric == "top5_sell_pct":       return s5 / tot
    if metric == "top10_buy_pct":       return b10 / tot
    if metric == "top10_sell_pct":      return s10 / tot
    if metric == "top5_net_pct":        return (b5 - s5) / tot
    if metric == "top10_net_pct":       return (b10 - s10) / tot
    if metric == "top5_concentration":  return (b5 + s5) / (2 * tot)
    if metric == "top10_concentration": return (b10 + s10) / (2 * tot)
    raise ValueError(f"Unknown metric {metric!r}. Choose from {_ALL_METRICS}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_large_trader(
    commodity: str = "TX",
    metric:    str = "top10_net_pct",
    expiry:    str = "all",          # 'all' → 999999; 'weekly' → 666666; or YYYYMM
    trader_type: int | str = "all",   # 0/'all' or 1/'special'
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.Series:
    """Return a daily large-trader time series for one (commodity, expiry, trader_type, metric).

    Parameters
    ----------
    commodity : str
        TAIFEX code (case-insensitive alias). Common:
        'TX' / 'TXF' (臺股期貨), 'MTX' / 'MXF' (小型臺指), 'TMF' (微型),
        'TE' (電子), 'TF' (金融), 'STF' (股票期貨), etc.
    metric : str
        One of the raw or derived metrics — see module docstring.
        Default 'top10_net_pct' = (top10_buy - top10_sell) / total_oi,
        the classic "how heavily is the top of the book leaning long?"
        indicator, safely normalised so it doesn't drift with market growth.
    expiry : str
        'all'    → 999999 (aggregate across all expiries — usually what you want)
        'weekly' → 666666 (aggregate for weekly contracts, when they exist)
        'YYYYMM' → specific monthly expiry
    trader_type : int | str
        0 / 'all'      — top-N traders overall
        1 / 'special'  — 特定法人 subset (foreign banks, prop shops, big insurers)
    pit_lag_days : int
        Shift returned series forward by N days. 0 (default) is safe for
        daily backtests with cta.Simulate (uses shift(2) internally).
        Use 1 when broadcasting to 1-minute intraday bars.
    csv_path : str | Path | None
        Override default CSV location.

    Returns
    -------
    pd.Series indexed by date, sorted ascending.

    Examples
    --------
    >>> tx_top10_net = cta.load_large_trader("TX", "top10_net_pct")
    >>> # % of OI held net-long by the top 10 traders, all expiries, all trader types
    >>>
    >>> # Only the institutional subset:
    >>> tx_top10_net_inst = cta.load_large_trader("TX", "top10_net_pct", trader_type="special")
    >>>
    >>> # Front-month only:
    >>> tx_top5_front = cta.load_large_trader("TX", "top5_net", expiry="202401")
    """
    if metric not in _ALL_METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {_ALL_METRICS}")

    df = _read_raw(csv_path, use_db=use_db)
    commodity = _resolve_commodity(commodity)
    tt = _resolve_trader_type(trader_type)

    if expiry == "all":       expiry_code = "999999"
    elif expiry == "weekly":  expiry_code = "666666"
    else:                     expiry_code = str(expiry).strip()

    sub = df[
        (df["commodity"] == commodity)
        & (df["expiry"] == expiry_code)
        & (df["trader_type"] == tt)
    ].copy()
    if sub.empty:
        raise KeyError(
            f"No rows for commodity={commodity!r}, expiry={expiry_code!r}, "
            f"trader_type={tt}. Try cta.show_large_trader_catalog()."
        )
    sub = sub.set_index("date").sort_index()
    ser = _compute_metric(sub, metric).rename(
        f"{commodity}_{expiry_code}_tt{tt}_{metric}"
    )
    if pit_lag_days > 0:
        ser = ser.shift(pit_lag_days)
    return ser


def load_large_trader_wide(
    commodity: str = "TX",
    metric:    str = "top10_net_pct",
    expiry:    str = "all",
    trader_types: list = None,
    pit_lag_days: int = 0,
    csv_path: str | Path | None = None,
    use_db: bool = True,
) -> pd.DataFrame:
    """Same as `load_large_trader` but returns a wide DataFrame with one column
    per trader_type (0 = all, 1 = special).
    """
    if trader_types is None:
        trader_types = [0, 1]
    out = {}
    for tt in trader_types:
        try:
            ser = load_large_trader(commodity, metric, expiry, tt,
                                     pit_lag_days=pit_lag_days,
                                     csv_path=csv_path,
                                     use_db=use_db)
            label = "all" if _resolve_trader_type(tt) == 0 else "special"
            out[label] = ser
        except KeyError:
            continue
    if not out:
        raise KeyError(f"No rows for commodity={commodity!r} at all")
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# Introspection helpers
# ─────────────────────────────────────────────────────────────────────────────

def available_large_trader_commodities(csv_path=None, use_db: bool = True) -> list[str]:
    """Every commodity code present in the source, sorted."""
    return sorted(_read_raw(csv_path, use_db=use_db)["commodity"].unique().tolist())


def available_large_trader_metrics() -> list[str]:
    """Every metric (raw + derived) supported by the loader."""
    return list(_ALL_METRICS)


def show_large_trader_catalog(csv_path=None, use_db: bool = True) -> pd.DataFrame:
    """Print + return a small catalog: each commodity code with its Chinese name
    and typical row count in the source."""
    df = _read_raw(csv_path, use_db=use_db)
    catalog = (df.groupby("commodity")
                  .agg(name=("commodity_name", "first"),
                       n_rows=("date", "size"),
                       first_date=("date", "min"),
                       last_date=("date", "max"))
                  .reset_index()
                  .sort_values("n_rows", ascending=False))
    print(f"{len(catalog)} commodities in the CSV")
    print(catalog.head(30).to_string(index=False))
    print(f"\nsupported metrics: {list(_ALL_METRICS)}")
    print(f"trader_type: 0/'all' (all top-N), 1/'special' (特定法人 subset)")
    return catalog


__all__ = [
    "load_large_trader",
    "load_large_trader_wide",
    "available_large_trader_commodities",
    "available_large_trader_metrics",
    "show_large_trader_catalog",
]
