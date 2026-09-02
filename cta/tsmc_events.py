"""
TSMC investor-relations event classifier.

Parses the TWSE-format `tsmc_ea.csv` and categorizes each row into:

  * `quarterly_call`   — the official 4x/year 法說會 earnings announcement.
                         Signature: description contains
                         "公布(佈)?[本公司]?YYYY年第X季...財務報告"
                         (or, for legacy rows with empty descriptions,
                         time == 14:00 at TSMC's classic Taipei venue).
  * `investor_day`     — TSMC-hosted capital-markets day (Investor Day).
                         Rare, in the US, ~biennial (2018, 2022, 2025, 2026).
  * `broker_conference`— company invited to a sell-side broker/bank
                         conference (UBS, Morgan Stanley, JPMorgan, etc.).
                         Signature: description contains "受邀參加",
                         "本公司將參加", "本公司將於...參加" or "本公司將出席".
                         Also emits `broker` (the counterparty).
  * `other`            — unknown / cannot be classified.

Usage
-----
    >>> import cta
    >>> events = cta.load_tsmc_events()      # DataFrame
    >>> qc = cta.load_tsmc_event_dates('quarterly_call', trading_index)
    >>> id = cta.load_tsmc_event_dates('investor_day',   trading_index)
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "tsmc_ea.csv"

_BROKERS: dict[str, str] = {
    "摩根大通": "JPMorgan", "摩根史坦利": "MorganStanley", "摩根士丹利": "MorganStanley",
    "高盛": "Goldman", "花旗": "Citi", "美銀": "BofA", "瑞銀": "UBS",
    "德意志": "DeutscheBank", "野村": "Nomura", "巴克萊": "Barclays",
    "美林": "MerrillLynch", "瑞信": "CreditSuisse", "麥格理": "Macquarie",
    "里昂": "CLSA", "法國興業": "SocGen", "法巴": "BNPP",
    "中金": "CICC", "凱基": "KGI", "元大": "Yuanta", "永豐": "SinoPac",
    "國泰": "Cathay", "群益": "CapitalSec",
}

# quarter char → int
_QMAP = {"一": 1, "二": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}

# 公布(佈)?[本公司]?...YYYY年第X季....財務報告
_RE_QUARTERLY = re.compile(r"公[布佈](本公司)?.*(\d{4})年第([一二三四1234]).?季.*財務報告")
_RE_BROKER    = re.compile(r"(受邀參加|本公司將參加|本公司將於.*參加|本公司將出席)")


def _roc_to_ts(date_field: str) -> pd.Timestamp | None:
    """Convert ROC date string ('YYYY/MM/DD', possibly a range) → Timestamp."""
    if not date_field:
        return None
    ds = date_field.strip().split()[0]        # take start of a range
    parts = ds.split("/")
    if len(parts) != 3:
        return None
    try:
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        return pd.Timestamp(year=y + 1911, month=m, day=d)
    except (ValueError, OverflowError):
        return None


def _classify(desc: str, time_field: str) -> tuple[str, str | None, int | None]:
    """Return (category, extra, quarter). `extra` = year for quarterly, broker for
    broker_conference; `quarter` = 1..4 for quarterly."""
    d = desc.strip()
    t = time_field.strip()

    m = _RE_QUARTERLY.search(d)
    if m:
        return "quarterly_call", m.group(2), _QMAP.get(m.group(3))

    if "Investor Day" in d or "投資者日" in d:
        return "investor_day", None, None

    if _RE_BROKER.search(d):
        broker = next((v for k, v in _BROKERS.items() if k in d), "Other")
        return "broker_conference", broker, None

    # Legacy pre-2010 rows: empty description, 14:00 → assumed quarterly call.
    if not d and t.startswith("14:"):
        return "quarterly_call", None, None

    return "other", None, None


def _load_from_db() -> pd.DataFrame:
    """Query `tsmc_ir_events` into the same DataFrame shape as the CSV parse."""
    import sys
    _LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
    if _LIBS not in sys.path: sys.path.insert(0, _LIBS)
    from db_utils import engine
    df = pd.read_sql(
        "SELECT date, time, location, category, "
        "NULLIF(broker, '') AS broker, quarter, fiscal_year, "
        "description AS desc FROM tsmc_ir_events "
        "ORDER BY date, category, broker",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])
    for c in ("time", "location", "desc"):
        df[c] = df[c].astype("object").where(df[c].notna(), None)
    df["quarter"]     = pd.to_numeric(df["quarter"], errors="coerce").astype("Int64")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    return df.sort_values("date").reset_index(drop=True)


def load_tsmc_events(csv_path: str | Path | None = None,
                      use_db: bool = True) -> pd.DataFrame:
    """Return one row per (date, category, broker) event.

    Source
    ------
    * `use_db=True` (default) + no explicit `csv_path` → query DB table
      `tsmc_ir_events` (already-classified).
    * Otherwise → parse the raw Big5-encoded CSV and classify with the
      module's regex rules.

    Columns
    -------
    date            : pandas Timestamp (calendar day, not snapped to trading day)
    time            : original time field (str)
    location        : original location field (str, truncated to 60 chars)
    category        : 'quarterly_call' | 'investor_day' | 'broker_conference' | 'other'
    broker          : broker name for broker_conference (else None)
    quarter         : 1..4 for quarterly_call (else None)
    fiscal_year     : year the quarterly report covers (else None)
    desc            : original description (truncated to 200 chars)
    """
    if csv_path is None and use_db:
        return _load_from_db()

    path = Path(csv_path) if csv_path else _DEFAULT_CSV
    if not path.exists():
        raise FileNotFoundError(f"TSMC EA CSV not found: {path}")

    records = []
    with open(path, encoding="big5", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return pd.DataFrame()
        for row in reader:
            if len(row) < 6 or not row[2].strip():
                continue
            ts = _roc_to_ts(row[2])
            if ts is None:
                continue
            desc = row[5]
            cat, extra, q = _classify(desc, row[3])
            records.append({
                "date": ts,
                "time": row[3].strip(),
                "location": row[4].strip()[:60],
                "category": cat,
                "broker": extra if cat == "broker_conference" else None,
                "quarter": q,
                "fiscal_year": int(extra) if (cat == "quarterly_call" and extra) else None,
                "desc": desc.strip()[:200],
            })
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    # Some old rows repeat one event across multiple PDF filenames — dedup on
    # (date, category, broker).
    df = df.drop_duplicates(subset=["date", "category", "broker"], keep="first")
    return df.reset_index(drop=True)


def load_tsmc_event_dates(
    category: str,
    trading_index: pd.DatetimeIndex,
    csv_path: str | Path | None = None,
    midnight_to_next_day: bool = True,
    use_db: bool = True,
) -> pd.DatetimeIndex:
    """Return trading-day-aligned event dates for one category.

    * `midnight_to_next_day`: shift 00:00 entries forward one day (see
      `load_tsmc_ea_dates` for rationale — overnight events land next session).
    * Each remaining date is snapped forward to the next trading day.
    """
    events = load_tsmc_events(csv_path, use_db=use_db)
    if category not in set(events["category"]):
        return pd.DatetimeIndex([])
    sub = events[events["category"] == category].copy()

    if midnight_to_next_day:
        mask = sub["time"].str.startswith("00:00").fillna(False)
        sub.loc[mask, "date"] = sub.loc[mask, "date"] + pd.Timedelta(days=1)

    if trading_index is None or len(trading_index) == 0:
        return pd.DatetimeIndex(sorted(sub["date"].unique()))

    snapped: list[pd.Timestamp] = []
    last_pos = len(trading_index)
    for ev in sorted(sub["date"].unique()):
        pos = trading_index.searchsorted(ev, side="left")
        if pos < last_pos:
            snapped.append(trading_index[pos])
    return pd.DatetimeIndex(sorted(set(snapped)))


__all__ = ["load_tsmc_events", "load_tsmc_event_dates"]
