"""
Download historical daily futures data from TAIFEX.

Uses the annual zip download endpoint (down_type=2), which contains all
contracts for that year. Filters by the requested commodity codes.

Usage
-----
    # Single contract (annual zip mode — completed years only)
    python download_taifex_history.py --commodity MTX

    # Multiple contracts in one pass (one zip download per year)
    python download_taifex_history.py --commodity TX MTX TMF SOF

    # All index / global futures (flag=0 from the TAIFEX API)
    python download_taifex_history.py --all-index

    # Every contract including stock futures (~618 codes)
    python download_taifex_history.py --all

    # List all available contract codes
    python download_taifex_history.py --list

    # Custom output directory
    python download_taifex_history.py --commodity TX --output-dir ./data

    # Current-year YTD: date-range mode (down_type=1, works for any range)
    python download_taifex_history.py --commodity MTX \
        --start-date 2026-01-01 --end-date 2026-05-16

    # Append YTD rows into the existing MTX_history.csv (dedup by date+contract+expiry+session)
    python download_taifex_history.py --commodity MTX \
        --start-date 2026-01-01 --append --output-dir history_data

Run with the project virtualenv:
    /Users/hsureggie/.pyenv/versions/3.14.0/envs/gary-research/bin/python3 download_taifex_history.py
"""

import argparse
import csv
import datetime
import io
import time
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
API_URL  = "https://www.taifex.com.tw/cht/3/getFutcontractDl"
HEADERS  = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://www.taifex.com.tw/cht/3/dlFutDailyMarketView",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Stable positional columns across all historical CSV formats (16-20 cols)
COLUMNS = [
    "交易日期", "契約", "到期月份(週別)", "開盤價", "最高價", "最低價",
    "收盤價", "漲跌價", "漲跌%", "成交量", "結算價", "未沖銷契約數",
    "最後最佳買價", "最後最佳賣價", "歷史最高價", "歷史最低價",
    "是否因訊息面暫停交易", "交易時段", "價差對單式委託成交量",
]

FIRST_YEAR = 1998   # earliest year available on TAIFEX

# Index / global futures visible in the TAIFEX daily market download dropdown.
# Used as a fallback when the API requires a browser session to respond correctly.
INDEX_FUTURES = {
    "BRF", "BTF", "E4F", "F1F", "G2F", "GDF", "GTF", "M1F",
    "MTX", "RHF", "RTF", "SHF", "SOF", "SPF", "SXF", "TE",
    "TF",  "TGF", "TJF", "TMF", "TX",  "UDF", "UNF", "XAF",
    "XBF", "XEF", "XIF", "XJF", "ZEF", "ZFF",
}


# -----------------------------------------------------------------------------
# TAIFEX API helpers
# -----------------------------------------------------------------------------

def fetch_contract_list(session: requests.Session) -> list[dict] | None:
    """
    Return all futures contracts from the TAIFEX commodity API.
    Returns None if the API requires a browser session (returns 'error').
    """
    resp = session.get(API_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("commodityList", [])
    if not isinstance(data, list):
        return None   # API returned {"commodityList": "error"}
    return data


def download_year(session: requests.Session, year: int, retries: int = 3) -> bytes:
    """POST to the annual zip endpoint; return raw zip bytes."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(
                BASE_URL,
                data={"down_type": "2", "his_year": str(year)},
                headers=HEADERS,
                timeout=60,
            )
            resp.raise_for_status()
            if resp.headers.get("Content-Type", "").startswith("text/html"):
                raise ValueError(f"Got HTML instead of zip for year {year}")
            return resp.content
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"    Attempt {attempt} failed ({exc}). Retrying in {wait}s...")
            time.sleep(wait)


def extract_rows(zip_bytes: bytes, commodity_ids: set | None) -> dict:
    """
    Unzip in-memory, parse Big5 CSV, return rows grouped by commodity code.

    Parameters
    ----------
    zip_bytes     : raw bytes of the downloaded zip
    commodity_ids : set of codes to keep, or None to keep everything
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names    = zf.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), names[0])
        raw      = zf.read(csv_name)

    text   = raw.decode("big5", errors="replace")
    result = {}

    for i, row in enumerate(csv.reader(io.StringIO(text))):
        if i == 0 or len(row) < 2:
            continue
        code = row[1].strip()
        if commodity_ids is None or code in commodity_ids:
            result.setdefault(code, []).append(row)

    return result


# -----------------------------------------------------------------------------
# Date-range mode (down_type=1) — works for current-year / YTD data
# -----------------------------------------------------------------------------

def parse_cli_date(s: str) -> datetime.date:
    """Accept YYYY-MM-DD or YYYY/MM/DD."""
    return datetime.date.fromisoformat(s.replace("/", "-"))


DAILY_VIEW_URL = "https://www.taifex.com.tw/cht/3/futDailyMarketView"


def prime_session(session: requests.Session) -> None:
    """GET the daily-view page once to establish session cookies."""
    session.get(DAILY_VIEW_URL,
                headers={"User-Agent": HEADERS["User-Agent"]},
                timeout=30)


def download_range_chunk(
    session: requests.Session,
    commodity_id: str,
    start_date: datetime.date,
    end_date: datetime.date,
    retries: int = 3,
) -> bytes:
    """
    POST to the date-range endpoint for ONE chunk (≤ 1 month wide).

    TAIFEX caps `down_type=1` requests at one calendar month. Use
    `download_range()` for arbitrary ranges — it chunks internally.

    Note: TAIFEX mislabels the CSV response as `text/html;charset=MS950`,
    so we identify success/failure by the body content (CSV header vs HTML).
    """
    for attempt in range(1, retries + 1):
        try:
            resp = session.post(
                BASE_URL,
                data={
                    "down_type":      "1",
                    "commodity_id":   commodity_id,
                    "commodity_id2":  "",
                    "queryStartDate": start_date.strftime("%Y/%m/%d"),
                    "queryEndDate":   end_date.strftime("%Y/%m/%d"),
                    "MarketCode":     "0",
                },
                headers={**HEADERS, "Referer": DAILY_VIEW_URL},
                timeout=60,
            )
            resp.raise_for_status()

            # TAIFEX returns the CSV with content-type "text/html;charset=MS950",
            # so we detect failure by sniffing the body for an HTML/JS error page.
            body_head = resp.content[:200].decode("big5", errors="replace").lstrip()
            looks_like_html = body_head.startswith("<") or "alert(" in body_head
            if looks_like_html:
                raise ValueError(
                    f"TAIFEX returned an error page for {commodity_id} "
                    f"{start_date}–{end_date}: {body_head[:120]!r}"
                )
            return resp.content
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 5 * attempt
            print(f"    Attempt {attempt} failed ({exc}). Retrying in {wait}s...")
            time.sleep(wait)


def _iter_month_chunks(start: datetime.date, end: datetime.date):
    """
    Yield (chunk_start, chunk_end) pairs covering [start, end], each
    contained within a single calendar month.

    TAIFEX rejects requests where `start_date + 1 month < end_date`
    (using JS setMonth semantics, so Jan 31 + 1 month = Feb 28).
    Chunking on calendar-month boundaries sidesteps every edge case.
    """
    cur = start
    while cur <= end:
        # last day of cur's month
        if cur.month == 12:
            month_end = datetime.date(cur.year, 12, 31)
        else:
            month_end = datetime.date(cur.year, cur.month + 1, 1) - datetime.timedelta(days=1)
        chunk_end = min(month_end, end)
        yield cur, chunk_end
        cur = chunk_end + datetime.timedelta(days=1)


def download_range(
    session: requests.Session,
    commodity_id: str,
    start_date: datetime.date,
    end_date: datetime.date,
    delay: float = 1.0,
    retries: int = 3,
) -> list:
    """
    Fetch all rows for `commodity_id` between start_date and end_date,
    chunking into ≤30-day windows (TAIFEX caps each request at 1 month).

    Returns a flat list of CSV rows (header excluded).
    """
    all_rows: list = []
    for chunk_start, chunk_end in _iter_month_chunks(start_date, end_date):
        csv_bytes = download_range_chunk(session, commodity_id, chunk_start, chunk_end, retries)
        parsed    = parse_csv_rows(csv_bytes, {commodity_id})
        rows      = parsed.get(commodity_id, [])
        all_rows.extend(rows)
        time.sleep(delay)
    return all_rows


def parse_csv_rows(csv_bytes: bytes, commodity_ids: set | None) -> dict:
    """Parse a direct Big5 CSV (no zip), group rows by commodity code."""
    text   = csv_bytes.decode("big5", errors="replace")
    result = {}
    for i, row in enumerate(csv.reader(io.StringIO(text))):
        if i == 0 or len(row) < 2:
            continue
        code = row[1].strip()
        if commodity_ids is None or code in commodity_ids:
            result.setdefault(code, []).append(row)
    return result


def _row_key(row: list) -> tuple:
    """Dedup key for a TAIFEX row: (date, contract, expiry, session)."""
    session_val = row[17].strip() if len(row) > 17 else "一般"
    return (row[0].strip(), row[1].strip(), row[2].strip(), session_val)


def merge_into_csv(out_path: Path, new_rows: list) -> tuple[int, int]:
    """
    Append `new_rows` into `out_path`, deduping by (date, contract, expiry, session).
    Existing rows are preserved. Output is re-sorted by date.

    Returns (rows_added, total_rows_after).
    """
    existing: list = []
    seen: set = set()

    if out_path.exists():
        with open(out_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0 or len(row) < 2:
                    continue
                key = _row_key(row)
                if key not in seen:
                    seen.add(key)
                    existing.append(row)

    added = 0
    for row in new_rows:
        if len(row) < 2:
            continue
        key = _row_key(row)
        if key not in seen:
            seen.add(key)
            existing.append(row)
            added += 1

    existing.sort(key=lambda r: r[0])

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(existing)

    return added, len(existing)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TAIFEX futures daily market history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--commodity", nargs="+", metavar="CODE",
                   help="One or more contract codes, e.g. TX MTX TMF")
    g.add_argument("--all-index", action="store_true",
                   help="All index / global futures (flag=0 from API)")
    g.add_argument("--all", action="store_true",
                   help="Every contract including stock futures (~618 codes)")
    g.add_argument("--list", action="store_true",
                   help="Print all available contract codes and exit")

    parser.add_argument("--output-dir", default=".",
                        help="Directory for output CSVs (default: current dir)")
    parser.add_argument("--start-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--end-year",   type=int, default=None,
                        help="Last year to download (default: current year)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds to wait between annual requests")

    # --- Date-range mode (down_type=1) — for current-year / YTD data ---------
    parser.add_argument("--start-date", type=parse_cli_date, default=None,
                        metavar="YYYY-MM-DD",
                        help="Start date for range download (YYYY-MM-DD or YYYY/MM/DD). "
                             "Triggers range-mode (down_type=1) instead of the annual zip.")
    parser.add_argument("--end-date", type=parse_cli_date, default=None,
                        metavar="YYYY-MM-DD",
                        help="End date for range download (default: today)")
    parser.add_argument("--append", action="store_true",
                        help="In range mode, merge new rows into existing {CODE}_history.csv "
                             "(dedup by date+contract+expiry+session) instead of overwriting.")

    args = parser.parse_args()

    end_year = args.end_year or datetime.date.today().year - 1

    with requests.Session() as session:

        # -- --list -----------------------------------------------------------
        if args.list:
            contracts = fetch_contract_list(session)
            if contracts:
                print(f"{'Code':<12} {'Type':<8} Name")
                print("-" * 60)
                for c in sorted(contracts, key=lambda x: (x["flag"], x["commodity_id"])):
                    kind = "index" if c["flag"] == 0 else "stock"
                    print(f"{c['commodity_id']:<12} {kind:<8} {c.get('commodity_name', '')}")
                print(f"\nTotal: {len(contracts)} contracts")
            else:
                print("Index futures (hardcoded from TAIFEX dropdown):")
                print("-" * 40)
                for code in sorted(INDEX_FUTURES):
                    print(f"  {code}")
                print(f"\nTotal index: {len(INDEX_FUTURES)} contracts")
                print("Note: full list unavailable — TAIFEX API requires a browser session.")
            return

        # -- resolve commodity_ids --------------------------------------------
        if args.all:
            commodity_ids = None                              # keep everything
        elif args.all_index:
            contracts = fetch_contract_list(session)
            if contracts:
                commodity_ids = {c["commodity_id"] for c in contracts if c["flag"] == 0}
                print(f"Fetched {len(commodity_ids)} index contracts from API")
            else:
                commodity_ids = INDEX_FUTURES
                print(f"Using hardcoded list of {len(commodity_ids)} index contracts")
        else:
            commodity_ids = set(args.commodity)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # -- range mode (down_type=1) ----------------------------------------
        if args.start_date is not None:
            if commodity_ids is None:
                parser.error("Range mode requires explicit --commodity / --all-index "
                             "(cannot use --all in range mode).")

            end_date = args.end_date or datetime.date.today()
            label = ", ".join(sorted(commodity_ids))
            print(f"Range mode  |  {label}  |  {args.start_date} → {end_date}")
            print(f"Output dir: {out_dir.resolve()}"
                  f"  ({'append + dedup' if args.append else 'overwrite'})\n")

            prime_session(session)

            for code in sorted(commodity_ids):
                print(f"  [{code}] downloading...", end=" ", flush=True)
                try:
                    new_rows = download_range(session, code,
                                              args.start_date, end_date,
                                              delay=args.delay)
                    out_path = out_dir / f"{code}_history.csv"

                    if args.append:
                        added, total = merge_into_csv(out_path, new_rows)
                        print(f"+{added:,} new rows  (total {total:,})  -> {out_path.name}")
                    else:
                        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                            writer = csv.writer(f)
                            writer.writerow(COLUMNS)
                            writer.writerows(new_rows)
                        print(f"{len(new_rows):,} rows  -> {out_path.name}")
                except Exception as exc:
                    print(f"FAILED: {exc}")

            print("\nDone.")
            return

        # -- annual zip mode (down_type=2) -----------------------------------
        label = "all contracts" if commodity_ids is None else ", ".join(sorted(commodity_ids))
        print(f"Downloading {label}  |  years {args.start_year}-{end_year}")
        print(f"Output dir: {out_dir.resolve()}\n")

        accumulated = {}

        for year in range(args.start_year, end_year + 1):
            print(f"  [{year}] downloading...", end=" ", flush=True)
            try:
                zip_bytes = download_year(session, year)
                year_rows = extract_rows(zip_bytes, commodity_ids)
                n_total   = sum(len(v) for v in year_rows.values())
                for code, rows in year_rows.items():
                    accumulated.setdefault(code, []).extend(rows)
                summary = "  ".join(
                    f"{k}:{len(v):,}" for k, v in sorted(year_rows.items())
                )
                print(f"{n_total:,} rows    {summary}")
            except Exception as exc:
                print(f"FAILED: {exc}")
            time.sleep(args.delay)

        # -- write one CSV per contract ---------------------------------------
        print()
        for code, rows in sorted(accumulated.items()):
            out_path = out_dir / f"{code}_history.csv"
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
                writer.writerows(rows)
            print(f"  {code}: {len(rows):,} rows -> {out_path.name}")

        print("\nDone.")


if __name__ == "__main__":
    main()
