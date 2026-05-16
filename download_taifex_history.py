"""
Download historical daily futures data from TAIFEX.

Uses the annual zip download endpoint (down_type=2), which contains all
contracts for that year. Filters by the requested commodity codes.

Usage
-----
    # Single contract
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


# -----------------------------------------------------------------------------
# TAIFEX API helpers
# -----------------------------------------------------------------------------

def fetch_contract_list(session: requests.Session) -> list[dict]:
    """Return all futures contracts from the TAIFEX commodity API."""
    resp = session.get(API_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("commodityList", [])


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
    args = parser.parse_args()

    end_year = args.end_year or datetime.date.today().year

    with requests.Session() as session:

        # -- --list -----------------------------------------------------------
        if args.list:
            contracts = fetch_contract_list(session)
            print(f"{'Code':<12} {'Type':<8} Name")
            print("-" * 60)
            for c in sorted(contracts, key=lambda x: (x["flag"], x["commodity_id"])):
                kind = "index" if c["flag"] == 0 else "stock"
                print(f"{c['commodity_id']:<12} {kind:<8} {c.get('commodity_name', '')}")
            print(f"\nTotal: {len(contracts)} contracts")
            return

        # -- resolve commodity_ids --------------------------------------------
        if args.all:
            commodity_ids = None                              # keep everything
        elif args.all_index:
            contracts     = fetch_contract_list(session)
            commodity_ids = {c["commodity_id"] for c in contracts if c["flag"] == 0}
            print(f"Fetched {len(commodity_ids)} index contracts from API")
        else:
            commodity_ids = set(args.commodity)

        # -- download years ---------------------------------------------------
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

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
