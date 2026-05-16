"""
Download full history of 小型台指期 (MTX) daily market data from TAIFEX.

Strategy:
- Uses the annual zip download endpoint (down_type=2) for each year 2001–present.
- Filters CSV rows for contract code "MTX".
- Outputs a single combined CSV to Research/mtx/mtx_history.csv

Usage:
    python download_mtx_history.py [--output mtx_history.csv] [--start-year 2001]

Run with the project virtualenv:
    /Users/hsureggie/.pyenv/versions/3.14.0/envs/gary-research/bin/python3 download_mtx_history.py
"""

import argparse
import csv
import io
import time
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://www.taifex.com.tw/cht/3/futDataDown"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://www.taifex.com.tw/cht/3/dlFutDailyMarketView",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

COLUMNS = [
    "交易日期", "契約", "到期月份(週別)", "開盤價", "最高價", "最低價",
    "收盤價", "漲跌價", "漲跌%", "成交量", "結算價", "未沖銷契約數",
    "最後最佳買價", "最後最佳賣價", "歷史最高價", "歷史最低價",
    "是否因訊息面暫停交易", "交易時段", "價差對單式委託成交量",
]

# MTX started trading on 2001/04/02
MTX_START_YEAR = 2001


def download_year(session: requests.Session, year: int, retries: int = 3) -> bytes:
    """POST to TAIFEX annual download endpoint; return raw zip bytes."""
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
            print(f"  Attempt {attempt} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)


def extract_mtx_rows(zip_bytes: bytes, year: int) -> list[list[str]]:
    """Unzip in-memory, parse Big5 CSV, return MTX rows (excluding header)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The zip contains one file named like 2024_fut.csv
        names = zf.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), names[0])
        raw = zf.read(csv_name)

    text = raw.decode("big5", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = []
    for i, row in enumerate(reader):
        if i == 0:
            continue  # skip header
        if len(row) > 1 and row[1].strip() == "MTX":
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Download MTX (小型台指期) history from TAIFEX")
    parser.add_argument("--output", default="/Users/hsureggie/coding/Research/mtx/mtx_history.csv", help="Output CSV path")
    parser.add_argument("--start-year", type=int, default=MTX_START_YEAR)
    parser.add_argument("--end-year", type=int, default=None, help="Last year to download (default: current year)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    args = parser.parse_args()

    import datetime
    end_year = args.end_year or datetime.date.today().year
    years = list(range(args.start_year, end_year + 1))

    output_path = Path(args.output)
    print(f"Downloading MTX data for years {args.start_year}–{end_year}")
    print(f"Output: {output_path.resolve()}\n")

    all_rows: list[list[str]] = []

    with requests.Session() as session:
        for year in years:
            print(f"  [{year}] downloading…", end=" ", flush=True)
            try:
                zip_bytes = download_year(session, year)
                rows = extract_mtx_rows(zip_bytes, year)
                all_rows.extend(rows)
                print(f"{len(rows):,} MTX rows")
            except Exception as exc:
                print(f"FAILED: {exc}")
            time.sleep(args.delay)

    print(f"\nTotal MTX rows: {len(all_rows):,}")
    print(f"Writing to {output_path}…")

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(all_rows)

    print("Done.")


if __name__ == "__main__":
    main()
