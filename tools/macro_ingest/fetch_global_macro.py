"""Fetch + validate the QNT-10 part-2 macro series from FRED."""
from __future__ import annotations

import concurrent.futures as cf
import io

import pandas as pd
import requests

import global_macro_sources as S


class FetchError(RuntimeError):
    pass


def fetch_one(series_id: str, spec: tuple) -> pd.DataFrame:
    """Return a tidy frame [date, value] for one series, or raise."""
    fred_id, label, country, cat, freq, units, lag = spec
    url = S.FRED_CSV.format(id=fred_id)
    r = requests.get(url, headers={"User-Agent": S.FRED_UA}, timeout=60)
    if r.status_code != 200:
        raise FetchError(f"{series_id} ({fred_id}): HTTP {r.status_code}")
    df = pd.read_csv(io.StringIO(r.text))
    if df.shape[1] < 2:
        raise FetchError(f"{series_id} ({fred_id}): unexpected shape {df.shape}")
    df.columns = ["date"] + list(df.columns[1:])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # FRED writes '.' for missing observations; to_numeric turns those into NaN,
    # which we DROP rather than store -- a stored NULL and a never-published
    # observation are different things and the difference matters downstream.
    df["value"] = pd.to_numeric(df[df.columns[-1]], errors="coerce")
    df = df.loc[df["date"].notna() & df["value"].notna(), ["date", "value"]]
    df["date"] = df["date"].dt.date
    return df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


# Each series was probed on 2026-09-01 and had a 2026 observation. Re-assert it
# on every run: a source that silently freezes is the documented failure mode
# here (half the candidates for this ticket were frozen OECD-MEI mirrors), and
# it never raises on its own.
MIN_LAST_OBS = pd.Timestamp("2026-01-01").date()


def validate(series_id: str, spec: tuple, df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems; empty means clean."""
    fred_id, label, country, cat, freq, units, lag = spec
    problems = []
    if df.empty:
        return [f"{series_id}: EMPTY after parsing"]
    last = df["date"].iloc[-1]
    if last < MIN_LAST_OBS:
        problems.append(
            f"{series_id} ({fred_id}): STALE -- last obs {last}, expected >= {MIN_LAST_OBS}. "
            "Source may have been discontinued; do not ingest without re-checking."
        )
    # Frequency sanity: median spacing should match the declared frequency.
    if len(df) > 10:
        gap = pd.Series(pd.to_datetime(df["date"])).diff().dt.days.median()
        want = {"D": (1, 6), "W": (5, 10), "M": (26, 33), "Q": (85, 95)}[freq]
        if not (want[0] <= gap <= want[1]):
            problems.append(
                f"{series_id} ({fred_id}): declared freq {freq} but median gap is {gap:.0f}d"
            )
    return problems


def fetch_all(series: dict | None = None, workers: int = 6):
    """Fetch every series. Returns (frames, problems)."""
    series = series if series is not None else S.FRED_SERIES
    frames, problems = {}, []

    def job(item):
        sid, spec = item
        try:
            return sid, fetch_one(sid, spec), None
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            return sid, None, f"{sid}: {type(e).__name__}: {e}"

    with cf.ThreadPoolExecutor(workers) as ex:
        for sid, df, err in ex.map(job, series.items()):
            if err:
                problems.append(err)
                continue
            problems.extend(validate(sid, series[sid], df))
            frames[sid] = df
    return frames, problems
