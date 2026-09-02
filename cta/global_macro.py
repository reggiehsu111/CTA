"""
Global / regional macro loaders — PIT-safe, TW-calendar aligned. (QNT-10 part 2)

Backs the tidy pair `macro_series_meta` + `macro_series`, ingested by
`tools/macro_ingest/ingest_global_macro.py`. Covers Korea, China, Japan trade,
the semiconductor cycle, and the US / global series the wide `us_*` tables do
not carry.

Same discipline as `asia_macro.py`, and for the same reason: every monthly
series is stamped with its REFERENCE period, not its release date. Joining a
monthly frame onto a daily calendar by reference date hands the backtest up to
five months of look-ahead here (`us_corp_profits` carries a 150-day lag). So
`load_macro_tw()` shifts each observation forward by the documented,
deliberately LATE publication lag stored in `macro_series_meta.pub_lag_days`,
then merge-asof's backward onto the TW trading calendar. Prefer it.

`load_macro()` returns the raw reference-dated series. It is for coverage audits
and event masks, not for building a tradable signal.

None of these lags is a scraped release date. They are conservative conventions.
State the assumption in any write-up, per the house rule on fill time.

Daily series (FX, yields, breakevens, WTI) carry `pub_lag_days = 0` in the
metadata because that is their reference-date convention. They are US-close
observations and are not knowable during the Taipei session of the same date, so
since QNT-19 `load_macro_tw` applies a hard **+1 calendar-day floor** to every
resolved lag — the same convention `load_us_index_tw` enforces with
`pit_lag_days >= 1`. An observation stamped D therefore lands on TW index D+1,
and the two loaders now agree. See `load_macro_tw` for the derivation, the
measured cost, and the `enforce_floor=False` escape used to reproduce
pre-QNT-19 numbers.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd


def _engine():
    from db_utils import engine
    return engine


@lru_cache(maxsize=1)
def macro_catalog() -> pd.DataFrame:
    """One row per available series: label, country, category, units, pub lag."""
    df = pd.read_sql(
        "SELECT series_id, label, country, category, freq, units, source, "
        "source_id, pub_lag_days, first_obs, last_obs, n_obs "
        "FROM macro_series_meta ORDER BY country, category, series_id",
        _engine())
    if df.empty:
        raise RuntimeError(
            "macro_series_meta is empty — run tools/macro_ingest/ingest_global_macro.py")
    return df.set_index("series_id")


def available_macro_series(country: str | None = None,
                           category: str | None = None) -> list[str]:
    cat = macro_catalog()
    if country:
        cat = cat[cat["country"] == country.upper()]
    if category:
        cat = cat[cat["category"] == category]
    return sorted(cat.index)


def show_macro_catalog(country: str | None = None, category: str | None = None):
    """Printable catalog — start here rather than guessing a series_id."""
    cat = macro_catalog()
    if country:
        cat = cat[cat["country"] == country.upper()]
    if category:
        cat = cat[cat["category"] == category]
    return cat[["label", "country", "category", "freq", "units",
                "pub_lag_days", "first_obs", "last_obs", "n_obs"]]


def macro_pub_lag(series_id: str) -> int:
    return int(macro_catalog().loc[series_id, "pub_lag_days"])


@lru_cache(maxsize=64)
def load_macro(series_id: str) -> pd.Series:
    """Raw series indexed by REFERENCE date. Not PIT-safe — see module docstring."""
    cat = macro_catalog()
    if series_id not in cat.index:
        raise ValueError(
            f"Unknown series_id {series_id!r}. "
            f"Try cta.show_macro_catalog() — {len(cat)} series available.")
    df = pd.read_sql(
        "SELECT date, value FROM macro_series WHERE series_id = %(s)s ORDER BY date",
        _engine(), params={"s": series_id})
    if df.empty:
        raise RuntimeError(f"{series_id} has a meta row but no observations")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].rename(series_id)


# ── QNT-19: the PIT floor ──────────────────────────────────────────────────
# No macro observation stamped with reference date D is knowable during the
# Taipei session of D. Every daily series in this layer is a FRED US/NY-close
# print landing ~04:00-05:00 TPE of D+1; every monthly and quarterly series
# already carries a lag well above 1. So a floor of one calendar day is exactly
# the 12 daily series in practice, and a no-op for the other 31.
#
# This makes `load_macro_tw` agree with `load_us_index_tw`'s `pit_lag_days >= 1`
# hard floor. Approved by Reggie on QNT-19 (2026-09-01).
MACRO_MIN_PUB_LAG_DAYS = 1


def _resolve_lag(series_id: str, pub_lag_days: int | None,
                 enforce_floor: bool) -> int:
    """Resolved publication lag in calendar days, with the QNT-19 floor applied.

    The default path (`pub_lag_days=None`) floors silently — that is the point
    of the floor. An EXPLICIT lag below the floor raises rather than being
    quietly overridden, mirroring `cta.load_us_index_tw`; pass
    `enforce_floor=False` if you are deliberately reproducing a pre-QNT-19
    number, and say so in the write-up.
    """
    lag = macro_pub_lag(series_id) if pub_lag_days is None else int(pub_lag_days)
    if not enforce_floor:
        return lag
    if lag >= MACRO_MIN_PUB_LAG_DAYS:
        return lag
    if pub_lag_days is not None:
        raise ValueError(
            f"pub_lag_days={pub_lag_days} for {series_id!r} is below the QNT-19 "
            f"PIT floor of {MACRO_MIN_PUB_LAG_DAYS} calendar day. A reference "
            f"date D is not observable during the TW session of D. Pass "
            f"enforce_floor=False only to reproduce a pre-QNT-19 number.")
    return MACRO_MIN_PUB_LAG_DAYS


def _align(raw: pd.Series, trading_index, lag_days: int, name: str) -> pd.Series:
    """Shift each observation to its earliest observable date, then merge-asof
    backward onto the TW trading calendar."""
    if trading_index is None or len(trading_index) == 0:
        return pd.Series([], dtype=float, name=name)
    if raw.empty:
        return pd.Series(index=pd.DatetimeIndex(trading_index), dtype=float, name=name)

    src = pd.DataFrame({
        "available_from": pd.DatetimeIndex(raw.index) + pd.Timedelta(days=lag_days),
        "value": raw.values,
    }).sort_values("available_from").reset_index(drop=True)

    tw_idx = pd.DatetimeIndex(trading_index).sort_values()
    tw = pd.DataFrame({"tw_date": tw_idx})
    # pandas 3 infers datetime64[s] from psycopg2 dates while the TW calendar is
    # ns — merge_asof raises on mismatched resolutions, so cast both keys.
    tw["tw_date"] = tw["tw_date"].astype("datetime64[ns]")
    src["available_from"] = src["available_from"].astype("datetime64[ns]")

    merged = pd.merge_asof(tw, src, left_on="tw_date", right_on="available_from",
                           direction="backward")
    return pd.Series(merged["value"].values, index=tw_idx, name=name).reindex(
        pd.DatetimeIndex(trading_index))


def load_macro_tw(series_id: str, trading_index,
                  pub_lag_days: int | None = None,
                  enforce_floor: bool = True) -> pd.Series:
    """PIT-safe series on the TW trading calendar.

    Each reference period starting on D is treated as first observable on
    `D + lag`, then forward-filled, where `lag` is
    `max(macro_series_meta.pub_lag_days, MACRO_MIN_PUB_LAG_DAYS)`.
    Override `pub_lag_days` only if you can justify a LATER fill time — and say
    so in the write-up.

    ── The +1 floor (QNT-19, approved 2026-09-01) ─────────────────────────────

    Before QNT-19 this loader was one full calendar day more aggressive than
    `cta.load_us_index_tw`. All 12 daily FRED series carry `pub_lag_days = 0`,
    so a US close stamped D landed on TW index D, while `load_us_index_tw`
    (hard `pit_lag_days >= 1`) landed the same observation on D+1. The two are
    now aligned:

        loader                convention                       US obs of D lands on
        ------------------    ---------------------------      --------------------
        cta.load_us_index_tw  pit_lag_days=1 (hard floor)      TW index D+1
        cta.load_macro_tw     max(pub_lag_days, 1)             TW index D+1

    Every daily series here is a FRED US-close / NY-close observation — the
    US-country rates and DXY, but also `twd_usd`, `krw_usd`, `cny_usd` (NY
    close FX) and `wti` (Cushing spot). The `country` column says CN/KR/TW/GL
    for some of them; that is the ECONOMY, not the observation timezone. All 12
    become knowable at roughly 04:00-05:00 TPE of D+1, and the floor is what
    encodes that. The 31 monthly/quarterly/weekly series already carry lags of
    7-150 days, so the floor never binds on them.

    ── What is PIT-legal now ─────────────────────────────────────────────────

    With the floor in place, `sig[t]` is built from a US close of date `<= t-1`,
    which landed ~04:00-05:00 TPE of day `t`. Every variant's DEFAULT shift is
    therefore legal, with margin, and `shift >= 1` is the rule for all six:

        variant     shift  entry            margin over the print
        ---------   -----  ---------------  ---------------------
        c2c           2*   13:45 TPE t-1    8.75h at shift 1
        o2o           2*   08:45 TPE t-1    3.75h at shift 1  (tightest)
        ongap         1    05:00 TPE t      24h
        day           1    08:45 TPE t      27.8h
        noonpause     1    13:45 TPE t      32.8h
        night         1    15:00 TPE t      34h
        (* the c2c/o2o variant defaults are one day looser than they need to be
           for a macro-sourced signal; that is conservative, not wrong.)

    Two consequences worth knowing:

    * The floor removes a **loader** asymmetry that used to be worth a median
      paired ΔSR of **+0.271** on o2o (76.8% win-rate) and **+0.145** on c2c
      across QNT-14's 198-cell grid, and manufactured 15 fake four-gate passers
      on o2o where the legal lag had none.
      Evidence: `mtx/signal_zoo/macro_windows/lag_decomposition.csv`.

      It does **not** license `shift_override = {"o2o": 1}`. The five `us_*` /
      `tv_*` signals used to carry that override; QNT-60 removed it from all
      five and QNT-72 deploys the removal. The binding constraint is not when
      the input lands, it is when the RUNNER WRITES the value — once a day at
      15:31 TPE, the only clock time that ever stamps a row with its own date.
      `signed[t-1]` therefore does not exist at o2o's 08:45 t-1 fill, floor or
      no floor, and the PIT-legal minimum shift is the variant default
      (c2c 2 · o2o 2 · day 1 · ongap 1 · night 1 · noonpause 1).
    * `ongap` at its default shift(1) is no longer a fill-time race. Pre-floor
      its 05:00 TPE window opened at or before the H.15 / Treasury-curve post.

    ── Cost of the floor ─────────────────────────────────────────────────────

    Measured on QNT-14's 198-cell macro grid before adoption (median paired
    ΔSR, floored minus pre-floor, variant shifts held fixed): c2c **+0.024**,
    o2o **−0.021**, day **−0.031**; no candidate crosses a house gate because
    of it, though individual cells move up to ~0.55 SR.
    Evidence: `mtx/signal_zoo/macro_windows/floor_cost.csv`.

    The floor is a CALENDAR-day floor, not a one-row shift, so it is close to
    but not the same as running the pre-QNT-19 loader one variant-shift later:
    the two disagree on 2.5% of rows (0.0-3.0% by series), all of them around TW
    holidays and long weekends, where a calendar floor still picks up the US
    print from a day the TWSE was closed. That 2.5% is why the QNT-19 re-run was
    done against the real loader rather than trusting the `lag+1` proxy in
    `floor_cost.py` — though in aggregate the proxy was accurate to ~0.005 SR.
    To reproduce a pre-QNT-19 number exactly, pass `enforce_floor=False` and
    state it in the write-up.

    Guard: `tools/check_macro_shift_overrides.py`.
    """
    raw = load_macro(series_id)
    lag = _resolve_lag(series_id, pub_lag_days, enforce_floor)
    return _align(raw, trading_index, lag, f"{series_id}_tw")


def load_macro_yoy_tw(series_id: str, trading_index, periods: int = 12,
                      pub_lag_days: int | None = None,
                      enforce_floor: bool = True) -> pd.Series:
    """Year-on-year percent change, computed on the RAW reference-dated series
    (so the lookback spans real periods) and only then aligned PIT.

    `periods` is in observations, not months — 12 for monthly, 4 for quarterly.
    Check `macro_catalog().loc[series_id, 'freq']` before trusting the default.

    Carries the same QNT-19 +1 calendar-day floor as `load_macro_tw`; see there.
    """
    raw = load_macro(series_id)
    yoy = (raw.pct_change(periods) * 100.0).dropna()
    lag = _resolve_lag(series_id, pub_lag_days, enforce_floor)
    return _align(yoy, trading_index, lag, f"{series_id}_yoy_tw")
