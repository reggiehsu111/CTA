"""
Source-name convention shared by ingest jobs and the signal runner.

Every signal declares its `sources` as a tuple of these string keys. Every
ingest job passes its source key when triggering the runner.

Example:
    class MySignal(Signal):
        sources = ("large_trader", "mtx_1d")

    # runner is invoked with {"source": "large_trader"}
    # → picks up MySignal (source overlap)
    # → checks that "mtx_1d" is also fresh in the DB
    # → recomputes recent history if yes
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# Source key → (table name, per-source freshness query)
# Query MUST return a single-row single-column result: MAX(date).
_FRESHNESS_QUERY: dict[str, str] = {
    "large_trader":  "SELECT MAX(date) FROM tw_large_trader",
    "three_majors":  "SELECT MAX(date) FROM tw_three_majors",
    "options":       "SELECT MAX(date) FROM tw_options_daily",
    "mtx_1d":        "SELECT MAX(date) FROM tw_index_futures_pv WHERE ticker = 'MTX'",
    "us_indexes":    "SELECT MAX(date) FROM us_index_pv",
    "nfci":          "SELECT MAX(date) FROM us_nfci",
}

# Per-source tolerance override in days. Only listed sources get overrides.
#
# NFCI is weekly, Friday-dated, published the following Wednesday 08:30 ET, and
# ingested by us-macro-ingest at 20:35/21:35 TPE that same Wednesday. The signal
# runner fires at 15:31 TPE — i.e. BEFORE the day's ingest — so the observation
# dated Friday F is the newest value MAX(date) can return, at gate time, from
# Wed F+5 right through to Wed F+12. The true maximum legitimate staleness is
# therefore 12 days, not 10:
#
#     run day (15:31 TPE)   staleness   @10      @14
#     Wed (pre-ingest)         12       FAIL     pass
#     Thu                       6       pass     pass
#     Fri                       7       pass     pass
#     Mon                      10       pass     pass
#     Tue                      11       FAIL     pass
#
# At 10 the gate deferred nfci_loose_drift_d3_12 on every Tuesday and every
# Wednesday, which re-levers the live book to 8/9 sleeves on those days (QNT-45,
# found in QNT-39). 14 = the 12-day bound plus two days of slack for a Chicago
# Fed release that slips to Thu/Fri (us-macro-ingest runs daily, so a slipped
# release lands the next evening), while still catching a genuinely dead feed
# inside two weeks. Keep in sync with signal-pnl-tracker/tw_index_routes.py
# ::_SOURCE_STALE_DAYS, which is the web copy of this number.
SOURCE_TOLERANCE_DAYS: dict[str, int] = {
    "nfci": 14,      # was 10; 12 is the true bound at a 15:31 TPE gate + 2 days slack
}

# When each source is expected to be fresh on a given trading day (TPE).
# Used purely for UI: shown in the signal card so users know when data lands.
SOURCE_UPDATE_TPE: dict[str, str] = {
    "large_trader":  "當日 15:00",   # TAIFEX publishes 15:00-15:30; ingest at 15:00 with retry
    "three_majors":  "當日 15:00",
    "options":       "當日 14:00",   # day session summary published shortly after 13:45 close
    "mtx_1d":        "當日 14:00",   # day session close 13:45 + settlement publish ~14:00
    "us_indexes":    "次日 06:00",   # US market closes ~04:00 TPE, yfinance ingest at 06:00
    "nfci":          "週三 20:35",   # Chicago Fed publishes Wed 08:30 ET → Lambda fires Wed 20:35 TPE
}

ALL_SOURCES: tuple[str, ...] = tuple(_FRESHNESS_QUERY.keys())


def taipei_today() -> date:
    return datetime.now(timezone(timedelta(hours=8))).date()


def latest_date(conn, source: str) -> date | None:
    """Return MAX(date) for the given source, or None if the source is empty."""
    from sqlalchemy import text
    if source not in _FRESHNESS_QUERY:
        raise ValueError(f"unknown source {source!r}; valid: {ALL_SOURCES}")
    row = conn.execute(text(_FRESHNESS_QUERY[source])).fetchone()
    return row[0] if row and row[0] else None


def fresh_enough(conn, source: str, target: date, tolerance_days: int = 3) -> bool:
    """True iff `source`'s latest date is within `tolerance_days` of `target`.

    A 3-day tolerance handles weekends and single-day US-market holidays,
    which cause the "last available date" to legitimately trail today.
    Sources listed in SOURCE_TOLERANCE_DAYS override the caller's default
    (e.g. weekly NFCI needs at least 10 days).
    """
    effective = SOURCE_TOLERANCE_DAYS.get(source, tolerance_days)
    d = latest_date(conn, source)
    return d is not None and d >= target - timedelta(days=effective)
