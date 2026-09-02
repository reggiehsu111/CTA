#!/usr/bin/env python3
"""QNT-45 — regression check for the NFCI freshness tolerance.

NFCI is weekly, Friday-dated, published the following Wednesday 08:30 ET and
ingested by us-macro-ingest at 20:35/21:35 TPE that same Wednesday. The signal
runner gates at 15:31 TPE, i.e. BEFORE the day's ingest, so the observation
dated Friday F is the newest value MAX(date) can return from Wed F+5 through
Wed F+12 → the true maximum legitimate staleness is 12 days, not 10.

At tolerance 10 `fresh_enough("nfci", ...)` deferred nfci_loose_drift_d3_12 on
40% of business days (every Tue and every Wed), which re-levers the live book to
8/9 sleeves on those days.

Run:  python3 signal_zoo/qnt45_verify.py     (read-only; needs RDS)
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import timedelta

sys.path.insert(0, "/home/ubuntu/mtx")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")

import pandas as pd
from db_utils import engine

from cta.signals import _sources as S

TOL = S.SOURCE_TOLERANCE_DAYS["nfci"]
GATE_HOUR, GATE_MIN = 15, 31          # runner pass, TPE
INGEST_LAG_DAYS = 5                   # Fri obs → Wed +5 ingest at 20:35 TPE
TRUE_BOUND = 12                       # = INGEST_LAG_DAYS + 7, measured below

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


obs = sorted(pd.read_sql("SELECT date FROM us_nfci ORDER BY date", engine)["date"].tolist())
created = pd.read_sql(
    "SELECT date, created_at FROM us_nfci WHERE created_at >= '2026-08-10' ORDER BY created_at", engine
)
created["c_tpe"] = pd.to_datetime(created["created_at"]).dt.tz_convert("Asia/Taipei")

print(f"\nSOURCE_TOLERANCE_DAYS['nfci'] = {TOL}   (us_nfci: {len(obs)} rows, {obs[0]} → {obs[-1]})")

# 1. the series really is weekly — the whole argument rests on it
gaps = Counter((b - a).days for a, b in zip(obs, obs[1:]))
check("us_nfci is strictly weekly", set(gaps) == {7}, f"gaps={dict(gaps)}")

# 2. ingest lands AFTER the 15:31 gate, so the gate never sees the same-day value
late = created["c_tpe"].dt.hour * 60 + created["c_tpe"].dt.minute > GATE_HOUR * 60 + GATE_MIN
check("every live ingest lands after the 15:31 TPE gate", bool(late.all()),
      "ingest times TPE: " + ", ".join(created["c_tpe"].dt.strftime("%a %H:%M")))

# 3. worst-case staleness under an on-schedule ingest is exactly 12 days.
#    Visible MAX(date) at 15:31 on T is the newest obs already ingested, i.e. d <= T-6.
rows = []
for T in pd.date_range("2025-09-01", str(obs[-1]), freq="B").date:   # tail needs the NEXT obs to exist
    vis = max((d for d in obs if d <= T - timedelta(days=INGEST_LAG_DAYS + 1)), default=None)
    if vis is not None:
        rows.append(((T - vis).days, T))
worst = max(r[0] for r in rows)
check(f"max staleness at the gate == {TRUE_BOUND}d", worst == TRUE_BOUND,
      f"observed {worst}d over {len(rows)} business days; "
      f"distribution {dict(sorted(Counter(r[0] for r in rows).items()))}")

# 4. the tolerance covers that bound, with slack for a release that slips a day or two
check(f"tolerance {TOL} >= true bound {TRUE_BOUND}", TOL >= TRUE_BOUND)
deferred = [r for r in rows if r[0] > TOL]
check("no on-schedule business day is deferred", not deferred,
      f"{len(deferred)}/{len(rows)} deferred")
was = [r for r in rows if r[0] > 10]
print(f"         (for contrast, the old tolerance of 10 deferred {len(was)}/{len(rows)} "
      f"= {len(was)/len(rows):.0%} — every Tue and every Wed)")

# 5. the tolerance still catches a real feed outage. 2026-07-27 → 08-07 is one:
#    us_nfci sat at 2026-07-17 for 11 days and staleness reached 20d.
check(f"tolerance {TOL} still flags the Jul-2026 ingest outage", TOL < 20,
      "staleness reached 20d on 2026-08-06 → gate correctly defers")

# 6. live call
with engine.connect() as c:
    today, latest = S.taipei_today(), S.latest_date(c, "nfci")
    check("fresh_enough(nfci, today) is True", S.fresh_enough(c, "nfci", today),
          f"today={today}, MAX(date)={latest}, staleness={(today - latest).days}d")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
sys.exit(1 if failures else 0)
