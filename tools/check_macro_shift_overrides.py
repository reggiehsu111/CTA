#!/usr/bin/env python3
"""Guard the QNT-19 macro PIT floor and the shift_override table that depends on it.

History: `cta.load_macro_tw` used to be one full calendar day more aggressive
than `cta.load_us_index_tw` — all 12 daily FRED series carry
`pub_lag_days = 0`, so a US close stamped D landed on TW index D, while
`load_us_index_tw` (hard `pit_lag_days >= 1`) landed it on D+1. Copying the five
`us_*` signals' ``shift_override = {"o2o": 1}`` onto a `ctx.macro(...)` signal
was therefore ~a full day of look-ahead: median paired ΔSR +0.271 on o2o across
QNT-14's 198-cell grid, and 15 fake four-gate passers where the legal lag had
none.

Reggie approved the +1 floor on 2026-09-01, so `load_macro_tw` now resolves
`max(pub_lag_days, cta.MACRO_MIN_PUB_LAG_DAYS)` and the two loaders agree. That
makes `{"o2o": 1}` legal on a macro signal — but ONLY while the floor holds.
This script therefore checks three things:

  1. the floor is still in place, and every daily series actually resolves to
     a lag >= 1 (catches a revert, or a new lag-0 series added to the catalog);
  2. no signal module bypasses the floor (`enforce_floor=False`, or an explicit
     `pub_lag_days=` handed to a macro loader);
  3. NO registered signal (macro-sourced or not) sets a `shift_override`
     below the PIT-legal minimum for that variant. QNT-60 re-derived that
     minimum from the RUNNER'S WRITE TIME (15:31 TPE, the only clock time
     that ever stamps a row with its own date) rather than from data-arrival
     time: it is exactly VARIANT_REGISTRY's default shift, so c2c and o2o
     may never be overridden down to 1. See MIN_SHIFT below.

Run standalone:  python3 tools/check_macro_shift_overrides.py
Run under pytest: pytest tools/check_macro_shift_overrides.py
"""
from __future__ import annotations

import inspect
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(os.path.dirname(_ROOT), "QuantResearch", "Libs"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# PIT-legal minimum shift, per variant. QNT-60 REPLACED the derivation here.
#
# The old rule was information-grounds: post-QNT-19-floor, sig[t] is built from
# a US close of date <= t-1 that landed ~04:00-05:00 TPE of day t, so shift 1
# looked legal on every variant (tightest o2o: entry 08:45 TPE of t-1, 3.75h).
# That is the WRONG TEST. The test is whether the VALUE EXISTED at fill time,
# and the value is whatever the runner has written into `mtx_signal_values`.
#
# Measured over the whole table (QNT-60): the only clock time that ever stamps
# a row with its own date is 15:31 TPE (n=1,620 lag-0 rows; no other computed_at
# bucket reaches lag 0). It cannot be earlier — a signal's value for D is indexed
# on `ctx.tw_index`, so it needs the MTX bar for D, which lands 14:00 TPE. So
# `signed[t-k]` first exists at 15:31 TPE of t-k, and the minimum legal shift is
# the smallest k whose 15:31 write precedes the variant's entry:
#
#   variant    entry              shift 1 -> value at   verdict
#   ongap      05:00 TPE t        15:31 t-1             legal, 13h29m
#   day        08:45 TPE t        15:31 t-1             legal, 17h14m
#   noonpause  13:45 TPE t        15:31 t-1             legal, 22h14m
#   night      15:00 TPE t        15:31 t-1             legal, 23h29m
#   c2c        13:45 TPE t-1      15:31 t-1             LOOK-AHEAD by 1h46m -> 2
#   o2o        08:45 TPE t-1      15:31 t-1             LOOK-AHEAD by 6h45m -> 2
#
# i.e. the minimum is exactly VARIANT_REGISTRY's default shift_days, and no
# shift_override below the default is legal while 15:31 is the only compute.
# This is a property of the RUNNER SCHEDULE, not of the loader, so it binds on
# every registered signal — macro-sourced or not. (The five us_* signals carried
# {"o2o": 1} on the old information-grounds reasoning; removed on QNT-60, and it
# bought nothing: mean paired dSR -0.072, worse on 4 of 5.)
#
# If a second, earlier runner invocation is ever built (e.g. a US-only 06:30 TPE
# pass), re-derive this table against ITS write time — do not relax it by hand.
MIN_SHIFT = {"c2c": 2, "o2o": 2, "day": 1,
             "ongap": 1, "night": 1, "noonpause": 1}
MACRO_MIN_SHIFT = MIN_SHIFT          # back-compat alias

# Markers that mean "this signal reads the macro layer".
_MACRO_MARKERS = (".macro(", ".macro_yoy(", "load_macro_tw", "load_macro_yoy_tw")

# Markers that mean "this module opted OUT of the floor". Legitimate only in
# signal_zoo reproduction scripts, never in a registered signal.
_BYPASS_PATTERNS = (
    re.compile(r"enforce_floor\s*=\s*False"),
    re.compile(r"load_macro(?:_yoy)?_tw\([^)]*pub_lag_days\s*="),
)


def _module_source(cls) -> str:
    try:
        return inspect.getsource(sys.modules[cls.__module__])
    except (OSError, KeyError):
        return ""


def _is_macro_sourced(src: str) -> bool:
    return any(m in src for m in _MACRO_MARKERS)


def check_floor() -> list[str]:
    """The loader-side half: the floor exists and binds on every daily series."""
    import cta
    from cta import global_macro as gm
    from cta.global_macro import _resolve_lag
    problems: list[str] = []
    for where, floor in (("cta.global_macro", getattr(gm, "MACRO_MIN_PUB_LAG_DAYS", 0)),
                         ("cta", getattr(cta, "MACRO_MIN_PUB_LAG_DAYS", 0))):
        if floor < 1:
            problems.append(
                f"{where}.MACRO_MIN_PUB_LAG_DAYS = {floor}; the QNT-19 floor is gone. "
                "Without it, `shift_override={'o2o': 1}` on a ctx.macro signal is "
                "a full day of look-ahead (median +0.271 SR on QNT-14's grid).")
    cat = cta.macro_catalog()
    for sid, row in cat.iterrows():
        lag = _resolve_lag(sid, None, True)
        if lag < 1:
            problems.append(
                f"{sid}: resolved PIT lag {lag} < 1 (freq={row['freq']}, "
                f"pub_lag_days={row['pub_lag_days']}) — the floor is not binding")
    return problems


def audit() -> list[str]:
    """Return a list of violation strings (empty == clean)."""
    import cta  # noqa: F401  - populates the signal registry
    from cta.signals._base import SIGNAL_REGISTRY, VARIANT_REGISTRY

    problems: list[str] = check_floor()
    for name, sig in sorted(SIGNAL_REGISTRY.items()):
        cls = type(sig)
        src = _module_source(cls)
        # Checks 1-2 are loader-specific -> macro-sourced modules only.
        if _is_macro_sourced(src):
            for pat in _BYPASS_PATTERNS:
                if pat.search(src):
                    problems.append(
                        f"{name}: its module bypasses the QNT-19 PIT floor "
                        f"({pat.pattern!r}). A registered signal must read the "
                        f"floored alignment; the escape hatch is for reproduction "
                        f"scripts in signal_zoo/ only.")
        # Check 3 is a property of the 15:31 TPE runner schedule (QNT-60), so it
        # binds on EVERY registered signal, not only the macro-sourced ones.
        for variant, shift in (cls.shift_override or {}).items():
            floor = MIN_SHIFT.get(variant)
            if floor is None:
                default = VARIANT_REGISTRY[variant].shift_days if variant in VARIANT_REGISTRY else None
                problems.append(
                    f"{name}: unknown variant '{variant}' in shift_override "
                    f"(variant default {default}); add it to MIN_SHIFT with a derivation")
                continue
            if shift < floor:
                problems.append(
                    f"{name}: shift_override['{variant}'] = {shift} but the PIT-legal "
                    f"minimum on '{variant}' is {floor}. The runner writes "
                    f"mtx_signal_values once a day at 15:31 TPE (QNT-60), so "
                    f"signed[t-{shift}] does not exist at this variant's fill time. "
                    f"Data-arrival time is not the test; the write time is.")
    return problems


def test_macro_signals_respect_pit_legal_shift():
    problems = audit()
    assert not problems, "QNT-19 macro PIT violations:\n  " + "\n  ".join(problems)


if __name__ == "__main__":
    probs = audit()
    if probs:
        print("FAIL — QNT-19 macro PIT violations:")
        for p in probs:
            print("  -", p)
        sys.exit(1)
    print("OK — floor binds on all macro series; no macro-sourced signal "
          "bypasses it; no registered signal sets a shift_override below the "
          "PIT-legal lag for its variant.")
