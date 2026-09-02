"""
Signal runner.

  python -m cta.signals.runner --backfill                 # compute full history for all signals
  python -m cta.signals.runner --backfill --start 2020    # bounded backfill
  python -m cta.signals.runner --name lt_top10npct_signth_w60  # just one signal
  python -m cta.signals.runner                            # daily incremental (last 30 days)

Steps for each signal:
  1. call signal.compute_raw(ctx)               → raw pandas Series
  2. cta.normalize_signal(raw, tanh, w=252)     → normalized to [-1, +1]
  3. multiply by signal.sign (frozen)           → SIGNED
  4. for each variant in signal.variants:
       - position = SIGNED.shift(variant.shift_days)
       - pnl      = position × ret − turnover × cost
       - upsert (date, name, variant) → mtx_signal_values

Also syncs signal metadata into mtx_signal_config on every run so the DB
mirrors the code.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_MTX  = _HERE.parent.parent               # .../Research/mtx
_LIBS = _MTX.parent / "QuantResearch" / "Libs"
for p in (str(_MTX), str(_LIBS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cta                                # noqa: E402
from cta.signals._base import SIGNAL_REGISTRY, VARIANT_REGISTRY, compute_variant_pnl  # noqa: E402
from cta.signals._ctx  import build_context                                            # noqa: E402
from db_utils import engine                                                            # noqa: E402
from sqlalchemy import text                                                            # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("signal_runner")

_NORM_WINDOW = 252


# ── config sync ────────────────────────────────────────────────────────────

def _sync_config(conn):
    """Upsert each signal's metadata into mtx_signal_config."""
    rows = []
    for name, sig in SIGNAL_REGISTRY.items():
        rows.append({
            "signal_name":          name,
            "cn_name":              sig.cn_name or None,
            "cn_short":             sig.cn_short or None,
            "enabled":              sig.enabled,
            "live_date":            sig.live_date or None,
            "description":          sig.description,
            "cn_description":       sig.cn_description,
            "sources":              list(sig.sources),
            "cadence":              sig.cadence,
            "recommended_variants": list(sig.recommended_variants) or None,
        })
    if not rows:
        return
    # Note: recommended_variants is INSERTed on first sync but is NOT
    # overwritten on subsequent syncs — that lets an operator tune it in
    # the DB without a code redeploy overwriting their choice.
    conn.execute(text("""
        INSERT INTO mtx_signal_config (signal_name, cn_name, cn_short, enabled, live_date,
                                        description, cn_description,
                                        sources, cadence, recommended_variants,
                                        updated_at)
        VALUES (:signal_name, :cn_name, :cn_short, :enabled, :live_date, :description,
                :cn_description, :sources, :cadence, :recommended_variants,
                now())
        ON CONFLICT (signal_name) DO UPDATE SET
          cn_name        = EXCLUDED.cn_name,
          cn_short       = EXCLUDED.cn_short,
          enabled        = EXCLUDED.enabled,
          live_date      = EXCLUDED.live_date,
          description    = EXCLUDED.description,
          cn_description = EXCLUDED.cn_description,
          sources        = EXCLUDED.sources,
          cadence        = EXCLUDED.cadence,
          updated_at     = now()
    """), rows)


# ── compute + upsert one signal ────────────────────────────────────────────

def _compute_one(sig, ctx, start: pd.Timestamp | None, end: pd.Timestamp | None):
    """Returns list of upsert row dicts for one signal, spanning all its variants."""
    raw = sig.compute_raw(ctx)
    if raw is None or raw.empty:
        log.warning("[%s] compute_raw returned empty", sig.name)
        return []

    # Sanitize + normalize + apply frozen sign (avoids look-ahead sign discovery)
    sanitized = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    normed    = cta.normalize_signal(
        sanitized, method="tanh", window=_NORM_WINDOW, force=True
    ).rename(sig.name)
    signed    = normed * sig.sign

    all_rows: list[dict] = []
    for variant_key in sig.variants:
        variant = VARIANT_REGISTRY[variant_key]
        shift = sig.shift_override.get(variant_key, variant.shift_days)
        position, pnl = compute_variant_pnl(signed, ctx.asset, variant, shift_days=shift)

        # Assemble frame indexed on tw calendar
        df = pd.DataFrame({
            "raw_value": signed,
            "position":  position,
            "pnl_1d":    pnl,
        }, index=ctx.tw_index).dropna(how="all")

        if start is not None: df = df.loc[df.index >= start]
        if end   is not None: df = df.loc[df.index <= end]

        for ts, row in df.iterrows():
            all_rows.append({
                "date":        ts.date() if isinstance(ts, pd.Timestamp) else ts,
                "signal_name": sig.name,
                "variant":     variant_key,
                "raw_value":   float(row["raw_value"]) if pd.notna(row["raw_value"]) else None,
                "position":    float(row["position"])  if pd.notna(row["position"])  else None,
                "pnl_1d":      float(row["pnl_1d"])    if pd.notna(row["pnl_1d"])    else None,
            })
    return all_rows


def _upsert_values(conn, rows):
    if not rows:
        return 0
    conn.execute(text("""
        INSERT INTO mtx_signal_values (date, signal_name, variant,
                                        raw_value, position, pnl_1d, computed_at)
        VALUES (:date, :signal_name, :variant,
                :raw_value, :position, :pnl_1d, now())
        ON CONFLICT (date, signal_name, variant) DO UPDATE SET
          raw_value   = EXCLUDED.raw_value,
          position    = EXCLUDED.position,
          pnl_1d      = EXCLUDED.pnl_1d,
          computed_at = now()
    """), rows)
    return len(rows)


# ── entry point ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true",
                    help="Compute full history (else last 30 days only).")
    ap.add_argument("--start", help="ISO start date (inclusive).")
    ap.add_argument("--end",   help="ISO end date (inclusive).")
    ap.add_argument("--name",  action="append",
                    help="Restrict to specific signal name(s); repeatable.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute but do not upsert.")
    args = ap.parse_args()

    ctx = build_context()
    log.info("loaded asset: %d rows  %s → %s",
             len(ctx.asset), ctx.tw_index.min().date(), ctx.tw_index.max().date())

    if args.backfill:
        start = pd.Timestamp(args.start) if args.start else None
        end   = pd.Timestamp(args.end)   if args.end   else None
    else:
        end   = pd.Timestamp(args.end) if args.end else ctx.tw_index.max()
        start = end - pd.Timedelta(days=30)
        if args.start:
            start = pd.Timestamp(args.start)

    log.info("range: %s → %s", start, end)

    names = args.name or list(SIGNAL_REGISTRY)
    log.info("computing %d signals: %s", len(names), names)

    # Sync config first (fast, one small commit)
    with engine.begin() as conn:
        _sync_config(conn)

    total = 0
    for name in names:
        sig = SIGNAL_REGISTRY[name]
        if not sig.enabled:
            log.info("[%s] disabled — skip", name); continue
        t0 = pd.Timestamp.now()
        rows = _compute_one(sig, ctx, start, end)
        elapsed_compute = (pd.Timestamp.now() - t0).total_seconds()
        if args.dry_run:
            log.info("[%s] would upsert %d rows (%d variants, compute %.1fs)",
                     name, len(rows), len(sig.variants), elapsed_compute)
            continue
        # Commit per signal so partial progress lands even if a later one fails
        with engine.begin() as conn:
            n = _upsert_values(conn, rows)
        log.info("[%s] upsert %d rows (%d variants, compute %.1fs)",
                 name, n, len(sig.variants), elapsed_compute)
        total += n

    log.info("done. total upserts = %d", total)


if __name__ == "__main__":
    main()
