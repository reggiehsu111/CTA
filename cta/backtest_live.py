"""
Strict, execution-realistic backtest of the live MTX signal book.

Motivation
----------
`mtx_signal_values.pnl_1d` (written by the mtx-signal-runner Lambda) is a
*research* P&L: fractional positions in [-1, +1], returns from raw
`close.pct_change()` / `open.pct_change()`, and a flat cost stub. None of
those three survive contact with a real account. This module rebuilds the
same book under the constraints an actual trader faces:

  1. ONE execution per day, at a time you can actually be at the screen:
        scheme 'close' → fill at the 13:45 day-session close   (variant c2c)
        scheme 'open'  → fill at the 08:45 day-session open    (variant o2o)
     Both use the positions the Lambda already stored (shift(2), i.e. the
     signal published ~15:31 TPE on day t is filled on day t+1). The stored
     c2c and o2o position paths are identical, so the two schemes differ
     ONLY in return window and cost basis.

  2. Contract-consistent returns. The front month changes ~12x/year; a raw
     pct_change across that boundary books the calendar spread as P&L. Here
     every daily return is computed on a SINGLE expiry held across the whole
     window, and the roll itself is charged as two extra legs of cost.

  3. Integer contracts. The book is capped at `max_contracts` at |pos|=1,
     rounded to whole MTX contracts, with a no-trade band to stop 1-lot
     churn. Costs are charged on the INTEGER change, not the fractional one.

  4. An itemised cost model in TWD per side, replacing the `20/(c*50)+2e-5`
     stub — see `Costs`.

The headline output is `ladder()`, which turns each of those on one at a
time so the Sharpe cost of every assumption is attributable.

Everything reads from RDS. The CSV under `mtx/history_data/` is stale
(ends 2026-07-27) and its final row is night-table-corrupt, so it is
deliberately never touched here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_LIBS = "/Users/hsureggie/coding/Research/QuantResearch/Libs"
if _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)


# MTX (小型臺指期貨) big-point value: TWD per index point per contract.
CONTRACT_MULT = 50.0

# Scheme → (price column used for the fill, mtx_signal_values.variant)
SCHEMES: dict[str, tuple[str, str]] = {
    "close": ("close", "c2c"),
    "open":  ("open",  "o2o"),
}


def _engine():
    from db_utils import engine
    return engine


# ─────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Costs:
    """Per-side, per-contract trading costs for MTX.

    Defaults are the honest retail numbers, and they are materially heavier
    than the runner's stub:

        runner stub : 20 TWD + 2bps            ≈  65 TWD/side @ index 45,000
        this model  : 20 + 45 (tax) + 50 (1pt) ≈ 115 TWD/side @ index 45,000

    The stub's `0.00002` is documented as "slippage/impact", which means
    期交稅 (also exactly 2bps per side) was never charged at all. Hence the
    ~1.8x understatement.

    commission_twd   期貨商手續費, per side per contract. Retail MTX is
                     ~15-30 TWD; 20 is mid.
    tax_rate         期交稅率 for 股價指數期貨: 十萬分之二 = 2e-5 of
                     contract value, levied per side.
    slippage_points  Execution slippage in index points per side. Median
                     MTX bid-ask over 2023+ is 3.0 points, so 1.5 = paying
                     the full half-spread; 1.0 assumes you capture a little
                     via the closing auction.
    """
    commission_twd:  float = 20.0
    tax_rate:        float = 2e-5
    slippage_points: float = 1.0

    def per_side_twd(self, price: pd.Series | float) -> pd.Series | float:
        """TWD cost of trading ONE contract, one side, at `price`."""
        return (self.commission_twd
                + self.tax_rate * price * CONTRACT_MULT
                + self.slippage_points * CONTRACT_MULT)


# The runner's stub, kept so the ladder can show what it was charging.
STUB_COSTS = Costs(commission_twd=20.0, tax_rate=2e-5, slippage_points=0.0)


# ─────────────────────────────────────────────────────────────────────────
# Price panel
# ─────────────────────────────────────────────────────────────────────────

def load_price_panel(ticker: str = "MTX") -> dict[str, pd.DataFrame]:
    """Return {'open': df, 'close': df} pivots of date × expiry_month.

    Monthly expiries only. Weekly codes ('202608W1') are excluded: they are
    never the contract a daily CTA holds, and their thin volume is what
    poisoned the tracker's returns during the July-August ingest outage.
    """
    q = """
        SELECT date, expiry_month, open, close
          FROM tw_index_futures_pv
         WHERE ticker = %(t)s
           AND expiry_month ~ '^[0-9]+$'
         ORDER BY date, expiry_month
    """
    raw = pd.read_sql(q, _engine(), params={"t": ticker.upper()})
    if raw.empty:
        raise RuntimeError(f"tw_index_futures_pv has no rows for {ticker!r}")

    raw["date"] = pd.to_datetime(raw["date"])
    raw["expiry_month"] = raw["expiry_month"].astype(int)
    for c in ("open", "close"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    return {
        field: raw.pivot(index="date", columns="expiry_month", values=field).sort_index()
        for field in ("open", "close")
    }


def held_expiry(panel: dict[str, pd.DataFrame], roll_lead_days: int = 0) -> pd.Series:
    """Which expiry the book holds on each date.

    `roll_lead_days=0` holds the nearest monthly expiry, i.e. rolls the day
    after it drops off the board. That is the convention the signal runner
    and the tracker both use, but it is NOT tradable: the last print of an
    expiring contract is the final settlement, which you cannot trade out
    of at will.

    `roll_lead_days=k` moves the roll k trading days earlier, which is what
    a human actually does. k=2 or 3 is typical for TAIFEX index futures.
    """
    closes = panel["close"]
    # Front = smallest expiry with a print on that date.
    front = closes.apply(lambda row: row.dropna().index.min(), axis=1)
    if roll_lead_days > 0:
        # Hold today whatever will be front `k` trading days from now.
        front = front.shift(-roll_lead_days).ffill()
    return front.rename("held_expiry")


def contract_prices(panel: dict[str, pd.DataFrame],
                    roll_lead_days: int = 0) -> pd.DataFrame:
    """Per-date price of the held contract, at both ends of each window.

    Columns
    -------
    held        expiry held during the window ENDING on this date
    px          held contract's price on this date          (fill-out price)
    px_prev     SAME contract's price on the previous date   (fill-in price)
    is_roll     True when `held` differs from the previous date's `held`
    front_px    nearest-monthly price, for reference/benchmarking

    Because `px` and `px_prev` are the same expiry, the return they imply
    contains no calendar spread. The roll shows up where it belongs: as
    `is_roll`, which `simulate()` charges two extra legs for.
    """
    held = held_expiry(panel, roll_lead_days)
    out = {}
    for field in ("open", "close"):
        px_mat = panel[field]
        dates = px_mat.index
        # Vectorised lookup: for each date, the held contract's price today
        # and the same contract's price on the prior date.
        col_pos = {e: i for i, e in enumerate(px_mat.columns)}
        vals = px_mat.to_numpy()
        idx = np.array([col_pos.get(e, -1) for e in held.to_numpy()])
        rows = np.arange(len(dates))

        px = np.full(len(dates), np.nan)
        ok = idx >= 0
        px[ok] = vals[rows[ok], idx[ok]]

        px_prev = np.full(len(dates), np.nan)
        ok_prev = ok.copy()
        ok_prev[0] = False
        px_prev[1:] = np.where(ok[1:], vals[rows[1:] - 1, np.maximum(idx[1:], 0)], np.nan)

        out[f"{field}_px"] = pd.Series(px, index=dates)
        out[f"{field}_px_prev"] = pd.Series(px_prev, index=dates)

    df = pd.DataFrame(out)
    df["held"] = held
    df["is_roll"] = held.ne(held.shift(1)) & held.shift(1).notna()
    df["front_px"] = panel["close"].apply(lambda r: r.dropna().iloc[0] if r.notna().any() else np.nan, axis=1)
    return df


def scheme_returns(prices: pd.DataFrame, scheme: str) -> pd.Series:
    """Roll-adjusted fractional return of the held contract for one scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; valid: {list(SCHEMES)}")
    field, _ = SCHEMES[scheme]
    px, prev = prices[f"{field}_px"], prices[f"{field}_px_prev"]
    return (px / prev - 1).rename(f"{scheme}_ret")


# ─────────────────────────────────────────────────────────────────────────
# Positions
# ─────────────────────────────────────────────────────────────────────────

def load_positions(scheme: str, enabled_only: bool = True,
                   us_morning_run: bool = False) -> pd.DataFrame:
    """date × signal_name matrix of stored fractional positions.

    These are the positions the live Lambda actually computed and wrote —
    not a re-derivation — so the backtest inherits the real signal path,
    including each signal's frozen sign and its PIT lag.

    The `us_morning_run` flag exists because of a real gap between what the
    stored `o2o` values assume and what the pipeline currently does.

    Five US-driven signals USED TO carry ``shift_override = {"o2o": 1}``,
    justified by "US data lands 06:00 TPE, before the 08:45 open". On
    information grounds that is true. Operationally it is NOT: the runner
    fires once a day at ~15:31 TPE, so the value those positions need at
    08:45 does not exist for another seven hours. QNT-60 removed the override
    on that reasoning; QNT-72 deploys the removal to `mtx-signal-runner` and
    re-backfills the affected rows.

        us_morning_run=False  (default) → shift(2) for every signal, i.e.
            the c2c position path. This is what you could actually trade
            at tomorrow's open with the pipeline as it stands.
        us_morning_run=True → the stored o2o path, valid only if a second,
            US-only morning runner invocation (~06:30 TPE) is ever built.

    NOTE: once QNT-72 is deployed and the o2o rows are re-backfilled, the
    stored o2o path is itself shift(2) and this flag becomes a no-op — both
    branches then give the same shift, differing only in nothing at all
    (positions are the shifted signal; the return window is chosen by
    `scheme`). Until the deploy lands, the stored o2o rows still hold the
    shift(1) path, so leave the default alone. The flag is kept so the
    distinction is not silently lost if the morning runner is built.

    The flag is a no-op for scheme='close': c2c has no shift overrides.
    """
    if scheme == "open" and not us_morning_run:
        variant = "c2c"
    else:
        _, variant = SCHEMES[scheme]
    q = """
        SELECT v.date, v.signal_name, v.position
          FROM mtx_signal_values v
          JOIN mtx_signal_config c USING (signal_name)
         WHERE v.variant = %(var)s
           AND (%(all)s OR c.enabled)
         ORDER BY v.date
    """
    df = pd.read_sql(q, _engine(), params={"var": variant, "all": not enabled_only})
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="signal_name", values="position").sort_index()


def load_live_dates() -> dict[str, pd.Timestamp]:
    """signal_name → live_date. Signals without one never count as live."""
    df = pd.read_sql(
        "SELECT signal_name, live_date FROM mtx_signal_config WHERE enabled", _engine()
    )
    return {r.signal_name: pd.Timestamp(r.live_date)
            for r in df.itertuples() if pd.notna(r.live_date)}


def aggregate_position(pos_mat: pd.DataFrame,
                       live_dates: dict[str, pd.Timestamp] | None = None,
                       ) -> pd.Series:
    """Equal-weight mean across signals — the tracker's composite rule.

    With `live_dates`, a signal contributes only from its own live_date
    onward, so the curve reflects the book as it was actually staffed on
    each date rather than as it looks in hindsight.
    """
    mat = pos_mat.copy()
    if live_dates is not None:
        for col in mat.columns:
            ld = live_dates.get(col)
            if ld is None:
                mat[col] = np.nan
            else:
                mat.loc[mat.index < ld, col] = np.nan
    return mat.mean(axis=1).rename("agg_position")


# ─────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────

def to_contracts(agg_pos: pd.Series, max_contracts: int = 5,
                 band: float = 0.0) -> pd.Series:
    """Fractional book position → integer MTX contracts.

    `band` is a no-trade threshold in CONTRACTS: the book only re-trades
    when the target differs from the current holding by more than `band`.
    With a 5-contract cap and an aggregate that mostly sits inside |0.4|,
    a band of 0.5-1.0 is what stops the book flipping 1 lot back and forth
    on noise that the fractional P&L never charged for.
    """
    target = (agg_pos * max_contracts).astype(float)
    held = np.zeros(len(target))
    cur = 0.0
    tvals = target.to_numpy()
    for i, t in enumerate(tvals):
        if np.isnan(t):
            held[i] = cur
            continue
        # Hysteresis on the UNROUNDED target. Comparing the rounded target
        # instead would make any band < 1.0 a no-op, since rounded targets
        # are already whole numbers.
        if abs(t - cur) > band:
            cur = float(np.clip(np.round(t), -max_contracts, max_contracts))
        held[i] = cur
    return pd.Series(held, index=target.index, name="contracts")


def simulate(agg_pos: pd.Series,
             prices: pd.DataFrame,
             scheme: str,
             max_contracts: int = 5,
             band: float = 0.0,
             costs: Costs | None = None,
             integer: bool = True) -> pd.DataFrame:
    """Run one book and return a per-date frame in TWD.

    The position on date t is the one HELD INTO date t, i.e. established at
    the previous date's fill and carried across the window that ends at t.

    Columns: contracts, ret, gross_twd, cost_twd, pnl_twd, legs.
    """
    costs = costs or Costs()
    field, _ = SCHEMES[scheme]
    ret = scheme_returns(prices, scheme)

    idx = ret.index.intersection(agg_pos.index)
    agg = agg_pos.reindex(idx)
    ret = ret.reindex(idx)
    px = prices[f"{field}_px"].reindex(idx)
    is_roll = prices["is_roll"].reindex(idx).fillna(False)

    if integer:
        size = to_contracts(agg, max_contracts, band)
    else:
        size = (agg * max_contracts).rename("contracts")

    # `size` is ALREADY the position held across window t: mtx_signal_values
    # .position is `signed_signal.shift(variant.shift_days)`, which the runner
    # pairs directly with ret[t] (no further shift) in compute_variant_pnl.
    # Applying another .shift(1) here would silently add a third day of lag
    # and understate the strategy.
    held = size

    px_prev = prices[f"{field}_px_prev"].reindex(idx)
    # P&L in points × big-point value. Uses the same-expiry price pair, so
    # no calendar spread leaks in on roll days.
    gross = (held * (px - px_prev) * CONTRACT_MULT).rename("gross_twd")

    # The trade that establishes held[t] happens at the fill that OPENS
    # window t, so it is priced at px_prev, not px. A roll adds a full round
    # trip on the whole book.
    delta = size.diff().abs().fillna(size.abs())
    roll_legs = np.where(is_roll, 2.0 * size.shift(1).abs().fillna(0.0), 0.0)
    legs = (delta + roll_legs).rename("legs")

    cost = (legs * costs.per_side_twd(px_prev)).rename("cost_twd")

    out = pd.DataFrame({
        "contracts": size,
        "held": held,
        "ret": ret,
        "gross_twd": gross,
        "cost_twd": cost,
        "legs": legs,
    })
    out["pnl_twd"] = out["gross_twd"].fillna(0.0) - out["cost_twd"].fillna(0.0)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────

def stored_pnl(scheme: str, live_dates: dict | None = None) -> pd.Series:
    """The tracker's own number: equal-weight mean of stored `pnl_1d`.

    This is rung 0 of `ladder()` — fractional positions, RAW (roll-unadjusted)
    returns and the runner's cost stub. Reproduced here so the ladder starts
    from the figure actually on the webpage rather than a re-derivation of it.
    """
    _, variant = SCHEMES[scheme]
    q = """
        SELECT v.date, v.signal_name, v.pnl_1d
          FROM mtx_signal_values v
          JOIN mtx_signal_config c USING (signal_name)
         WHERE v.variant = %(var)s AND c.enabled
    """
    df = pd.read_sql(q, _engine(), params={"var": variant})
    df["date"] = pd.to_datetime(df["date"])
    mat = df.pivot(index="date", columns="signal_name", values="pnl_1d").sort_index()
    if live_dates is not None:
        for col in mat.columns:
            ld = live_dates.get(col)
            if ld is None:
                mat[col] = np.nan
            else:
                mat.loc[mat.index < ld, col] = np.nan
    return mat.mean(axis=1).rename("stored_pnl")


def ladder(scheme: str, max_contracts: int = 5, band: float = 0.0,
           start: str | None = None) -> pd.DataFrame:
    """Turn on one realism assumption at a time and report the Sharpe cost.

    Rungs
    -----
    0 as-published    stored pnl_1d: fractional, raw returns, cost stub
    1 +executable lag drop the shift(1) override 5 US signals use for o2o —
                      it assumes a 06:30 TPE compute that does not exist
    2 +roll-adjusted  returns computed on ONE expiry per window
    3 +TWD sizing     same fractional position, but P&L in points x 50 TWD
    4 +integer lots   position rounded to whole MTX contracts, capped
    5 +real costs     commission + 期交稅 + 1 index point of slippage

    Rungs are cumulative, so each row's delta is that assumption's price.

    Rung 1 is a no-op for scheme='close' (c2c has no shift overrides), and
    rung 3 changes no trading decision at all. Both exist to stop a delta
    being credited to the wrong cause. Rung 3 matters more than it looks:
    a percentage-return series has roughly constant volatility, while a TWD
    series scales with the index — and MTX went from ~4,000 to ~45,000 over
    this history, so early years barely register in TWD. Without this rung
    the unit change hides inside the integer-lot delta and makes rounding
    look like it IMPROVES Sharpe, which it does not.
    """
    prices = contract_prices(load_price_panel(), roll_lead_days=0)
    pos = aggregate_position(load_positions(scheme))
    ret = scheme_returns(prices, scheme)
    field, _ = SCHEMES[scheme]
    px = prices[f"{field}_px"]

    def _sr(p):
        p = p.dropna()
        if start: p = p.loc[p.index >= start]
        return float(np.sqrt(252) * p.mean() / p.std()) if len(p) > 50 and p.std() else np.nan

    # rung 0 — exactly what the tracker publishes
    r0 = stored_pnl(scheme)

    stub_pct = 20.0 / (px * 50.0) + 0.00002
    raw_ret = (px / px.shift(1) - 1)   # roll-UNadjusted, as the runner uses

    # rung 1 — executable lag, still raw returns. `pos` already uses the
    # shift(2) path (us_morning_run defaults False), so this isolates the
    # lag change from the roll fix that follows.
    # No .shift() here either — `pos` is already the held position (see
    # simulate()). This mirrors compute_variant_pnl exactly, so rung 1 should
    # land just ABOVE rung 0: averaging positions before costing lets
    # opposing sleeves net their trades, which averaging per-signal P&L
    # cannot. That reconciliation is the check that the lag is right.
    p1 = pos * raw_ret - pos.diff().abs().fillna(0) * stub_pct
    # rung 2 — same positions, roll-adjusted returns
    p2 = pos * ret - pos.diff().abs().fillna(0) * stub_pct
    # rung 3 — TWD, still fractional contracts (no trading decision changes)
    r3 = simulate(pos, prices, scheme, max_contracts, band, STUB_COSTS, integer=False)
    # rungs 4 and 5 — integer contracts
    r4 = simulate(pos, prices, scheme, max_contracts, band, STUB_COSTS, integer=True)
    r5 = simulate(pos, prices, scheme, max_contracts, band, Costs(), integer=True)

    rows = [
        {"rung": "0 as-published",    "SR": _sr(r0)},
        {"rung": "1 +executable lag", "SR": _sr(p1)},
        {"rung": "2 +roll-adjusted",  "SR": _sr(p2)},
        {"rung": "3 +TWD sizing",     "SR": _sr(r3["pnl_twd"])},
        {"rung": "4 +integer lots",   "SR": _sr(r4["pnl_twd"])},
        {"rung": "5 +real costs",     "SR": _sr(r5["pnl_twd"])},
    ]
    out = pd.DataFrame(rows).set_index("rung")
    out["delta"] = out["SR"].diff()
    return out


def stats(pnl: pd.Series, capital: float | None = None) -> dict:
    """SR / ann return / vol / max drawdown for a daily TWD P&L series."""
    a = pnl.dropna()
    if len(a) < 5:
        return {}
    mu, sd = a.mean(), a.std()
    cum = a.cumsum()
    dd = cum - cum.cummax()
    out = {
        "n_days":     int(len(a)),
        "SR":         float(np.sqrt(252) * mu / sd) if sd else np.nan,
        "ann_twd":    float(252 * mu),
        "ann_vol":    float(np.sqrt(252) * sd),
        "total_twd":  float(a.sum()),
        "max_dd_twd": float(dd.min()),
    }
    if capital:
        out["ann_pct"] = 100.0 * out["ann_twd"] / capital
        out["max_dd_pct"] = 100.0 * out["max_dd_twd"] / capital
    return out
