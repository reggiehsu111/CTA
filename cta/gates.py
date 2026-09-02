"""
gates.py — the MTX house gates, in one place.

The four house gates (`mtx/best_signals_aggregate.ipynb`) were copy-pasted into
every sweep script in `signal_zoo/` as four inline comparisons. That is how the
QNT-100 defect survived: the beta rule was written as `df.beta.abs() < 0.15` in
eight places, so there was nowhere to fix it once. This module is that one
place; the rule it applies is the house rule, unchanged.

**The measured defect (QNT-100, QNT-99 Part A3).** `beta` is measured on the
realised PnL, so it scales with how much of the time the book is actually in
the market. The MTX night session has an unconditional buy-and-hold
`SR_net +1.146` (`SR_of_SR 1.764`, `positive_years 1.00`, `beta 1.00`, 2010-,
real costs). Buy-and-hold is stopped by `|beta| < 0.15` at full size — but a
long-only mask over 5-10% of nights keeps the drift while shrinking beta into
the pass region, and `SR_of_SR` / `positive_years` do not shrink with it.
Purely RANDOM long-night masks carrying no information at all pass all four
gates 10-15% of the time at 5-10% exposure. That measurement stands.

**The decision (Reggie, 2026-09-02): REJECTED.** The proposed replacement was
`|beta| / mean|exec_w| < 0.15` plus an exposure floor `mean|exec_w| >= 0.30`.
It was rejected because the two halves are not separable in practice: below
`mean|exec_w| ~ 0.31` the ratio's own standard error exceeds the threshold
(`sd(beta_per_w) ~ 0.084 / sqrt(mean|exec_w|)`), so the only thing that closes
the leak at sparse exposure is removing sparse books from the board wholesale —
and the house does not want to stop building low-exposure signals.

So:

* **The house beta gate remains `|beta| < 0.15`.** `beta_mode` defaults to
  `"raw"`, and no sweep script's threshold was changed.
* `beta_per_w = beta / mean|exec_w|` and `mean_abs_w` are **reported
  diagnostics, not gates** — `cta.signal_stats` emits both on every book. Read
  them: a survivor with `|beta|` 0.04 and `beta_per_w` 0.60 is a scaled-down
  long index bet that the gate cannot see, and that is a fact about the result,
  not about the rule. Quote them beside any sparse / event-triggered survivor.
* `beta_mode="per_w"` / `"both"` and the `min_abs_w` / `min_held_pct` floors
  stay available for a diagnostic pass (that is how QNT-99's 70 calendar
  "passers" were shown to be the leak), and are opt-in only.

Entry point
-----------
    df = cta.house_gates(df)                     # house rule: |beta| < 0.15
    df = cta.house_gates(df, beta_mode="both")   # + the diagnostic ratio

`beta_per_w` comes from `cta.signal_stats`; sweep harnesses that roll their own
stats need only add `abs_exec_w = mean|exec_w|` and this module will derive it.

Reference: QNT-100, QNT-99 Part A3, `signal_zoo/qnt100/QNT100_FINDINGS.md`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The house thresholds. The beta rule is the pre-existing one: QNT-100's
# per-exposure replacement was rejected on 2026-09-02 (see module docstring).
SR_OF_SR_MIN       = 0.6
POSITIVE_YEARS_MIN = 0.65
BETA_MAX           = 0.15
N_YEARS_MIN        = 5

# The same 0.15, applied to `beta_per_w` in the opt-in diagnostic modes. Kept
# under its own name so a future revisit can move one without the other.
BETA_PER_W_MAX     = BETA_MAX

# Optional, opt-in floors. BOTH DEFAULT TO None AND SHOULD STAY THERE: the
# QNT-100 proposal to make `mean|exec_w| >= 0.30` a house gate was rejected —
# the house builds sparse books deliberately and will not drop them wholesale.
#
# The number below is still worth knowing when reading a sparse result. Fitted
# on books whose true beta is zero by construction (QNT-100 parts B and E,
# night leg, ~4,100 bars, exposures 0.02-1.00):
#
#     sd(beta_per_w) ~ 0.084 / sqrt(mean|exec_w|)
#
#   exposure 1.00 -> sd 0.084 (= SE(beta): the ratio is exactly as noisy as
#                    plain beta for a fully-invested book)
#   exposure 0.31 -> sd 0.151  == the 0.15 threshold
#   exposure 0.10 -> sd 0.266  -> a genuinely NEUTRAL book reads > 0.15 57% of
#                    the time by estimation error alone
#   exposure 0.05 -> sd 0.376  -> ... 69% of the time
#
# Below ~0.31 exposure NO beta threshold — raw or per-w — separates a
# directional book from a neutral one. Under the house rule that limitation is
# explicit rather than fixed: at low exposure a passing beta gate is not
# evidence of neutrality, so report `beta_per_w` and `mean_abs_w` and say which
# side of 0.31 the book sits on.
ABS_W_MIN    = None        # rejected as a house gate, 2026-09-02
HELD_PCT_MIN = None
MIN_ABS_W_MEASURABLE = 0.31   # where sd(beta_per_w) == BETA_MAX

_ALIASES = {
    "SR_of_SR":       ("SR_of_SR", "sr_of_sr", "full_SR_of_SR"),
    "positive_years": ("positive_years", "positive_years_pct", "pos_years", "full_positive_years"),
    "n_years":        ("n_years", "full_n_years"),
    "beta":           ("beta", "full_beta"),
    "abs_w":          ("mean_abs_w", "abs_exec_w", "mean_abs_exec_w", "full_abs_exec_w",
                       "full_mean_abs_w"),
    "beta_per_w":     ("beta_per_w", "full_beta_per_w"),
    "held_pct":       ("held_pct", "full_held_pct"),
}


def _col(df: pd.DataFrame, key: str, prefix: str = "") -> pd.Series | None:
    """Resolve one logical metric to a column, tolerating the sweep prefixes."""
    for name in _ALIASES[key]:
        for cand in (f"{prefix}{name}", name):
            if cand in df.columns:
                return df[cand]
    return None


def beta_per_w(
    beta,
    mean_abs_w,
    min_abs_w: float = 0.01,
    max_abs_w: float = 1.0,
):
    """`beta / mean|exec_w|` — beta at unit exposure. NaN when the book is flat.

    Scale-invariant by construction: halving every position halves both the
    numerator and the denominator. An always-long book reads 1.0 whether it is
    on 100% or 5% of the time; a genuinely two-sided book reads ~0 either way.

    Two guards, because a ratio can be gamed from either end:

    * `min_abs_w` — below it the book is effectively flat and both numerator
      and denominator are noise; return NaN (which fails the gate).
    * `max_abs_w` — MTX positions are bounded in [-1, +1] by house convention,
      so `mean|exec_w| > 1` means an unnormalised signal. A denominator above 1
      can only *shrink* the ratio, i.e. it is the same scale-evasion running the
      other way (lever up, divide the beta away). Measured on the QNT-99 put-OI
      grid: 135 of 1,584 cells had `mean|exec_w|` up to 1.0e5. Clipping the
      denominator at 1.0 makes those cells fall back to plain `|beta|` — the
      pre-QNT-100 rule, which is the conservative answer for a book whose
      exposure unit is undefined.
    """
    b = np.asarray(beta, dtype=float)
    w = np.asarray(mean_abs_w, dtype=float)
    w = np.clip(w, None, max_abs_w)
    out = np.where(np.isfinite(w) & (w >= min_abs_w), b / np.where(w == 0, np.nan, w), np.nan)
    return out if out.ndim else float(out)


def house_gates(
    df: pd.DataFrame,
    beta_mode: str = "raw",
    min_abs_w: float | None = ABS_W_MIN,
    min_held_pct: float | None = HELD_PCT_MIN,
    prefix: str = "",
) -> pd.DataFrame:
    """Add the four house-gate booleans, `n_gates` and `passes` to a stats frame.

    Parameters
    ----------
    df : any frame with the house metrics. Column names may carry a sweep
        prefix (`full_beta`, `full_SR_of_SR`, ...) — see `_ALIASES`.
    beta_mode :
        ``"raw"``    (default) gate on `|beta|` — THE HOUSE RULE.
        ``"per_w"``  gate on `beta_per_w` instead — diagnostic only; the
                     QNT-100 proposal to make this the house rule was rejected
                     on 2026-09-02. Use it to ask "is this survivor a
                     scaled-down index bet?", not to declare a pass/fail.
        ``"both"``   require both; adds `gate_beta_raw` alongside. This is the
                     usual diagnostic call — it shows what the house rule says
                     and what the exposure-adjusted view says side by side.
    min_abs_w : if set, also require `mean|exec_w| >= min_abs_w`. Adds
        `gate_exposure`. **Not a house gate** — rejected as one. As a
        diagnostic, `MIN_ABS_W_MEASURABLE` (0.31) is the exposure below which
        `sd(beta_per_w)` exceeds 0.15, i.e. below which no beta rule of any
        kind separates a directional book from a neutral one.
    min_held_pct : if set, also require `held_pct >= min_held_pct` (as a
        percentage, 0-100). Adds `gate_held`. Also not a house gate, and note
        that a book can be held every bar at 5% size, so `held_pct` does not
        measure exposure — `min_abs_w` is the one that does.
    prefix : column-name prefix to try first, e.g. ``"full_"``.

    Returns a copy; never mutates the input. `beta_per_w` is derived from
    `beta` and `mean|exec_w|` when not already present, and the derived column
    is written back so the caller can report it.
    """
    if beta_mode not in ("per_w", "raw", "both"):
        raise ValueError(f"beta_mode must be per_w/raw/both, got {beta_mode!r}")
    out = df.copy()

    srsr  = _col(out, "SR_of_SR", prefix)
    posyr = _col(out, "positive_years", prefix)
    nyr   = _col(out, "n_years", prefix)
    beta  = _col(out, "beta", prefix)
    for label, s in (("SR_of_SR", srsr), ("positive_years", posyr),
                     ("n_years", nyr), ("beta", beta)):
        if s is None:
            raise KeyError(f"house_gates: no column for {label!r} in {list(out.columns)[:20]}")

    bpw = _col(out, "beta_per_w", prefix)
    if bpw is None and beta_mode in ("per_w", "both"):
        absw = _col(out, "abs_w", prefix)
        if absw is None:
            raise KeyError(
                "house_gates: beta_mode='per_w' needs mean|exec_w| (column "
                f"`mean_abs_w`/`abs_exec_w`) or a precomputed `beta_per_w`; "
                f"found neither in {list(out.columns)[:20]}. beta_mode='raw' "
                "(the house rule, and the default) needs neither."
            )
        bpw = pd.Series(beta_per_w(beta, absw), index=out.index)
        out["beta_per_w"] = bpw.round(3)

    out["gate_srsr"]  = srsr  > SR_OF_SR_MIN
    out["gate_posyr"] = posyr >= POSITIVE_YEARS_MIN
    out["gate_nyr"]   = nyr   >= N_YEARS_MIN
    if beta_mode == "raw":
        out["gate_beta"] = beta.abs() < BETA_MAX
    else:
        out["gate_beta_per_w"] = bpw.abs() < BETA_PER_W_MAX
        out["gate_beta"] = out["gate_beta_per_w"]
        if beta_mode == "both":
            # `passes` here is the INTERSECTION, i.e. stricter than the house
            # rule — that is what the caller asked for by saying "both". Read
            # `gate_beta_raw` alone for the house verdict.
            out["gate_beta_raw"] = beta.abs() < BETA_MAX
            out["gate_beta"] = out["gate_beta_per_w"] & out["gate_beta_raw"]

    cols = ["gate_srsr", "gate_posyr", "gate_beta", "gate_nyr"]
    if min_abs_w is not None:
        absw = _col(out, "abs_w", prefix)
        if absw is None:
            raise KeyError("house_gates: min_abs_w set but no mean|exec_w| column")
        out["gate_exposure"] = absw >= min_abs_w
        cols.append("gate_exposure")
    if min_held_pct is not None:
        held = _col(out, "held_pct", prefix)
        if held is None:
            raise KeyError("house_gates: min_held_pct set but no `held_pct` column")
        out["gate_held"] = held >= min_held_pct
        cols.append("gate_held")

    # NaN in any metric is a failure, not a pass: `>`/`>=` on NaN is already
    # False, but `.abs() < x` on NaN is False too, so this is only defensive.
    out[cols] = out[cols].fillna(False).astype(bool)
    out["n_gates"] = out[cols].sum(axis=1)
    out["passes"]  = out[cols].all(axis=1)
    return out


__all__ = ["house_gates", "beta_per_w", "SR_OF_SR_MIN", "POSITIVE_YEARS_MIN",
           "BETA_MAX", "BETA_PER_W_MAX", "N_YEARS_MIN", "ABS_W_MIN",
           "HELD_PCT_MIN", "MIN_ABS_W_MEASURABLE"]
