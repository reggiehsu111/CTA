"""sweep_report.py — the QNT-25 reporting line for transform x window sweeps.

Every MTX factor sweep in `signal_zoo/` has the same shape: `S` source series
x 6 transforms x 3 windows. QNT-25 measured that those 18 transform-window
cells are worth **~1.5 independent tests** (eigenvalue n_eff 1.42-1.64, mean
pairwise PnL correlation 0.61-0.82, PC1 77-83%), and that `sd(SR)` across cells
on every macro grid (0.139-0.154) sits *below* `SE(SR | 25y) = 0.20`.

Two consequences, which this module exists to make automatic:

1. A headline computed at cell level uses an `n` that is ~10x too large. Collapse
   to one number per **source series** before any test, and quote the per-series
   `n`, never the cell count.
2. When cross-cell dispersion is under the noise floor, the ranking of cells is a
   ranking of estimation error. No "best cell" may be quoted as a result.

Entry points
------------
* `sweep_headline(df, value, ...)`   — one honest line for a single-leg grid
* `paired_headline(a, b, ...)`       — one honest line for a paired dSR grid,
                                       which needs its own (larger) SE
* `se_sr(sr, n_years)`               — Lo (2002) iid SE of an annualised Sharpe
* `icc_neff(df, value, group)`       — ICC + design-effect n_eff

Both headline functions return a `Headline` dataclass whose `str()` is the line;
`.detail()` adds the per-rule breakdown. Nothing here re-runs a sweep, selects a
sign, or writes anything — it only re-describes numbers already computed.

Reference: `signal_zoo/QNT25_REPORTING_STANDARD.md` (the six rules), QNT-25.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# scipy is imported lazily, inside the two functions that use it.
# QNT-50: cta/__init__.py imports this module, so a module-level `from scipy
# import stats` makes scipy a hard dependency of `import cta` - including in the
# mtx-signal-runner Lambda, whose handler imports cta at module scope and which
# has no need of the sweep helpers. scipy is ~90 MB in a Lambda package.


# Cells whose dispersion is below the noise floor cannot be ranked; these are the
# cut-offs QNT-25 used when phrasing the verdict.
_RATIO_BELOW = 1.0
_RATIO_ABOUT = 1.5


def se_sr(sr, n_years) -> np.ndarray | float:
    """SE of an annualised Sharpe estimated over `n_years` years (Lo 2002, iid).

    ~0.20 at 25 years for a small Sharpe. This is the noise floor that every
    sweep dispersion must be compared against.
    """
    return np.sqrt((1.0 + 0.5 * np.asarray(sr, dtype=float) ** 2) / np.asarray(n_years, dtype=float))


def icc_neff(df: pd.DataFrame, value: str, group: str) -> tuple[float, float, float]:
    """One-way random-effects ICC and design-effect `n_eff` for a sweep grid.

    Returns `(icc, cells_per_group, n_eff)`. `n_eff = N / (1 + (k-1) * ICC)`:
    with ICC 0.6 and 18 cells per series, 522 cells are ~46 tests, not 522.

    Uses Sokal-Rohlf's `k0` for unbalanced groups; `k0 == mean(k)` when the grid
    is balanced, so the QNT-25 numbers reproduce exactly.
    """
    d = df[[value, group]].dropna()
    g = d.groupby(group)[value]
    n_g, sizes = g.ngroups, g.size()
    if n_g < 2:                        # everything in one group: one effective test
        return float("nan"), float(sizes.mean()) if len(sizes) else float("nan"), float(n_g)
    k0 = (len(d) - (sizes ** 2).sum() / len(d)) / (n_g - 1)
    if k0 <= 1.0:                      # one cell per group: a cell IS a series, ICC undefined
        return float("nan"), float(k0), float(len(d))
    grand = d[value].mean()
    msb = (sizes * (g.mean() - grand) ** 2).sum() / (n_g - 1)
    msw_den = len(d) - n_g
    msw = ((d[value] - g.transform("mean")) ** 2).sum() / msw_den if msw_den > 0 else 0.0
    r = 1.0 if msw <= 0 else float(np.clip((msb - msw) / (msb + (k0 - 1) * msw), 0.0, 1.0))
    return r, float(k0), float(len(d) / (1.0 + (k0 - 1) * r))


def _wilcoxon(x: pd.Series) -> float:
    """Wilcoxon signed-rank p, NaN rather than an exception on a degenerate sample."""
    try:
        from scipy import stats
        return float(stats.wilcoxon(x).pvalue)
    except ValueError:
        return float("nan")


def _verdict(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "noise floor not computable"
    if ratio < _RATIO_BELOW:
        return "dispersion BELOW the noise floor - ranking cells ranks noise"
    if ratio < _RATIO_ABOUT:
        return "dispersion ~ the noise floor - ranking is mostly noise"
    return "dispersion exceeds the noise floor"


@dataclass
class Headline:
    """The QNT-25 line, plus every number in it, for programmatic use."""
    line: str
    detail_lines: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __str__(self) -> str:            # so `print(sweep_headline(...))` just works
        return self.line

    def __repr__(self) -> str:
        return self.line

    def detail(self) -> str:
        return "\n".join([self.line, *self.detail_lines])

    def print(self) -> "Headline":
        print(self.detail())
        return self


def sweep_headline(
    df: pd.DataFrame,
    value: str = "SR_net",
    series_col: str = "series",
    n_years=None,
    label: str = "grid",
    cells_per_series: int | None = None,
) -> Headline:
    """The standard one-line headline for a `transform x window` sweep.

    > `<label>`: N cells = S source series x C transform-windows, n_eff ~ E
    > (ICC = r). Per-series median X, k/S positive, Wilcoxon p = p.
    > sd(value) across cells = a vs SE(SR | Y years) = b (ratio a/b) -> verdict.

    Parameters
    ----------
    df : the FULL sweep, one row per cell — not the gated or top-N slice.
    value : column to headline (`SR_net`, `SR_full`, `SR_of_SR`, ...).
    series_col : the SOURCE-SERIES column. If absent, every row is treated as its
        own series and the line says so — that is the honest reading for a
        registered-signal grid where a cell already *is* a series.
    n_years : years of history behind one cell. Defaults to the median of the
        `n_years` column, else 25.
    cells_per_series : cosmetic override for the "x C" term.

    Notes
    -----
    The noise floor is the mean per-cell `SE(SR)`, matching QNT-25's `noise_floor`.
    A `value` that is not a Sharpe (`positive_years`, `beta`) still gets the ICC
    and per-series test; read the sd-vs-SE clause as Sharpe-specific.
    """
    d = df.dropna(subset=[value]).copy()
    n_dropped = len(df) - len(d)
    if not len(d):
        return Headline(f"{label}: no finite `{value}` cells — nothing to report.")

    if n_years is None:
        yrs = d["n_years"] if "n_years" in d.columns else 25.0
    else:
        yrs = n_years
    n_years = float(np.median(np.asarray(yrs, dtype=float)))   # display only

    per_cell = d[series_col] if series_col in d.columns else pd.Series(d.index, index=d.index)
    has_series = series_col in d.columns
    d["_series"] = per_cell.values

    r, k0, neff = icc_neff(d, value, "_series")
    per = d.groupby("_series")[value].median()
    n_series = int(per.size)
    k = cells_per_series if cells_per_series is not None else k0

    sd = float(d[value].std(ddof=1)) if len(d) > 1 else float("nan")
    se = float(np.mean(se_sr(d[value], yrs)))          # per-cell SE, then averaged
    ratio = sd / se if se > 0 else float("nan")
    pos = int((per > 0).sum())
    wp = _wilcoxon(per)

    if k <= 1.0:
        shape = f"{len(d)} cells = {n_series} source series (a cell IS a series, so n was already honest)"
    else:
        if has_series and np.isfinite(r):
            icc_txt = f"ICC({series_col}) = {r:.2f}"
        else:
            icc_txt = "ICC n/a — a single series" if n_series < 2 else "no series grouping"
        shape = (f"{len(d)} cells = {n_series} source series x {k:.0f} transform-windows, "
                 f"n_eff ~ {neff:.0f} ({icc_txt})")
    line = (
        f"{label}: {shape}. "
        f"Per-series median {value} {per.median():+.3f}, {pos}/{n_series} positive, "
        f"Wilcoxon p = {wp:.3f}. "
        f"sd({value}) across cells = {sd:.3f} vs SE(SR | {n_years:.0f}y) = {se:.3f} "
        f"(ratio {ratio:.2f}) -> {_verdict(ratio)}."
    )

    detail = []
    if not has_series:
        detail.append(f"  note: no `{series_col}` column — each cell counted as its own series (n_eff = n).")
    if n_dropped:
        detail.append(f"  note: {n_dropped} of {len(df)} rows dropped for non-finite `{value}`.")
    best = d.loc[d[value].idxmax()]
    from scipy import stats
    exp_max = d[value].mean() + sd * stats.norm.ppf(1 - 1 / (neff + 1)) if neff > 0 else float("nan")
    detail.append(
        f"  best cell {value} = {best[value]:+.3f}"
        + (f" ({best.get('cand', best.get('signal', ''))})" if ("cand" in d.columns or "signal" in d.columns) else "")
        + f"; expected max of {neff:.0f} independent draws ~ {exp_max:+.3f}"
        + ("  -> the best cell is what a null grid this size produces." if best[value] <= exp_max
           else "  -> above the null best-of-n, but still search-corrected before quoting.")
    )
    if ratio < _RATIO_BELOW:
        detail.append("  rule 2: dispersion is under the noise floor — do NOT quote a best cell as a result.")

    # QNT-100: the house beta gate is `|beta| < 0.15` measured on realised PnL,
    # so it shrinks with exposure — a long-only mask over 5-10% of nights
    # inherits the MTX night drift (buy-and-hold SR_net +1.146) and passes all
    # four gates 10-15% of the time carrying no information at all. The gate was
    # left as it is (Reggie, 2026-09-02: the house builds sparse books on
    # purpose), so the check moves here, into the line that any positive sweep
    # claim has to carry.
    from .gates import _col as _gate_col, beta_per_w as _beta_per_w, MIN_ABS_W_MEASURABLE
    _absw = _gate_col(d, "abs_w")
    if _absw is not None:
        _bpw = _gate_col(d, "beta_per_w")
        if _bpw is None:
            _b = _gate_col(d, "beta")
            _bpw = pd.Series(_beta_per_w(_b, _absw), index=d.index) if _b is not None else None
        txt = f"  exposure: median mean|exec_w| = {float(_absw.median()):.2f}"
        if _bpw is not None:
            txt += f", median |beta per unit exposure| = {float(_bpw.abs().median()):.2f}"
        n_sparse = int((_absw < MIN_ABS_W_MEASURABLE).sum())
        if n_sparse:
            txt += (f"; {n_sparse}/{len(d)} cells sit below {MIN_ABS_W_MEASURABLE:.2f} exposure, where "
                    "|beta| < 0.15 cannot tell a directional book from a neutral one (QNT-100) "
                    "— quote beta_per_w and mean_abs_w for any of those you headline.")
        detail.append(txt)
    if has_series and {"SR_IS", "SR_OOS"} <= set(d.columns):
        is_med = d.groupby("_series")["SR_IS"].median().median()
        oos_med = d.groupby("_series")["SR_OOS"].median().median()
        detail.append(
            f"  rule 4: sign fitted IS — per-series median SR_IS {is_med:+.3f} vs SR_OOS {oos_med:+.3f}; "
            "an IS/full-sample SR is not evidence of edge."
        )
    if has_series and "sign_IS" in d.columns:
        mixed = int(d.groupby("_series")["sign_IS"].nunique().gt(1).sum())
        if mixed:
            detail.append(f"  {mixed}/{n_series} series disagree with themselves on sign across their own cells.")

    return Headline(line, detail, dict(
        n_cells=len(d), n_series=n_series, cells_per_series=k, icc=r, n_eff=neff,
        per_series_median=float(per.median()), n_positive=pos, wilcoxon_p=wp,
        sd_cells=sd, se_sr=se, ratio=ratio, n_years=n_years,
        best_cell=float(best[value]), expected_max_null=float(exp_max),
    ))


def paired_headline(
    a: pd.Series,
    b: pd.Series,
    series=None,
    n_years=25.0,
    label: str = "paired dSR",
    a_name: str = "treat",
    b_name: str = "base",
    sr_ref: float = 0.1,
) -> Headline:
    """Headline for a PAIRED sweep (variant A vs variant B on the same cells).

    Rule 3 of the standard: a paired dSR has its own, larger, SE —
    `SE(d) ~ SE(SR) * sqrt(2 * (1 - rho))`, with `rho` the correlation of the two
    legs across cells. On the QNT-14 day-vs-c2c grid rho ~ 0.42, so SE(d) ~ 0.22
    — three times the +0.071 that was reported as an effect.

    Parameters
    ----------
    a, b : aligned per-cell Sharpes for the two variants (a - b is the delta).
    series : per-cell source-series labels. Omit only when a cell IS a series.
    sr_ref : Sharpe at which the SE is evaluated (small, so ~sqrt(1/n_years)).
    """
    a, b = pd.Series(a).astype(float), pd.Series(b).astype(float)
    d = pd.DataFrame({"a": a.values, "b": b.values})
    d["_series"] = list(series) if series is not None else list(range(len(d)))
    d = d.dropna(subset=["a", "b"])
    if not len(d):
        return Headline(f"{label}: no paired cells — nothing to report.")
    d["delta"] = d["a"] - d["b"]
    n_years = float(n_years)

    r, k0, neff = icc_neff(d, "delta", "_series")
    per = d.groupby("_series")["delta"].median()
    n_series = int(per.size)
    rho = float(d["a"].corr(d["b"])) if len(d) > 2 else float("nan")
    se_d = float(se_sr(sr_ref, n_years) * np.sqrt(max(2.0 * (1.0 - rho), 0.0))) if np.isfinite(rho) else float("nan")
    med = float(per.median())
    pos = int((per > 0).sum())
    wp = _wilcoxon(per)
    cell_pos, cell_wp = int((d["delta"] > 0).sum()), _wilcoxon(d["delta"])

    if k0 <= 1.0:
        shape = f"{len(d)} cells = {n_series} source series (a cell IS a series, so n was already honest)"
    else:
        shape = (f"{len(d)} cells = {n_series} source series x {k0:.0f} transform-windows, "
                 f"n_eff ~ {neff:.0f} (ICC = {r:.2f})")
    line = (
        f"{label} ({a_name} - {b_name}): {shape}. "
        f"Per-series median {med:+.3f}, {pos}/{n_series} positive, Wilcoxon p = {wp:.3f}."
    )
    if np.isfinite(se_d) and se_d > 0:
        line += (f" corr({a_name},{b_name}) = {rho:.2f} -> SE(dSR | {n_years:.0f}y) = {se_d:.3f}; "
                 f"the effect is {med / se_d:+.2f} SE.")
    detail = [] if k0 <= 1.0 else [
        f"  at CELL level (the inflated reading): n={len(d)}, median {d['delta'].median():+.3f}, "
        f"{cell_pos}/{len(d)} positive, Wilcoxon p = {cell_wp:.3f} "
        f"— quote the per-series n above, not this one."
    ]  # noqa: E501
    if np.isfinite(se_d) and se_d > 0 and abs(med) < se_d:
        detail.append("  rule 3: |effect| < SE(dSR) — this is not a measured window effect.")
    return Headline(line, detail, dict(
        n_cells=len(d), n_series=n_series, cells_per_series=k0, icc=r, n_eff=neff,
        per_series_median=med, n_positive=pos, wilcoxon_p=wp,
        cell_median=float(d["delta"].median()), cell_wilcoxon_p=cell_wp,
        rho=rho, se_delta=se_d, n_years=n_years,
    ))


__all__ = ["sweep_headline", "paired_headline", "se_sr", "icc_neff", "Headline"]
