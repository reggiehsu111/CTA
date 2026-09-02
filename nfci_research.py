#!/usr/bin/env python3
"""
NFCI research script:
  1. Compare 8 normalization schemes on NFCI level + short-window delta.
  2. Event study — CAAR / CASR for tightening and loosening shocks.

Saves plots to /tmp/nfci_plots/.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cta

OUT = "/tmp/nfci_plots"
os.makedirs(OUT, exist_ok=True)

ASSET = cta.load_asset("mtx", "1d")
nfci  = cta.load_nfci_tw("nfci", ASSET.index).dropna()

print(f"MTX 1d:  {len(ASSET)} rows, {ASSET.index.min().date()} → {ASSET.index.max().date()}")
print(f"NFCI TW: {len(nfci)} rows,  {nfci.index.min().date()} → {nfci.index.max().date()}")

ret     = ASSET["close"].pct_change()
cost    = 20.0/(ASSET["close"]*50.0) + 0.00002
EVAL_ST = pd.Timestamp("2013-01-01")
EVAL_EN = ASSET.index.max()


# ══════════════════════════════════════════════════════════════════════
# Part 1 — 8 normalization schemes
# ══════════════════════════════════════════════════════════════════════

# For each raw input x (NFCI level or its N-week delta), we test 8 ways to
# collapse it into a [-1, +1] tradable signal.

def _selfz(x, w):
    mu = x.rolling(w, min_periods=w//4).mean()
    sd = x.rolling(w, min_periods=w//4).std().replace(0, np.nan)
    return (x - mu) / sd

def _robust_z(x, w):
    med = x.rolling(w, min_periods=w//4).median()
    mad = (x - med).abs().rolling(w, min_periods=w//4).median()
    return (x - med) / (1.4826 * mad.replace(0, np.nan))

def _rank_c(x, w):
    """centered rank ∈ [-1, +1]"""
    return (x.rolling(w, min_periods=w//4).apply(
        lambda a: (pd.Series(a).rank(pct=True).iloc[-1] - 0.5) * 2, raw=False))


NORMS = {
    "raw_tanh":            lambda x, w: np.tanh(x),
    "z_tanh":              lambda x, w: np.tanh(_selfz(x, w)),
    "z_winsor_c3":         lambda x, w: _selfz(x, w).clip(-3, 3) / 3,
    "z_sigmoid":           lambda x, w: 2 / (1 + np.exp(-_selfz(x, w))) - 1,
    "robust_z_tanh":       lambda x, w: np.tanh(_robust_z(x, w)),
    "rank_c":              _rank_c,
    "sign_thresh_0p5":     lambda x, w: pd.Series(
                                np.where(_selfz(x, w) >  0.5,  1.0,
                                         np.where(_selfz(x, w) < -0.5, -1.0, 0.0)),
                                index=x.index),
    "sign_thresh_1p0":     lambda x, w: pd.Series(
                                np.where(_selfz(x, w) >  1.0,  1.0,
                                         np.where(_selfz(x, w) < -1.0, -1.0, 0.0)),
                                index=x.index),
}

W = 252  # rolling normalization window

# Raw inputs to normalize
INPUTS = {
    "level":     nfci,
    "delta_1w":  nfci.diff(5),
    "delta_2w":  nfci.diff(10),
    "delta_4w":  nfci.diff(20),
}

# Sign convention: NFCI positive = tight = bearish. So the signal that
# BUYS TW when NFCI is loose (or dropping) needs a NEGATIVE sign flip.
# We test both signs and pick the one with positive SR.
def _pnl(sig):
    ex = sig.reindex(ASSET.index).shift(2)
    return (ex*ret - ex.fillna(0).diff().abs()*cost).loc[EVAL_ST:EVAL_EN].dropna()

def _sr(p):
    return float(np.sqrt(252) * p.mean() / p.std()) if len(p) > 30 and p.std() > 0 else np.nan


rows = []
for input_name, x in INPUTS.items():
    for norm_name, fn in NORMS.items():
        sig = fn(x, W)
        p_pos = _pnl(sig)
        p_neg = _pnl(-sig)
        # Pick the sign that yields positive SR
        sr_pos, sr_neg = _sr(p_pos), _sr(p_neg)
        best_sr   = max(sr_pos, sr_neg) if not (np.isnan(sr_pos) and np.isnan(sr_neg)) else np.nan
        best_sign = "+" if sr_pos >= sr_neg else "-"
        best_pnl  = p_pos if sr_pos >= sr_neg else p_neg
        rows.append({
            "input":    input_name,
            "norm":     norm_name,
            "SR_pos":   round(sr_pos, 3) if not np.isnan(sr_pos) else None,
            "SR_neg":   round(sr_neg, 3) if not np.isnan(sr_neg) else None,
            "best_SR":  round(best_sr, 3) if not np.isnan(best_sr) else None,
            "sign":     best_sign,
            "ann_ret%": round(float(best_pnl.mean()*252*100), 2) if len(best_pnl) else None,
            "max_dd%":  round(float((best_pnl.cumsum()-best_pnl.cumsum().cummax()).min()*100), 2) if len(best_pnl) else None,
        })

grid = pd.DataFrame(rows)
print("\n=== NORMALIZATION GRID (best sign, MTX c2c 2013→now) ===\n")
print(grid.to_string(index=False))


# ── Heatmap: best_SR by (input × norm) ─────────────────────────────
piv = grid.pivot(index="norm", columns="input", values="best_SR")
piv = piv[["level", "delta_1w", "delta_2w", "delta_4w"]]

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if pd.notna(v):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     color="white" if abs(v) > 0.4 else "black", fontsize=11, fontweight="bold")
ax.set_title(f"NFCI Signal SR — {len(NORMS)} normalizations × {len(INPUTS)} raw inputs\n"
             f"(best sign chosen; MTX c2c PnL 2013→now)", fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.7, label="Sharpe Ratio")
plt.tight_layout()
plt.savefig(f"{OUT}/01_normalization_heatmap.png", dpi=130, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# Part 2 — Event studies (CAAR / CASR)
# ══════════════════════════════════════════════════════════════════════

# Event definition: a WEEK where the NFCI change is > +1σ (tightening
# shock) or < -1σ (loosening shock). We use the raw weekly NFCI, then
# map to TW dates via forward-fill index.

# Build event-date list on the weekly index first
nfci_weekly = cta.load_nfci("nfci")   # Friday-dated raw values
delta_1w    = nfci_weekly.diff()
sigma       = delta_1w.rolling(52, min_periods=13).std()

tight_events = delta_1w[delta_1w >  1 * sigma].index
loose_events = delta_1w[delta_1w < -1 * sigma].index
print(f"\nTightening shocks (Δ_1w > +σ_52w):  {len(tight_events)} events "
      f"({tight_events.min().date()} → {tight_events.max().date()})")
print(f"Loosening   shocks (Δ_1w < -σ_52w):  {len(loose_events)} events")

# Filter to eval window
tight_events = tight_events[tight_events >= EVAL_ST]
loose_events = loose_events[loose_events >= EVAL_ST]
print(f"Post-2013 tightening: {len(tight_events)}, loosening: {len(loose_events)}")

# Event window on TW trading calendar
WIN_PRE, WIN_POST = 10, 30    # trading days before / after

def _event_car(event_dates, ret_series, pre, post):
    """For each event, extract return window and cumulate. Returns matrix
    n_events × (pre+post+1)."""
    car_rows = []
    tw_index = ret_series.index
    for ev in event_dates:
        # The event value is dated Friday W and becomes usable +6 days later.
        # Anchor the event WINDOW at day 0 = Thursday of week+1 (first TW
        # date the event is observable).
        exec_date = ev + pd.Timedelta(days=6)
        # Snap to next TW trading date
        pos = tw_index.searchsorted(exec_date, side="left")
        if pos < pre or pos >= len(tw_index) - post: continue
        window_idx = tw_index[pos-pre : pos+post+1]
        r_win      = ret_series.reindex(window_idx).fillna(0).values
        car_rows.append(r_win.cumsum())
    if not car_rows: return np.zeros((0, pre+post+1))
    return np.array(car_rows)

tight_car = _event_car(tight_events, ret, WIN_PRE, WIN_POST)
loose_car = _event_car(loose_events, ret, WIN_PRE, WIN_POST)

def _caar_casr(car_matrix):
    """CAAR = mean CAR across events; CASR = mean/(std/√n) at each t."""
    if car_matrix.shape[0] == 0: return None, None, None
    caar = car_matrix.mean(axis=0) * 100     # percent
    std  = car_matrix.std(axis=0) * 100
    casr = caar / (std / np.sqrt(car_matrix.shape[0]))    # t-stat
    return caar, std, casr

t_caar, t_std, t_casr = _caar_casr(tight_car)
l_caar, l_std, l_casr = _caar_casr(loose_car)

# ── Two-panel figure: CAAR + CASR for both event types ──────────────
xs = np.arange(-WIN_PRE, WIN_POST+1)
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

def _plot_car(ax, xs, car_mat, caar, std, casr, title, color):
    # Individual events (thin translucent)
    for row in car_mat:
        ax.plot(xs, row * 100, color=color, alpha=0.06, lw=0.6)
    # CAAR ± 1σ across events
    ax.fill_between(xs, caar - std/np.sqrt(len(car_mat)),
                        caar + std/np.sqrt(len(car_mat)),
                    color=color, alpha=0.2, label="±1 SE")
    ax.plot(xs, caar, color=color, lw=2.5, label=f"CAAR (N={len(car_mat)})")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.axvline(0, color="red",   lw=0.8, ls="--", alpha=0.7, label="event day")
    ax.set_xlabel("TW trading days from event"); ax.set_ylabel("CAAR (%)")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)

_plot_car(axes[0,0], xs, tight_car, t_caar, t_std, t_casr,
          f"Tightening shocks (Δ_1w > +σ_52w), N={len(tight_car)}", "#C62828")
_plot_car(axes[0,1], xs, loose_car, l_caar, l_std, l_casr,
          f"Loosening shocks (Δ_1w < -σ_52w), N={len(loose_car)}",  "#2E7D32")

def _plot_casr(ax, xs, casr, title, color):
    ax.plot(xs, casr, color=color, lw=2, marker="o", ms=3)
    ax.axhline( 0,    color="black", lw=0.5, alpha=0.5)
    ax.axhline( 1.96, color="grey",  lw=0.8, ls=":",  alpha=0.6, label="p=0.05")
    ax.axhline(-1.96, color="grey",  lw=0.8, ls=":",  alpha=0.6)
    ax.axvline( 0,    color="red",   lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("TW trading days from event"); ax.set_ylabel("CASR (t-stat)")
    ax.set_title(title); ax.legend(fontsize=9); ax.grid(alpha=0.3)

_plot_casr(axes[1,0], xs, t_casr,
           "Tightening shocks — CASR (statistical significance)", "#C62828")
_plot_casr(axes[1,1], xs, l_casr,
           "Loosening shocks — CASR (statistical significance)", "#2E7D32")

plt.suptitle("NFCI shock event study on MTX 1d — CAAR + CASR", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(f"{OUT}/02_event_study_caar_casr.png", dpi=130, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# Part 3 — Best signal cumulative PnL (visual proof)
# ══════════════════════════════════════════════════════════════════════

# Take the top-3 signals from the grid and plot their cumulative PnL
top3 = grid.dropna(subset=["best_SR"]).sort_values("best_SR", ascending=False).head(3)
print(f"\n=== TOP 3 SIGNAL CONFIGS ===\n{top3.to_string(index=False)}")

fig, ax = plt.subplots(figsize=(12, 5))
for _, row in top3.iterrows():
    inp = INPUTS[row["input"]]
    fn  = NORMS[row["norm"]]
    sig = fn(inp, W) * (1 if row["sign"] == "+" else -1)
    p = _pnl(sig)
    ax.plot(p.cumsum().index, p.cumsum().values * 100, lw=1.6,
            label=f"{row['input']} · {row['norm']} · {row['sign']}  (SR {row['best_SR']:+.2f})")

ax.axhline(0, color="black", lw=0.5, alpha=0.5)
ax.set_ylabel("cum PnL (%)"); ax.set_xlabel("date")
ax.set_title("Top-3 NFCI signal cumulative PnL — MTX c2c, 2013 → now", fontsize=12)
ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/03_top3_cum_pnl.png", dpi=130, bbox_inches="tight")
plt.close()

print(f"\n✅ Plots saved to {OUT}/")
print(f"    01_normalization_heatmap.png")
print(f"    02_event_study_caar_casr.png")
print(f"    03_top3_cum_pnl.png")
