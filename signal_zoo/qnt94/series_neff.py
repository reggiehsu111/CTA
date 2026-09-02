"""QNT-94: n_eff ACROSS source series, measured on the per-series SR statistic.

QNT-78 Rule 2 states a power target in raw `S` (the count of source series).
The series are not independent, so `S` is an upper bound on power. QNT-78
measured n_eff of the raw INPUTS (11 -> 4.29, 26 -> 6.75); the number the rule
actually needs is n_eff of the per-series SR *statistic*, which needs PnLs.

This rebuilds the per-cell net PnL streams of the two macro grids already on
disk (QNT-12's 29x18 c2c grid and QNT-14's 11x18x3 window grid), collapses each
source series to ONE PnL (equal-weight across its 18 transform-windows, and
separately its median-SR cell), and measures across-series redundancy:
  * mean / mean-abs pairwise PnL correlation
  * eigenvalue  n_eff = (sum L)^2 / sum L^2
  * design-effect n_eff = S / (1 + (S-1) * rho_bar)   <- the one the power
    formula wants, since d_min is about SE of a MEAN across series
Then restates the QNT-32/QNT-78 power table in n_eff terms.

Read-only w.r.t. the DB. No sign is chosen for live use (signs are frozen on
the in-sample half exactly as the original grids did), no config is touched.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
from scipy import stats
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context
from cta.signals._base import _o2o_ret, _prev_open_cc

OUT = "/home/ubuntu/mtx/signal_zoo/qnt94"
Z = "/home/ubuntu/mtx/signal_zoo"
IS_END = "2018-12-31"
PV, FIXED, FEE, PPY = 50.0, 70.0, 0.00004, 252     # REAL costs
REAL = dict(fixed_per_side=FIXED, fee_rate=FEE)

TF = {"selfz": ops.selfz, "robustz": ops.robust_z, "bdtanh": ops.bd_selftanh,
      "rankc": ops.rank_c, "signth": ops.sign_thresh, "dev": ops.dev}
WS = (60, 120, 252)

ctx = build_context(); A = ctx.asset
_o, _c, _nc = A["open"].astype(float), A["close"].astype(float), A["night_close"].astype(float)
_RET = {"c2c": A.returns, "o2o": _o2o_ret(A), "day": _c / _o - 1, "ongap": _o / _nc - 1}
_ENTRY = {"c2c": _c, "o2o": _prev_open_cc(A), "day": _o, "ongap": _nc}
_COST = {k: FIXED / (p * PV) + FEE for k, p in _ENTRY.items()}
_SHIFT = {"c2c": 2, "o2o": 2, "day": 1, "ongap": 1}


def sr(v):
    v = pd.Series(v).dropna()
    return float(np.sqrt(PPY) * v.mean() / v.std()) if len(v) > 30 and v.std() > 0 else np.nan


def net_pnl(sig, sign, variant):
    """Net PnL stream of one (signal, sign, variant) cell, costs on its own entry price."""
    e = (sig * sign).shift(_SHIFT[variant]).reindex(A.index)
    tc = e.fillna(0).diff().abs() * _COST[variant]
    return (e * _RET[variant] - tc).dropna()


def frozen_sign(sig, variant):
    """Sign fitted on the IS half only, exactly as the source grids did."""
    e = sig.shift(_SHIFT[variant]).reindex(A.index)
    g = (e * _RET[variant]).loc[:IS_END].dropna()
    if len(g) < 60 or not np.isfinite(g.std()) or g.std() == 0:
        return None
    return -1 if float(g.mean()) < 0 else 1


def build_grid(series_ids, variant, label):
    """{series -> DataFrame of its 18 cell PnLs} for one variant."""
    out = {}
    for sid in series_ids:
        x = ctx.macro(sid).astype(float)
        cells = {}
        for tn, tf in TF.items():
            for w in WS:
                s = tf(x, w).replace([np.inf, -np.inf], np.nan)
                s = cta.normalize_signal(s, method="tanh", window=252)
                if s.dropna().empty:
                    continue
                g = frozen_sign(s, variant)
                if g is None:
                    continue
                cells[f"{tn}|w{w}"] = net_pnl(s, g, variant)
        if cells:
            out[sid] = pd.DataFrame(cells)
    print(f"  {label}/{variant}: {len(out)} series x "
          f"{np.median([d.shape[1] for d in out.values()]):.0f} cells")
    return out


def collapse(grid, how="ew"):
    """One PnL per source series."""
    cols = {}
    for sid, d in grid.items():
        if how == "ew":
            cols[sid] = d.mean(axis=1)
        else:                                   # median-SR cell
            srs = d.apply(sr)
            cols[sid] = d[srs.sort_values().index[len(srs) // 2]]
    return pd.DataFrame(cols)


def redundancy(P, label, note=""):
    """Across-series redundancy of a per-series PnL panel."""
    common = P.dropna()
    C = common.corr()
    S = C.shape[0]
    off = C.values[np.triu_indices(S, 1)]
    ev = np.linalg.eigvalsh(C.values)[::-1]
    ev = np.clip(ev, 0, None)
    neff_eig = float(ev.sum() ** 2 / (ev ** 2).sum())
    rho = float(off.mean())
    neff_deff = float(S / (1 + (S - 1) * rho)) if (1 + (S - 1) * rho) > 0 else np.nan
    # pairwise-complete version (uses each pair's full overlap, not the common window)
    Cp = P.corr()
    offp = Cp.values[np.triu_indices(S, 1)]
    evp = np.clip(np.linalg.eigvalsh(np.nan_to_num(Cp.values, nan=0.0))[::-1], 0, None)
    neff_eig_pw = float(evp.sum() ** 2 / (evp ** 2).sum())
    srs = P.apply(sr)
    print(f"\n  [{label}] S={S}  common sample {common.index.min().date()}..{common.index.max().date()}"
          f" ({len(common)/252:.1f}y of {len(P)/252:.1f}y union)  {note}")
    print(f"    pairwise PnL corr : mean {rho:+.3f}   mean|.| {np.abs(off).mean():.3f}"
          f"   min {off.min():+.3f}  max {off.max():+.3f}")
    print(f"    eigenvalue n_eff  : {neff_eig:.2f}   (pairwise-complete {neff_eig_pw:.2f})"
          f"   PC1 {ev[0]/ev.sum():.1%}")
    print(f"    design-eff n_eff  : {neff_deff:.2f}   <- the one d_min uses")
    print(f"    per-series SR     : median {srs.median():+.3f}  sd(between series) {srs.std():.3f}")
    return dict(label=label, S=S, rho=rho, rho_abs=float(np.abs(off).mean()),
                neff_eig=neff_eig, neff_eig_pw=neff_eig_pw, neff_deff=neff_deff,
                pc1=float(ev[0] / ev.sum()), sr_sd=float(srs.std()),
                sr_med=float(srs.median()), years_common=len(common) / 252)


BAR = "=" * 100
res = []

# ══ GRID 1 — QNT-12, 29 series, c2c only ══════════════════════════════════
print(BAR); print("GRID 1 — QNT-12 standalone macro sweep: 29 series x 18 cells, c2c@shift2"); print(BAR)
g1meta = pd.read_csv(f"{Z}/qnt19_postfloor/full_sweep.csv")
G1_IDS = list(g1meta.series.drop_duplicates())
# QNT-12 used yoy on the monthly/quarterly members; ctx.macro_yoy for those.
KIND = g1meta.drop_duplicates("series").set_index("series")["kind"].to_dict()
cat = cta.macro_catalog()

def raw_g1(sid):
    if KIND[sid] == "yoy":
        per = {"M": 12, "Q": 4, "D": 252, "W": 52}.get(cat.loc[sid, "freq"], 12)
        return ctx.macro_yoy(sid, per).astype(float)
    return ctx.macro(sid).astype(float)

g1 = {}
for sid in G1_IDS:
    x = raw_g1(sid)
    cells = {}
    for tn, tf in TF.items():
        for w in WS:
            s = cta.normalize_signal(tf(x, w).replace([np.inf, -np.inf], np.nan),
                                     method="tanh", window=252)
            if s.dropna().empty or s.loc[:IS_END].dropna().shape[0] < 500:
                continue
            gsign = frozen_sign(s, "c2c")
            if gsign is None:
                continue
            cells[f"{tn}|w{w}"] = net_pnl(s, gsign, "c2c")
    if cells:
        g1[sid] = pd.DataFrame(cells)
print(f"  built {len(g1)} series, {sum(d.shape[1] for d in g1.values())} cells")

P1_ew = collapse(g1, "ew"); P1_md = collapse(g1, "median")
res.append(redundancy(P1_ew, "G1 QNT-12 (S=29), EW-of-18"))
res.append(redundancy(P1_md, "G1 QNT-12 (S=29), median-cell"))
P1_ew.to_csv(f"{OUT}/g1_series_pnl_ew.csv")

# ══ GRID 2 — QNT-14, 11 daily series, c2c / day / o2o ═════════════════════
print("\n" + BAR); print("GRID 2 — QNT-14 daily-macro window sweep: 11 series x 18 cells x 3 variants"); print(BAR)
DAILY = ["us_dxy_broad", "us_real_10y", "us_breakeven_10y", "us_breakeven_5y5y",
         "us_dgs5", "us_dgs30", "us_term_premium_10y", "twd_usd", "krw_usd",
         "cny_usd", "wti"]
g2 = {v: build_grid(DAILY, v, "G2") for v in ("c2c", "day", "o2o")}
P2 = {v: collapse(g2[v], "ew") for v in g2}
res.append(redundancy(P2["c2c"], "G2 QNT-14 (S=11), EW-of-18, c2c"))
res.append(redundancy(collapse(g2["c2c"], "median"), "G2 QNT-14 (S=11), median-cell, c2c"))

# the statistic QNT-14 actually quoted was a PAIRED dSR(day - c2c)
D2 = (P2["day"] - P2["c2c"]).dropna(how="all")
res.append(redundancy(D2, "G2 paired d(day-c2c) PnL", note="the +0.089 statistic"))
D2.to_csv(f"{OUT}/g2_series_dpnl.csv")
P2["c2c"].to_csv(f"{OUT}/g2_series_pnl_c2c_ew.csv")

R = pd.DataFrame(res)
R.round(3).to_csv(f"{OUT}/qnt94_neff.csv", index=False)

# ══ POWER TABLE, restated ═════════════════════════════════════════════════
print("\n" + BAR); print("POWER TABLE — QNT-78 Rule 2 restated in n_eff"); print(BAR)
Zc = 2.80                                        # z(.975) + z(.80)
SD_HOUSE = 0.13                                  # QNT-32/78 between-series sd
rows = []
for r in res:
    for sd_name, sd in (("house 0.13", SD_HOUSE), ("measured", r["sr_sd"])):
        rows.append(dict(grid=r["label"], sd_source=sd_name, sd=sd, S=r["S"],
                         neff_eig=r["neff_eig"], neff_deff=r["neff_deff"],
                         d_min_rawS=Zc * sd / np.sqrt(r["S"]),
                         d_min_eig=Zc * sd / np.sqrt(r["neff_eig"]),
                         d_min_deff=Zc * sd / np.sqrt(r["neff_deff"])))
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
T.round(4).to_csv(f"{OUT}/qnt94_power_table.csv", index=False)

print("\n  S -> n_eff -> smallest resolvable d (sd = 0.13), using the measured "
      "eigenvalue n_eff of the per-series statistic:")
print(f"  {'grid':38s} {'S':>3s} {'n_eff':>6s} {'d_min(S)':>9s} {'d_min(n_eff)':>13s}")
for r in res:
    print(f"  {r['label']:38s} {r['S']:3d} {r['neff_eig']:6.2f}"
          f" {Zc*SD_HOUSE/np.sqrt(r['S']):9.3f} {Zc*SD_HOUSE/np.sqrt(r['neff_eig']):13.3f}")

json.dump({r["label"]: r for r in res}, open(f"{OUT}/qnt94_neff.json", "w"), indent=1, default=float)
print(f"\nwrote {OUT}/qnt94_neff.csv, qnt94_power_table.csv, per-series PnL CSVs")
