"""QNT-18 part 1: does the `day` window help the SLOW (non-daily) macro inputs too?

QNT-14 found `day` (08:45→13:45) beat `c2c` across the whole 198-cell grid of the
11 DAILY US-close macro series, and decomposed the gain ~2/3 window, ~1/3 the
extra session of information (shift 1 vs 2). This tests the WINDOW half alone on
the slow inputs, where the information half is not available.

Scope: every QNT-12 macro input that is NOT a daily series — 20 monthlies plus
`us_stlfsi` (weekly). NB the ticket says "the other 18": that is 29 QNT-12 inputs
minus QNT-14's 11-name DAILY list, but 3 of those 11 (us_breakeven_10y, us_dgs30,
cny_usd) were never in QNT-12's FAMILIES. The true complement is 21 series.

Lag: shift(2) for BOTH variants. A monthly with pub_lag_days 16-90 does not
publish overnight TPE, so the shift(1) fill QNT-14 used for the dailies is not
available to it. Holding the lag fixed at 2 isolates the pure window effect.
Note `day` shift(2) is strictly MORE conservative than `c2c` shift(2): the c2c
entry is 13:45 of t-1, the day entry is 08:45 of t, 19 hours later.

Reuses `macro_window_sweep.py`'s setup block verbatim (exec of everything above
the "Build the signals once" marker) so `_RET`, `_COST` and `wstats` are
bit-identical to the QNT-14 evidence, and nothing here overwrites its CSVs.

Sign frozen on the IS half and carried into OOS. Nothing here selects a sign or
a recommended variant.
"""
import sys, warnings, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")

SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
_marker = "# ── Build the signals once"
assert _marker in _src, "macro_window_sweep.py changed shape - re-check the split point"
exec(compile(_src.split(_marker)[0], SWEEP, "exec"))          # noqa: S102

import numpy as np, pandas as pd
import cta

OUT = "/home/ubuntu/mtx/signal_zoo/macro_windows"
IS_END, OOS_START = "2018-12-31", "2019-01-01"

# ── QNT-12 FAMILIES, verbatim from signal_zoo/macro_sweep/macro_sweep.py ──
FAMILIES = {
    "semis": [("us_semi_ip", "yoy", 12), ("us_semi_ip_nsa", "yoy", 12),
              ("us_semi_ppi", "yoy", 12), ("us_electronics_ppi", "yoy", 12)],
    "kr_tech_demand": [("kr_exports_sa", "yoy", 12), ("kr_exports", "yoy", 12),
                       ("kr_kospi", "yoy", 12)],
    "cn_cycle": [("cn_leading_idx", "level", 0), ("cn_exports", "yoy", 12),
                 ("cn_shanghai_comp", "yoy", 12)],
    "us_cycle": [("us_cfnai", "level", 0), ("us_mfg_new_orders", "yoy", 12),
                 ("us_freight_tsi", "yoy", 12), ("us_retail_inv_sales", "level", 0),
                 ("us_recession_prob", "level", 0)],
    "us_survey": [("us_empire_state", "level", 0), ("us_philly_fed", "level", 0)],
    "rates": [("us_term_premium_10y", "level", 0), ("us_real_10y", "level", 0),
              ("us_breakeven_5y5y", "level", 0), ("us_dgs5", "level", 0)],
    "risk": [("us_stlfsi", "level", 0), ("epu_global", "level", 0)],
    "fx": [("twd_usd", "level", 0), ("krw_usd", "level", 0), ("us_dxy_broad", "level", 0)],
    "commodity": [("copper", "yoy", 12), ("wti", "level", 0), ("igrea", "level", 0)],
}

cat = cta.macro_catalog()
raw_inputs, meta = {}, {}
for fam, entries in FAMILIES.items():
    for sid, kind, per in entries:
        freq = cat.loc[sid, "freq"]
        if freq == "D":                      # QNT-14 already covered these
            continue
        if kind == "yoy":
            per = {"M": 12, "Q": 4, "D": 252, "W": 52}.get(freq, 12)
            x = ctx.macro_yoy(sid, per)
        else:
            x = ctx.macro(sid)
        key = f"{sid}_{kind}"
        raw_inputs[key] = x.astype(float)
        meta[key] = dict(family=fam, series=sid, kind=kind, periods=per, freq=freq,
                         pub_lag_days=int(cat.loc[sid, "pub_lag_days"]))
print(f"slow inputs: {len(raw_inputs)}  "
      f"(freqs: {pd.Series([m['freq'] for m in meta.values()]).value_counts().to_dict()})")

SIGS = {}
for key, x in raw_inputs.items():
    for tn, tf in TRANSFORMS.items():
        for w in WINDOWS:
            s = tf(x, w).replace([np.inf, -np.inf], np.nan)
            s = cta.normalize_signal(s, method="tanh", window=252)
            # same admission filter QNT-12 used, so the cell set is comparable
            if s.dropna().empty or s.loc[:IS_END].dropna().shape[0] < 500:
                continue
            SIGS[f"{key}|{tn}|w{w}"] = (s, key)
print(f"cells per variant: {len(SIGS)}")

# BOTH variants at shift(2): the information channel is closed, only the
# window differs.  ongap/o2o included for completeness at the same lag.
VARIANTS = ("c2c", "day", "o2o", "ongap")
LAG = 2

rows = []
for cand, (s, key) in SIGS.items():
    _sid, tn, w = cand.rsplit("|", 2)
    for v in VARIANTS:
        keep, _SHIFT[v] = _SHIFT[v], LAG
        try:
            is_st = wstats(s, v, end=IS_END)
            if is_st is None:
                continue
            sign = is_st["sign"]
            oos  = wstats(s, v, start=OOS_START, sign=sign)
            full = wstats(s, v, sign=sign)
            if full is None:
                continue
            rows.append(dict(cand=cand, **meta[key], transform=tn, window=int(w[1:]),
                             variant=v, shift=LAG, sign_IS=sign,
                             SR_IS=is_st["SR_net"], SR_OOS=(oos or {}).get("SR_net", np.nan),
                             **{k: full[k] for k in
                                ("SR_net", "SR_gross", "SR_of_SR", "positive_years",
                                 "yr_sr_min", "n_years", "beta", "mean_exec_w",
                                 "abs_exec_w", "max_dd_pct", "max_dd_days",
                                 "turnover_ann", "held_pct", "n_bars",
                                 "start_date", "end_date")}))
        finally:
            _SHIFT[v] = keep

df = pd.DataFrame(rows)
for g, c in [("gate_srsr", df.SR_of_SR > 0.6), ("gate_posyr", df.positive_years >= 0.65),
             ("gate_beta", df.beta.abs() < 0.15), ("gate_nyr", df.n_years >= 5)]:
    df[g] = c
df["n_gates"] = df[["gate_srsr", "gate_posyr", "gate_beta", "gate_nyr"]].sum(axis=1)
df.to_csv(f"{OUT}/slow_window_sweep.csv", index=False)
print(f"rows: {len(df)} -> slow_window_sweep.csv")

# ── Paired comparison, cell by cell ───────────────────────────────────────
pd.set_option("display.width", 220, "display.max_columns", 40)
piv = df.pivot_table(index="cand", columns="variant",
                     values=["SR_net", "SR_of_SR", "beta", "mean_exec_w", "positive_years"])
print("\n=== PAIRED day(2) - c2c(2), slow macro inputs, full sample, net of real costs ===")
for m in ["SR_net", "SR_of_SR", "positive_years"]:
    d = (piv[m]["day"] - piv[m]["c2c"]).dropna()
    print(f"  Δ{m:15s} median {d.median():+.4f}  mean {d.mean():+.4f}  "
          f"win-rate {(d > 0).mean():.1%}  n={len(d)}")
for m in ["beta", "mean_exec_w"]:
    print(f"  |{m}| median  c2c {piv[m]['c2c'].abs().median():.4f}   "
          f"day {piv[m]['day'].abs().median():.4f}")

print("\n=== four-gate passers by variant (all at shift 2) ===")
print(df.groupby("variant")["n_gates"].agg(
    cells="size", pass4=lambda s: int((s == 4).sum()), mean_gates="mean").round(3).to_string())

print("\n=== median SR_net by variant ===")
print(df.groupby("variant")[["SR_net", "SR_of_SR", "positive_years", "beta", "mean_exec_w"]]
        .median().round(4).to_string())

print("\n=== Δ(day-c2c) SR_net by family ===")
fam = df[["cand", "family"]].drop_duplicates().set_index("cand")["family"]
d = (piv["SR_net"]["day"] - piv["SR_net"]["c2c"]).dropna().rename("dSR").to_frame()
d["family"] = fam.reindex(d.index)
print(d.groupby("family")["dSR"].agg(["size", "median", "mean",
                                      ("winrate", lambda s: (s > 0).mean())]).round(3).to_string())

print("\n=== Δ(day-c2c) SR_net by transform ===")
d["transform"] = [c.rsplit("|", 2)[1] for c in d.index]
print(d.groupby("transform")["dSR"].agg(["size", "median",
                                         ("winrate", lambda s: (s > 0).mean())]).round(3).to_string())

allg = df[df.n_gates == 4]
print(f"\n=== cells passing all four house gates: {len(allg)} ===")
if len(allg):
    print(allg[["cand", "variant", "SR_net", "SR_of_SR", "positive_years", "n_years",
                "beta", "mean_exec_w", "SR_IS", "SR_OOS"]].round(3).to_string(index=False))


# ── QNT-32 / QNT-25 reporting line ────────────────────────────────────────
# Same correction as QNT-14's grid: 378 cells are 21 source series x 18
# transform-windows, ICC(series) ~ 0.53, so n_eff ~ 38 — and the per-CELL
# Wilcoxon p=0.000 printed above collapses to p~0.24 once each series
# contributes one number. Both variants are at shift 2 here, so this is the
# pure window effect with no information leg.
print(f"\n{'='*100}\n=== QNT-25 REPORTING LINE (quote this, not the cell-level p) ===")
_ny = float(df.n_years.median())
_p = df.pivot_table(index=["series", "transform", "window"],
                    columns="variant", values="SR_net").reset_index()
cta.sweep_headline(df[df.variant == "c2c"], "SR_net", n_years=_ny,
                   label="QNT-18 slow-macro window sweep, SR_net on c2c").print()
for _v in [c for c in _p.columns if c not in ("series", "transform", "window", "c2c")]:
    _q = _p.dropna(subset=[_v, "c2c"])
    cta.paired_headline(_q[_v], _q["c2c"], series=_q["series"], n_years=_ny,
                        label="QNT-18 slow-macro window sweep",
                        a_name=f"{_v}@2", b_name="c2c@2").print()
