"""QNT-99 Part A2 — the RELEASE-SURPRISE grid.

PRE-RUN POWER TARGET (QNT-78 Rule 2, stated before the run):
    target d = 0.15 SR units (an uplift big enough to change a decision here)
    assumed between-series sd = 0.13
    S_required (raw)                = (2.80*0.13/0.15)^2 = 5.9  -> 6
    S_required (QNT-94 corrected)   = 2.33 * 5.9          = 13.8 -> 14
    S available = 15 release events (16 fields) -> ADEQUATE for d = 0.15 and for
    nothing smaller: d_min = 4.26*0.13/sqrt(15) = 0.143.
    Raw S is optimistic; the events correlate (CPI/PPI/PCE, NFP/claims/JOLTS).

Transform palette kept DELIBERATELY NARROW (Rule 2: add source series, not
transforms): 2 standardisation windows x 3 hold horizons x 2 shapes = 12 cells
per field, and the cell axis is not where the power is.

Sign FROZEN on the in-sample half (<=2018-12-31) and carried into OOS unchanged.
"""
import sys, os, io, warnings, itertools
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
sys.path.insert(0, "/home/ubuntu/mtx/signal_zoo/qnt99")

SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))  # noqa: S102
# -> ctx, A, wstats, _RET, _COST, _SHIFT, PV, FIXED, FEE, PPY

import numpy as np, pandas as pd
import event_inputs as ei

OUT = "/home/ubuntu/mtx/signal_zoo/qnt99"
IS_END, OOS_START = "2018-12-31", "2019-01-01"
TI = A.index
VARIANTS = ("c2c", "o2o", "day", "ongap")
KWIN   = (12, 36)            # trailing releases used to standardise the surprise
HOLDS  = (1, 3, 10)          # trading days the impulse is held
SHAPES = {"z":  lambda z: z.clip(-3, 3) / 3.0,
          "sgn": lambda z: np.sign(z) * (z.abs() > 1.0)}

rows = []
pnl_store = {}
for ev, (tbl, fields, kind) in ei.EVENTS.items():
    evd = ei.event_tw_dates(ev, TI)
    if not len(evd):
        continue
    for fld in fields:
        for k in KWIN:
            z = ei.surprise_series(ev, fld, kind, k)
            for h in HOLDS:
                imp = ei.impulse(z, evd, TI, h)
                for sh, fn in SHAPES.items():
                    sig = pd.Series(fn(imp), index=TI).fillna(0.0)
                    if sig.abs().sum() == 0:
                        continue
                    for v in VARIANTS:
                        is_ = wstats(sig, v, end=IS_END)
                        if is_ is None:
                            continue
                        oos = wstats(sig, v, start=OOS_START, sign=is_["sign"])
                        full = wstats(sig, v, sign=is_["sign"])
                        if full is None:
                            continue
                        cell = f"{ev}|{fld}|k{k}|h{h}|{sh}|{v}"
                        rows.append(dict(event=ev, field=fld, kwin=k, hold=h,
                                         shape=sh, variant=v, cell=cell,
                                         sign=is_["sign"], SR_IS=is_["SR_net"],
                                         SR_OOS=(oos or {}).get("SR_net", np.nan),
                                         **{f"full_{a}": full[a] for a in
                                            ("SR_net", "SR_of_SR", "positive_years",
                                             "yr_sr_min", "n_years", "beta",
                                             "mean_exec_w", "abs_exec_w",
                                             "turnover_ann", "held_pct",
                                             "max_dd_days", "n_bars")}))
                        # keep the net PnL for the n_eff measurement
                        pos = sig.shift(_SHIFT[v]) * is_["sign"]
                        pnl = (pos * _RET[v] - pos.fillna(0).diff().abs() * _COST[v])
                        pnl_store[cell] = pnl

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/event_surprise_full.csv", index=False)
print(f"cells: {len(df)}   events: {df.event.nunique()}   fields: {df.field.nunique()}")

g = df[(df.full_SR_of_SR > 0.6) & (df.full_positive_years >= 0.65)
       & (df.full_beta.abs() < 0.15) & (df.full_n_years >= 5)]
print(f"\n=== 4-GATE PASSES: {len(g)} of {len(df)} ===")
if len(g):
    print(g.sort_values("full_SR_of_SR", ascending=False)
           .head(25)[["cell", "sign", "SR_IS", "SR_OOS", "full_SR_net", "full_SR_of_SR",
                      "full_positive_years", "full_beta", "full_n_years",
                      "full_abs_exec_w", "full_turnover_ann"]].to_string(index=False))
g.to_csv(f"{OUT}/event_surprise_gated.csv", index=False)

# per-EVENT best and median, the axis that carries the power
per = df.groupby("event").agg(cells=("full_SR_net", "size"),
                              med_SR=("full_SR_net", "median"),
                              best_SR=("full_SR_net", "max"),
                              med_IS=("SR_IS", "median"),
                              med_OOS=("SR_OOS", "median"),
                              n_gate=("cell", "size"))
per["n_gate"] = g.groupby("event").size().reindex(per.index).fillna(0).astype(int)
print("\n=== per event ===")
print(per.sort_values("med_SR", ascending=False).to_string())
per.to_csv(f"{OUT}/event_surprise_per_event.csv")

pd.DataFrame(pnl_store).to_pickle(f"{OUT}/event_surprise_pnl.pkl")
print("\nwrote event_surprise_full.csv / _gated.csv / _per_event.csv / _pnl.pkl")
