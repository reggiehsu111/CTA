"""QNT-99 Part A3 — the release CALENDAR as timing, with no value attached.

Scheduled US releases are known months ahead, so a calendar signal is PIT by
construction (no revision problem, no publication lag). Two questions:

 Q1 EVENT STUDY: does MTX earn an abnormal return in a specific execution window
    on / around the TW day an event becomes public? FOMC is included here even
    though it carries no value.  FOMC 14:00 ET = 02:00-03:00 TPE, i.e. INSIDE the
    TAIFEX night session of the previous TW day, so `ongap` (05:00->08:45) of the
    actable day is the first cash window that can hold the reaction.
 Q2 CALENDAR SIGNAL: an always-on position built only from "days until the next
    scheduled event", swept over event type and offset, scored on the gates.

Discipline: same _RET/_COST/_SHIFT harness as every other MTX sweep, real costs,
sign frozen on the IS half. Offsets are measured in TW trading days from the
actable date; a NEGATIVE offset is the PRE-event window and is legitimate because
the schedule is published in advance.
"""
import sys, os, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
sys.path.insert(0, "/home/ubuntu/mtx/signal_zoo/qnt99")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))  # noqa: S102

import numpy as np, pandas as pd
from scipy import stats
import event_inputs as ei

OUT = "/home/ubuntu/mtx/signal_zoo/qnt99"
IS_END, OOS_START = "2018-12-31", "2019-01-01"
TI = A.index
EVENTS = sorted(set(ei.release_calendar()["release_name"]) - {"GDP_second", "GDP_third"})
OFFSETS = (-2, -1, 0, 1, 2)
WINDOWS = ("c2c", "o2o", "day", "ongap")
# night session return, for completeness: 15:00 t -> 05:00 t+1 == night_close[t+1]/night_open[t+1]
_RET["night"] = (A["night_close"] / A["night_open"] - 1).astype(float)
_COST["night"] = FIXED / (A["night_open"].astype(float) * PV) + FEE
_SHIFT["night"] = 1

def ev_mask(event, offset):
    d = ei.event_tw_dates(event, TI)
    if not len(d):
        return None
    pos = pd.Index(TI).get_indexer(pd.DatetimeIndex(d.values))
    pos = pos[pos >= 0] + offset
    pos = pos[(pos >= 0) & (pos < len(TI))]
    m = pd.Series(0.0, index=TI); m.iloc[np.unique(pos)] = 1.0
    return m

# ── Q1: event study ────────────────────────────────────────────────────────
rows = []
for ev in EVENTS:
    for off in OFFSETS:
        m = ev_mask(ev, off)
        if m is None: continue
        for w in ("day", "ongap", "night", "c2c"):
            r = _RET[w].reindex(TI).astype(float)
            j = pd.concat([r.rename("r"), m.rename("m")], axis=1).dropna()
            j = j[j.index >= "2010-01-01"]
            on, off_ = j.loc[j.m == 1, "r"], j.loc[j.m == 0, "r"]
            if len(on) < 30: continue
            t, p = stats.ttest_ind(on, off_, equal_var=False)
            # per-year sign consistency of the event-day mean
            yr = on.groupby(on.index.year).mean()
            rows.append(dict(event=ev, offset=off, window=w, n_ev=len(on),
                             mean_bps=on.mean()*1e4, base_bps=off_.mean()*1e4,
                             diff_bps=(on.mean()-off_.mean())*1e4, t=t, p=p,
                             pos_years=float((yr > 0).mean()), n_years=len(yr),
                             ann_SR_if_traded=float(np.sqrt(252)*on.mean()/on.std())))
es = pd.DataFrame(rows).sort_values("p")
es.to_csv(f"{OUT}/event_study.csv", index=False)
print(f"=== Q1 event study: {len(es)} (event x offset x window) cells, 2010- ===")
print(es.head(18)[["event","offset","window","n_ev","mean_bps","base_bps","diff_bps","t","p","pos_years","n_years"]].to_string(index=False))
nb = len(es)
print(f"\nBonferroni threshold at {nb} tests: p < {0.05/nb:.5f}   "
      f"cells below it: {(es.p < 0.05/nb).sum()}   cells with raw p<0.05: {(es.p<0.05).sum()} "
      f"(expected by chance {0.05*nb:.1f})")

# ── Q2: calendar signals scored on the gates ───────────────────────────────
rows = []
for ev in EVENTS:
    for off in OFFSETS:
        m = ev_mask(ev, off)
        if m is None: continue
        for hold in (1, 2):
            sig = m.rolling(hold, min_periods=1).max()          # on for `hold` days
            for w in WINDOWS + ("night",):
                is_ = wstats(sig, w, end=IS_END)
                if is_ is None: continue
                oos = wstats(sig, w, start=OOS_START, sign=is_["sign"])
                full = wstats(sig, w, sign=is_["sign"])
                if full is None: continue
                rows.append(dict(event=ev, offset=off, hold=hold, window=w,
                                 cell=f"{ev}|off{off}|h{hold}|{w}", sign=is_["sign"],
                                 SR_IS=is_["SR_net"], SR_OOS=(oos or {}).get("SR_net", np.nan),
                                 **{f"full_{a}": full[a] for a in
                                    ("SR_net","SR_of_SR","positive_years","yr_sr_min",
                                     "n_years","beta","abs_exec_w","turnover_ann",
                                     "held_pct","n_bars")}))
cs = pd.DataFrame(rows)
cs.to_csv(f"{OUT}/event_calendar_sweep.csv", index=False)
g = cs[(cs.full_SR_of_SR>0.6)&(cs.full_positive_years>=0.65)&(cs.full_beta.abs()<0.15)&(cs.full_n_years>=5)]
print(f"\n=== Q2 calendar sweep: {len(cs)} cells, {len(g)} pass all 4 gates ===")
if len(g):
    print(g.sort_values("full_SR_net",ascending=False).head(15)[
        ["cell","sign","SR_IS","SR_OOS","full_SR_net","full_SR_of_SR","full_positive_years",
         "full_beta","full_n_years","full_held_pct"]].to_string(index=False))
import cta
h = cs.rename(columns={"full_SR_net":"SR_net","full_n_years":"n_years"})
cta.sweep_headline(h, value="SR_net", series_col="event", label="QNT-99 event calendar").print()
cta.sweep_headline(h.dropna(subset=["SR_OOS"]), value="SR_OOS", series_col="event",
                   label="QNT-99 event calendar OOS").print()
d = cs.dropna(subset=["SR_IS","SR_OOS"])
print(f"IS->OOS: med IS {d.SR_IS.median():+.3f}  med OOS {d.SR_OOS.median():+.3f}  "
      f"frac OOS>0 {(d.SR_OOS>0).mean():.3f}  corr {d.SR_IS.corr(d.SR_OOS):+.3f}")
