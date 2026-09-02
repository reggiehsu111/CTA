"""QNT-104 step 6 — the circular-shift NULL CONTROL for the 18 four-gate passers.

QNT-99 Part A3 showed the house gates have a no-information pass rate that is
NOT zero, so "9 of 3,744 cells passed" means nothing until the same grid is run
on features that cannot possibly predict anything. This applies QNT-98's method:
circularly shift every feature by a random offset (which destroys the alignment
with returns but preserves each feature's autocorrelation, distribution and the
sawtooth expiry cycle exactly), rebuild the identical grid, and count.

Stats are recomputed here in vectorised numpy rather than through `wstats` —
40 reps x 3,744 cells is 150k scorings. The definitions are copied from
`macro_window_sweep.wstats` and the real grid is re-scored with them first, so
any drift shows up as a mismatch in the printed reconciliation.
"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read().split("# ── Build the signals once")[0],
             SWEEP, "exec"))  # noqa: S102
import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops

OUT, NREP = "/home/ubuntu/mtx/signal_zoo/qnt104", 40
IS_END = "2016-12-31"
TI = A.index
CALL_F = ["call_oi_total","call_cog","call_disp","call_otm_share","call_far_share","call_wall",
          "call_front_share","call_churn","call_oi_growth","call_cog_chg","call_far_chg"]
COMB_F = ["pcr_oi","pcr_vol","pcr_oi_chg","pcr_far","pcr_atm","cog_gap","cog_mid","far_asym",
          "wall_gap","wall_mid","churn_ratio","oi_growth_diff","disp_ratio","front_diff","max_pain"]
TRANSFORMS = {"selfz": ops.selfz, "robustz": ops.robust_z, "bdtanh": ops.bd_selftanh,
              "rankc": ops.rank_c, "signth": ops.sign_thresh, "dev": ops.dev}
WINDOWS, VARIANTS = (20, 60, 120, 252), ("c2c", "o2o", "day")

names, cols = [], []
for ef in ("monthly", "all"):
    P = (pd.read_csv(f"{OUT}/call_features_{ef}.csv", index_col=0, parse_dates=True)
           .join(pd.read_csv(f"{OUT}/comb_features_{ef}.csv", index_col=0, parse_dates=True))
           .reindex(TI))
    for f in CALL_F + COMB_F:
        for tn, tf in TRANSFORMS.items():
            for w in WINDOWS:
                s = pd.Series(tf(P[f].astype(float), w), index=TI).replace([np.inf, -np.inf], np.nan)
                if s.notna().sum() < 400: continue
                names.append(f"{ef}|{f}|{tn}|w{w}"); cols.append(s.values.astype(float))
S = np.column_stack(cols)                       # T x C
YR   = TI.year.values
ISM  = TI <= pd.Timestamp(IS_END)
print(f"signal matrix {S.shape}")


def score(Smat, variant):
    """Vectorised copy of wstats: sign frozen on IS, then full-sample metrics."""
    lag  = _SHIFT[variant]
    ret  = _RET[variant].reindex(TI).values.astype(float)
    cost = _COST[variant].reindex(TI).values.astype(float)
    pos  = np.vstack([np.full((lag, Smat.shape[1]), np.nan), Smat[:-lag]])
    g    = pos * ret[:, None]
    to   = np.abs(np.diff(np.nan_to_num(pos), axis=0, prepend=0.0))
    tc   = to * cost[:, None]
    ok   = np.isfinite(g)
    def sr(mask):
        m = ok & mask[:, None]
        n = m.sum(0)
        x = np.where(m, np.nan_to_num(g) - np.nan_to_num(tc), np.nan)
        mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
        return np.where((n > 60) & (sd > 0), np.sqrt(252) * mu / np.where(sd == 0, np.nan, sd), np.nan), n
    sr_is, _ = sr(ISM)
    sgn = np.where(np.nan_to_num(sr_is) < 0, -1.0, 1.0)
    g, tc, pos = g * sgn, tc, pos * sgn
    net = np.where(ok, g - np.nan_to_num(tc), np.nan)
    full = np.ones(len(TI), bool)
    mu, sd = np.nanmean(np.where(ok, net, np.nan), 0), np.nanstd(np.where(ok, net, np.nan), 0)
    sr_net = np.sqrt(252) * mu / np.where(sd == 0, np.nan, sd)
    yrs = np.unique(YR)
    ysr = np.full((len(yrs), S.shape[1]), np.nan)
    for i, y in enumerate(yrs):
        m = (YR == y)[:, None] & ok
        n = m.sum(0)
        x = np.where(m, net, np.nan)
        s_ = np.nanstd(x, 0)
        ysr[i] = np.where((n > 20) & (s_ > 0), np.sqrt(252) * np.nanmean(x, 0) / np.where(s_ == 0, np.nan, s_), np.nan)
    ny   = np.sum(np.isfinite(ysr), 0)
    ysd  = np.nanstd(ysr, 0, ddof=1)
    srsr = np.where((ny >= 2) & (ysd > 0), np.nanmean(ysr, 0) / np.where(ysd == 0, np.nan, ysd), np.nan)
    pyr  = np.nansum(ysr > 0, 0) / np.maximum(ny, 1)
    absw = np.nanmean(np.abs(np.where(ok, pos, np.nan)), 0)
    r    = np.where(ok, ret[:, None], np.nan)
    rm, nm = np.nanmean(r, 0), np.nanmean(np.where(ok, net, np.nan), 0)
    cov  = np.nanmean((r - rm) * (np.where(ok, net, np.nan) - nm), 0)
    beta = cov / np.nanvar(r, 0)
    return pd.DataFrame(dict(cell=[f"{n}|{variant}" for n in names], SR_IS=sr_is, SR_net=sr_net,
                             SR_of_SR=srsr, positive_years=pyr, n_years=ny, beta=beta,
                             mean_abs_w=absw))


def grid(Smat):
    d = pd.concat([score(Smat, v) for v in VARIANTS], ignore_index=True)
    return cta.house_gates(d, beta_mode="both")


real = grid(S)
print(f"reconciliation — vectorised real grid: {len(real)} cells, "
      f"{int(real.passes.sum())} four-gate passers (wstats grid gave 9 of 3744)")

rng = np.random.default_rng(104)
cnt = []
for r in range(NREP):
    k = int(rng.integers(250, len(TI) - 250))
    d = grid(np.roll(S, k, axis=0))
    cnt.append(dict(rep=r, shift=k, passers=int(d.passes.sum()),
                    passers_rawbeta=int((d.gate_srsr & d.gate_posyr & d.gate_nyr
                                         & d.gate_beta_raw).sum()),
                    best_SR=float(d.SR_net.max()),
                    med_SR=float(d.SR_net.median())))
    print(f"  rep {r:2d} shift {k:5d}: passers {cnt[-1]['passers']:3d} "
          f"(raw-beta {cnt[-1]['passers_rawbeta']:3d})  best SR {cnt[-1]['best_SR']:+.3f}")
nl = pd.DataFrame(cnt); nl.to_csv(f"{OUT}/null_control.csv", index=False)
obs, obs_raw = int(real.passes.sum()), int((real.gate_srsr & real.gate_posyr & real.gate_nyr
                                            & real.gate_beta_raw).sum())
print(f"\n=== circular-shift null, {NREP} reps, {len(real)} cells each ===")
print(f"beta_per_w gate : observed {obs} passers; null mean {nl.passers.mean():.1f} "
      f"(sd {nl.passers.std():.1f}, range {nl.passers.min()}-{nl.passers.max()}), "
      f"empirical p = {(nl.passers >= obs).mean():.3f}")
print(f"raw-beta gate   : observed {obs_raw}; null mean {nl.passers_rawbeta.mean():.1f} "
      f"(sd {nl.passers_rawbeta.std():.1f}), empirical p = {(nl.passers_rawbeta >= obs_raw).mean():.3f}")
print(f"best cell SR_net: observed {real.SR_net.max():+.3f}; null best mean "
      f"{nl.best_SR.mean():+.3f} (sd {nl.best_SR.std():.3f}, max {nl.best_SR.max():+.3f}), "
      f"empirical p = {(nl.best_SR >= real.SR_net.max()).mean():.3f}")
# and for the standout family
for f in ("pcr_oi", "front_diff", "call_oi_total"):
    sub = real[real.cell.str.contains(f"\\|{f}\\|")]
    print(f"  {f:15s}: observed passers {int(sub.passes.sum()):2d} of {len(sub)} cells, "
          f"median SR_net {sub.SR_net.median():+.3f}")
real.to_csv(f"{OUT}/null_real_vectorised.csv", index=False)
