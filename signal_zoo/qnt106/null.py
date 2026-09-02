"""QNT-106 step 5 — circular-shift null: how many four-gate passers does a
grid of the SAME size and autocorrelation produce when the relation to MTX is
destroyed?  (QNT-104 method: shift the SOURCE SERIES, re-choose the sign on IS,
re-score.)"""
import sys, io, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
exec(compile(io.open(SWEEP, encoding="utf-8").read()
             .split("# ── Build the signals once")[0], SWEEP, "exec"))
import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops
sys.path.insert(0, "/home/ubuntu/mtx/signal_zoo/qnt106")
from features import load as load_features
OUT = "/home/ubuntu/mtx/signal_zoo/qnt106"
rng = np.random.default_rng(20261106)

TI = A.index; T = len(TI)
F = load_features().reindex(TI)
TRANSFORMS = {"selfz": ops.selfz, "robustz": ops.robust_z, "bdtanh": ops.bd_selftanh,
              "rankc": ops.rank_c, "signth": ops.sign_thresh, "dev": ops.dev}
WINDOWS = (20, 60, 120, 252)
NIGHT_START = A["night_close"].first_valid_index()
YR = TI.year.values; YEARS = np.unique(YR)

def build(shift=0):
    """(n_sig, T) matrix of tanh-normalised signals; `shift` circularly shifts
    the SOURCE series before the transform, so autocorrelation survives."""
    out = []
    for f in F.columns:
        x = F[f].astype(float)
        if shift:
            v = x.values.copy(); x = pd.Series(np.roll(v, shift), index=TI)
        for tf in TRANSFORMS.values():
            for w in WINDOWS:
                s = pd.Series(tf(x, w), index=TI).replace([np.inf, -np.inf], np.nan)
                out.append(cta.normalize_signal(s, method="tanh", window=252).values)
    return np.asarray(out, dtype=float)

def gate_count(S, regime):
    r_start, is_end, variants = regime
    npass = 0
    for v in variants:
        ret = _RET[v].reindex(TI).values.astype(float)
        cost = _COST[v].reindex(TI).values.astype(float)
        lag = _SHIFT[v]
        pos = np.full_like(S, np.nan); pos[:, lag:] = S[:, :-lag]
        g = pos * ret
        to = np.abs(np.diff(np.nan_to_num(pos), axis=1, prepend=0.0))
        tc = to * cost
        win = np.ones(T, bool)
        if r_start is not None: win &= np.asarray(TI >= r_start)
        is_m = win & np.asarray(TI <= pd.Timestamp(is_end))
        with np.errstate(invalid="ignore"):
            sgn = np.sign(np.nanmean(np.where(is_m, g, np.nan), axis=1))
        sgn[sgn == 0] = 1.0
        n = (g * sgn[:, None]) - tc
        n = np.where(win, n, np.nan)
        yr_sr = []
        for y in YEARS:
            m = (YR == y)
            if m.sum() < 21: yr_sr.append(np.full(S.shape[0], np.nan)); continue
            blk = n[:, m]
            cnt = np.sum(~np.isnan(blk), axis=1)
            sd = np.nanstd(blk, axis=1, ddof=1)
            sr = np.sqrt(252) * np.nanmean(blk, axis=1) / np.where(sd > 0, sd, np.nan)
            yr_sr.append(np.where(cnt > 20, sr, np.nan))
        Y = np.asarray(yr_sr)                       # (years, n_sig)
        ny = np.sum(~np.isnan(Y), axis=0)
        m_, s_ = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0, ddof=1)
        sr_of_sr = m_ / np.where(s_ > 0, s_, np.nan)
        pos_yr = np.nansum(Y > 0, axis=0) / np.maximum(ny, 1)
        # beta of the net series vs buy-and-hold of the same window
        x = np.where(win, ret, np.nan)
        xm = np.nanmean(x); xv = np.nanvar(x)
        beta = np.nanmean((n - np.nanmean(n, axis=1, keepdims=True)) * (x - xm), axis=1) / xv
        npass += int(np.sum((sr_of_sr > 0.6) & (pos_yr >= 0.65)
                            & (np.abs(beta) < 0.15) & (ny >= 5)))
    return npass

REG = {"full":  (None, "2017-12-31", ("c2c", "o2o", "day")),
       "night": (NIGHT_START, "2022-12-31", ("c2c", "o2o", "day", "ongap"))}
S0 = build(0)
obs = {k: gate_count(S0, v) for k, v in REG.items()}
print(f"observed (vectorised re-score) : {obs}", flush=True)

R = 40
draws = {k: [] for k in REG}
for i in range(R):
    k = int(rng.integers(300, T - 300))
    Sk = build(k)
    for name, v in REG.items():
        draws[name].append(gate_count(Sk, v))
    if (i + 1) % 10 == 0: print(f"  rep {i+1}/{R}", flush=True)

print("\n=== circular-shift null, four-gate passers ===")
out = []
for name in REG:
    d = np.array(draws[name], float)
    p = (np.sum(d >= obs[name]) + 1) / (R + 1)
    print(f"{name:6s} observed {obs[name]:4d}   null mean {d.mean():6.1f} "
          f"sd {d.std(ddof=1):5.1f}   p = {p:.3f}")
    out.append(dict(regime=name, observed=obs[name], null_mean=d.mean(),
                    null_sd=d.std(ddof=1), p=p, reps=R))
pd.DataFrame(out).to_csv(f"{OUT}/null_control.csv", index=False)
pd.DataFrame(draws).to_csv(f"{OUT}/null_draws.csv", index=False)
