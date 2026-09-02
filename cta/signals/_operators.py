"""
Small helpers used by many signals. Copied from the notebook so signals
don't need to reach into notebook globals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cta


def selftanh(x: pd.Series) -> pd.Series:
    """tanh(x) with inf → NaN sanitation."""
    return np.tanh(x.replace([np.inf, -np.inf], np.nan))


def dev(x: pd.Series, w: int) -> pd.Series:
    """x − rolling mean (window w)."""
    return x - cta.InstMean(w, x)


def selfz(x: pd.Series, w: int) -> pd.Series:
    """Rolling z-score."""
    mu = cta.InstMean(w, x)
    sd = cta.InstStdev(w, x).replace(0, np.nan)
    return (x - mu) / sd


def selfz_winsor(x: pd.Series, w: int, c: float = 3.0) -> pd.Series:
    """selfz clipped to [-c, +c] then divided by c → in [-1, +1]."""
    return selfz(x, w).clip(-c, c) / c


def robust_z(x: pd.Series, w: int) -> pd.Series:
    """MAD-based z. Robust to outliers."""
    med = x.rolling(w, min_periods=max(3, w // 2)).median()
    mad = (x - med).abs().rolling(w, min_periods=max(3, w // 2)).median()
    return (x - med) / (1.4826 * mad).replace(0, np.nan)


def sign_thresh(x: pd.Series, w: int, t: float = 0.5) -> pd.Series:
    """±1 when |selfz| > t, else 0."""
    z = selfz(x, w)
    return pd.Series(np.where(z > t, 1.0, np.where(z < -t, -1.0, 0.0)), index=x.index)


def rank_c(x: pd.Series, w: int) -> pd.Series:
    """Centered rank in [-1, +1]."""
    return (cta.InstRank(w, x) - 0.5) * 2


def bd_selftanh(x: pd.Series, w: int) -> pd.Series:
    """Shorthand: tanh(selfz(x, w))."""
    return selftanh(selfz(x, w))


def pair_chg_ratio(A: pd.Series, B: pd.Series, N: int) -> pd.Series:
    """(A_chg_N / B_chg_N), floored on B's rolling-mean-abs to avoid division blowups."""
    A_chg = A - A.shift(N)
    B_chg = B - B.shift(N)
    floor = 0.0005 * B.abs().rolling(120, min_periods=20).mean().shift(1)
    denom = B_chg.where(B_chg.abs() > floor, np.nan)
    return (A_chg / denom).clip(-20, 20)
