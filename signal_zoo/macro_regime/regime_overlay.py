"""QNT-15: do the existing MTX signals behave differently across macro regimes?

Measurement only. Nothing here selects a sign, a weight, or a variant.

Discipline (台指期 standing brief):
  * every macro input is PIT-aligned via ctx.macro / ctx.nfci_tw (publication-
    lag shifted), never cta.load_macro
  * the regime LABEL is shifted by the same exec_lag as the position, so the
    label attached to bar t's return was observable when that position was set
  * each signal keeps its COMMITTED sign (auto_flip=False); nothing is re-flipped
  * roll-adjusted returns, realistic costs (fixed_per_side 70, fee_rate 4e-5)
  * n_bars AND n_episodes reported per cell — regimes are persistent, so the
    honest sample size is the number of contiguous regime episodes, not days
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context
from cta.signals._base import SIGNAL_REGISTRY
import cta.signals                      # triggers registration

OUT = "/home/ubuntu/mtx/signal_zoo/macro_regime"
EXEC_LAG = 2
REAL = dict(fixed_per_side=70.0, fee_rate=0.00004)
IS_END, OOS_START = "2018-12-31", "2019-01-01"
ZWIN = 120
RNG = np.random.default_rng(20260901)
NBOOT = 2000

ctx = build_context()
A = ctx.asset
IDX = A.index
PPY = int(A.periods_per_year)
assert IDX.max() >= pd.Timestamp("2026-08-29"), "stale asset"
assert A["volume"].iloc[-1] == A["volume"].iloc[-1], "NaN volume"
assert A["open"].iloc[-1] != A["night_open"].iloc[-1], "day==night corrupted row"

RET = A.returns                                     # roll-adjusted
CLOSE = A["close"].astype(float)
COST_PCT = REAL["fixed_per_side"] / (CLOSE * 50.0) + REAL["fee_rate"]

# ── 1. signals: committed sign, runner-identical normalisation ──────────────
SIGS, SMETA = {}, {}
for name, s in SIGNAL_REGISTRY.items():
    raw = s.compute_raw(ctx)
    sig = raw if s.pre_normalized else cta.normalize_signal(raw, method="tanh", window=252)
    sig = pd.Series(sig).reindex(IDX).astype(float) * s.sign
    SIGS[name] = sig
    SMETA[name] = dict(sign=s.sign, enabled=bool(s.enabled),
                       pre_normalized=bool(s.pre_normalized),
                       recommended=",".join(s.recommended_variants) or "-",
                       first=str(sig.first_valid_index().date()) if sig.first_valid_index() is not None else "-")
print(f"signals built: {len(SIGS)}")

# equal-weight book of the ENABLED signals — reference row only, not a proposal
en = [n for n in SIGS if SMETA[n]["enabled"]]
SIGS["_EW_enabled"] = pd.concat([SIGS[n] for n in en], axis=1).mean(axis=1, skipna=True)
SMETA["_EW_enabled"] = dict(sign=0, enabled=True, pre_normalized=False,
                            recommended="-", first="-")
SIGS["_buy_and_hold"] = pd.Series(1.0, index=IDX)
SMETA["_buy_and_hold"] = dict(sign=0, enabled=True, pre_normalized=True,
                              recommended="-", first="-")

# ── 2. regimes, PIT ─────────────────────────────────────────────────────────
def z(x, w=ZWIN):
    return ops.selfz(x.astype(float), w)

igrea   = ctx.macro("igrea")
stlfsi  = ctx.macro("us_stlfsi")
epu     = ctx.macro("epu_global")
nfci    = ctx.nfci_tw("nfci")

REGIMES = {
    # dim            -> (boolean series (True = first state), name_true, name_false)
    "igrea":   (z(igrea)  >= 0, "igrea_high",   "igrea_low"),
    "stlfsi":  (z(stlfsi) >= 0, "stress_on",    "stress_off"),
    "epu":     (z(epu)    >= 0, "epu_high",     "epu_low"),
    "nfci":    (z(nfci)   >= 0, "nfci_tight",   "nfci_loose"),
    # absolute-level robustness checks (these indices are constructed around 0)
    "stlfsi_lvl": (stlfsi > 0,  "stress_on_lvl",  "stress_off_lvl"),
    "nfci_lvl":   (nfci   > 0,  "nfci_tight_lvl", "nfci_loose_lvl"),
}
# mask out days where the underlying is NaN (pre-history / warm-up)
VALID = {}
for k, (b, *_ ) in REGIMES.items():
    src = {"igrea": igrea, "stlfsi": stlfsi, "epu": epu, "nfci": nfci,
           "stlfsi_lvl": stlfsi, "nfci_lvl": nfci}[k]
    warm = z(src) if not k.endswith("_lvl") else src
    VALID[k] = warm.reindex(IDX).notna()

print("\nregime balance (label shifted by exec_lag, on the full asset index):")
REG_LAB = {}
for k, (b, tname, fname) in REGIMES.items():
    lab = b.reindex(IDX).where(VALID[k]).shift(EXEC_LAG)
    REG_LAB[k] = lab
    v = lab.dropna()
    print(f"  {k:11s} {tname:15s} {v.mean():.1%}   {fname:16s} {1-v.mean():.1%}   n={len(v)}")

def episodes(mask: pd.Series) -> int:
    """Number of contiguous True runs — the honest sample size for a persistent regime."""
    m = mask.fillna(False).astype(bool).values
    return int(np.sum(m & ~np.concatenate([[False], m[:-1]])))

# ── 3. per-cell stats ───────────────────────────────────────────────────────
def pnl_series(sig):
    ex = sig.shift(EXEC_LAG)
    g  = ex * RET
    tc = ex.fillna(0).diff().abs() * COST_PCT
    return ex, g, (g - tc)

def cell_stats(ex, net, gross, mask, start=None, end=None):
    m = mask.fillna(False)
    if start is not None: m = m & (m.index >= pd.Timestamp(start))
    if end   is not None: m = m & (m.index <= pd.Timestamp(end))
    n  = net.where(m).dropna()
    n  = n[n.index.isin(RET.dropna().index)]
    if len(n) < 10 or not np.isfinite(n.std()) or n.std() == 0:
        return dict(n_bars=int(len(n)), n_episodes=episodes(m.reindex(net.index)),
                    SR_net=np.nan, SR_gross=np.nan, ann_ret_net_pct=np.nan,
                    hit_rate=np.nan, beta=np.nan, mean_pos=np.nan, held_pct=np.nan)
    g  = gross.reindex(n.index)
    e  = ex.reindex(n.index).fillna(0)
    bh = RET.reindex(n.index)
    j  = pd.concat([n.rename("y"), bh.rename("x")], axis=1).dropna()
    beta = float(np.cov(j.y.values, j.x.values, ddof=0)[0, 1] / j.x.var(ddof=0)) \
           if len(j) >= 30 and j.x.var() > 0 else np.nan
    act = e.abs() > 0.01
    return dict(
        n_bars=int(len(n)),
        n_episodes=episodes((m & net.notna()).reindex(net.index)),
        SR_net=float(np.sqrt(PPY) * n.mean() / n.std()),
        SR_gross=float(np.sqrt(PPY) * g.mean() / g.std()) if g.std() > 0 else np.nan,
        ann_ret_net_pct=float(n.mean() * PPY * 100),
        hit_rate=float((n[act] > 0).mean()) if act.any() else np.nan,
        beta=beta,
        mean_pos=float(e.mean()),
        held_pct=float(act.mean() * 100),
    )

def episode_blocks(mask, net):
    """List of per-episode pnl arrays for the block bootstrap."""
    m = (mask.fillna(False) & net.notna()).values
    vals = net.values
    out, cur = [], []
    for i, flag in enumerate(m):
        if flag:
            cur.append(vals[i])
        elif cur:
            out.append(np.array(cur)); cur = []
    if cur: out.append(np.array(cur))
    return [b for b in out if len(b) > 0]

def boot_dsr(blocks_a, blocks_b):
    """95% CI and P(delta>0) for SR_a - SR_b, resampling whole regime episodes."""
    if len(blocks_a) < 3 or len(blocks_b) < 3:
        return np.nan, np.nan, np.nan
    out = np.empty(NBOOT)
    for i in range(NBOOT):
        a = np.concatenate([blocks_a[k] for k in RNG.integers(0, len(blocks_a), len(blocks_a))])
        b = np.concatenate([blocks_b[k] for k in RNG.integers(0, len(blocks_b), len(blocks_b))])
        sa = np.sqrt(PPY) * a.mean() / a.std() if a.std() > 0 else np.nan
        sb = np.sqrt(PPY) * b.mean() / b.std() if b.std() > 0 else np.nan
        out[i] = sa - sb
    out = out[np.isfinite(out)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out > 0).mean())

rows, drows = [], []
for sname, sig in SIGS.items():
    ex, gross, net = pnl_series(sig)
    live = net.notna()
    full = cell_stats(ex, net, gross, live)
    rows.append(dict(signal=sname, dim="ALL", state="all", window="full",
                     **SMETA[sname], **full))
    for dim, (b, tname, fname) in REGIMES.items():
        lab = REG_LAB[dim]
        mt, mf = (lab == True), (lab == False)
        for win, kw in (("full", {}), ("IS", dict(end=IS_END)), ("OOS", dict(start=OOS_START))):
            st = cell_stats(ex, net, gross, mt, **kw)
            sf = cell_stats(ex, net, gross, mf, **kw)
            rows.append(dict(signal=sname, dim=dim, state=tname, window=win, **SMETA[sname], **st))
            rows.append(dict(signal=sname, dim=dim, state=fname, window=win, **SMETA[sname], **sf))
            d = dict(signal=sname, dim=dim, window=win,
                     state_a=tname, state_b=fname,
                     SR_a=st["SR_net"], SR_b=sf["SR_net"],
                     dSR=st["SR_net"] - sf["SR_net"],
                     n_a=st["n_bars"], n_b=sf["n_bars"],
                     ep_a=st["n_episodes"], ep_b=sf["n_episodes"],
                     hit_a=st["hit_rate"], hit_b=sf["hit_rate"],
                     beta_a=st["beta"], beta_b=sf["beta"],
                     pos_a=st["mean_pos"], pos_b=sf["mean_pos"],
                     enabled=SMETA[sname]["enabled"])
            if win == "full":
                na = net.where(mt); nb = net.where(mf)
                lo, hi, p = boot_dsr(episode_blocks(mt, net), episode_blocks(mf, net))
                # Welch t on the daily-mean difference, for reference only —
                # it treats persistent-regime days as independent and so is
                # optimistic; the episode bootstrap above is the honest one.
                a, bb = na.dropna().values, nb.dropna().values
                if len(a) > 2 and len(bb) > 2:
                    se = np.sqrt(a.var(ddof=1)/len(a) + bb.var(ddof=1)/len(bb))
                    tstat = float((a.mean() - bb.mean()) / se) if se > 0 else np.nan
                else:
                    tstat = np.nan
                d.update(dSR_ci_lo=lo, dSR_ci_hi=hi, p_dSR_gt0=p, welch_t=tstat)
            drows.append(d)

tbl = pd.DataFrame(rows)
dlt = pd.DataFrame(drows)
tbl.to_csv(f"{OUT}/regime_cells.csv", index=False)
dlt.to_csv(f"{OUT}/regime_deltas.csv", index=False)
print(f"\nwrote {len(tbl)} cells, {len(dlt)} deltas")

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300)

print("\n=== baseline: full-sample, no conditioning (roll-adjusted, real cost, committed sign) ===")
base = tbl[tbl.dim == "ALL"].set_index("signal")
print(base[["enabled","sign","recommended","first","n_bars","SR_net","SR_gross",
            "ann_ret_net_pct","hit_rate","beta","mean_pos","held_pct"]].round(3).to_string())

for dim in REGIMES:
    print(f"\n=== {dim}  (full sample) ===")
    d = dlt[(dlt.dim == dim) & (dlt.window == "full")].set_index("signal")
    print(d[["state_a","SR_a","n_a","ep_a","state_b","SR_b","n_b","ep_b","dSR",
             "dSR_ci_lo","dSR_ci_hi","p_dSR_gt0","welch_t","hit_a","hit_b",
             "beta_a","beta_b"]].round(3).to_string())

print("\n=== IS / OOS sign agreement of dSR (full-sample dSR vs IS and OOS) ===")
piv = dlt.pivot_table(index=["signal","dim"], columns="window", values="dSR")
piv["sign_agree_IS_OOS"] = np.sign(piv.get("IS")) == np.sign(piv.get("OOS"))
piv["agree_with_full"] = (np.sign(piv.get("IS")) == np.sign(piv.get("full"))) & \
                         (np.sign(piv.get("OOS")) == np.sign(piv.get("full")))
print(piv.round(3).to_string())
print("\nIS/OOS dSR sign agreement rate: "
      f"{piv['sign_agree_IS_OOS'].mean():.1%} of {len(piv)} signal x regime pairs")

print("\n=== how many cells look 'significant', vs how many chance predicts ===")
f = dlt[(dlt.window == "full") & (dlt.signal.str.startswith("_") == False)]
for label, col, thr in (("|welch_t| > 2", "welch_t", 2.0),):
    k = int((f[col].abs() > thr).sum())
    print(f"  {label}: {k} of {len(f)} cells "
          f"(chance at 5% two-sided ≈ {0.05*len(f):.1f})")
k_boot = int(((f.p_dSR_gt0 > 0.975) | (f.p_dSR_gt0 < 0.025)).sum())
print(f"  episode-bootstrap 95% CI excludes 0: {k_boot} of {len(f)} cells "
      f"(chance ≈ {0.05*len(f):.1f})")

# ── 4. multiple testing + effect-size framing ───────────────────────────────
print("\n=== cells whose episode-bootstrap 95% CI excludes 0 (full sample) ===")
sig_cells = f[(f.p_dSR_gt0 > 0.975) | (f.p_dSR_gt0 < 0.025)]
if len(sig_cells):
    cols = ["signal","dim","state_a","SR_a","state_b","SR_b","dSR","dSR_ci_lo","dSR_ci_hi","p_dSR_gt0","ep_a","ep_b"]
    print(sig_cells[cols].round(3).to_string(index=False))
    j = piv.reset_index()
    for _, r in sig_cells.iterrows():
        q = j[(j.signal == r.signal) & (j.dim == r.dim)]
        if len(q):
            q = q.iloc[0]
            print(f"    {r.signal} x {r.dim}: dSR full {q['full']:+.3f} | IS {q['IS']:+.3f} | OOS {q['OOS']:+.3f}"
                  f"  -> IS/OOS agree: {bool(np.sign(q['IS']) == np.sign(q['OOS']))}")
else:
    print("  none")

# Benjamini-Hochberg on the two-sided bootstrap p-values
pv = f.dropna(subset=["p_dSR_gt0"]).copy()
pv["p2"] = 2 * np.minimum(pv.p_dSR_gt0, 1 - pv.p_dSR_gt0)
pv = pv.sort_values("p2").reset_index(drop=True)
pv["bh"] = 0.10 * (pv.index + 1) / len(pv)
n_bh = int((pv.p2 <= pv.bh).sum())
print(f"\nBenjamini-Hochberg at FDR 10%: {n_bh} of {len(pv)} cells survive. "
      f"smallest two-sided p = {pv.p2.min():.3f}")

print("\n=== effect size vs measurement precision (full sample, real signals only) ===")
w = (f.dSR_ci_hi - f.dSR_ci_lo).dropna()
print(f"  |dSR| : median {f.dSR.abs().median():.3f}  max {f.dSR.abs().max():.3f}")
print(f"  95% CI width on dSR : median {w.median():.3f}  min {w.min():.3f}")
print("  -> a split is only actionable if |dSR| is large relative to the CI width")

# ── 5. plots ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARY = ["igrea", "stlfsi", "epu", "nfci"]
hm = (f[f.dim.isin(PRIMARY)]
      .pivot_table(index="signal", columns="dim", values="dSR")[PRIMARY])
order = [n for n in SIGS if not n.startswith("_")]
hm = hm.reindex(order)
lab = {"igrea": "igrea\nhigh−low", "stlfsi": "STLFSI\nstress−calm",
       "epu": "EPU\nhigh−low", "nfci": "NFCI\ntight−loose"}

fig, axes = plt.subplots(1, 2, figsize=(17, 6.6),
                         gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
im = ax.imshow(hm.values, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
ax.set_xticks(range(len(PRIMARY))); ax.set_xticklabels([lab[c] for c in PRIMARY], fontsize=9)
ax.set_yticks(range(len(hm))); ax.set_yticklabels(
    [n + ("" if SMETA[n]["enabled"] else "  (retired)") for n in hm.index], fontsize=8)
for i in range(hm.shape[0]):
    for k in range(hm.shape[1]):
        v = hm.values[i, k]
        if np.isfinite(v):
            ax.text(k, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.75 else "black")
ax.set_title("ΔSR_net by macro regime (state A − state B)\nroll-adjusted, real cost, committed sign",
             fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.035, label="ΔSR")

ax = axes[1]
d = f[f.dim.isin(PRIMARY)].dropna(subset=["dSR_ci_lo"]).copy()
d["k"] = d.signal + " | " + d.dim
d = d.sort_values("dSR")
y = np.arange(len(d))
ax.errorbar(d.dSR, y, xerr=[d.dSR - d.dSR_ci_lo, d.dSR_ci_hi - d.dSR],
            fmt="o", ms=3.5, lw=0.9, capsize=2, color="#33628d",
            ecolor="#9bb7cd")
ax.axvline(0, color="black", lw=1)
ax.set_yticks(y); ax.set_yticklabels(d.k, fontsize=6.5)
ax.set_xlabel("ΔSR_net  (95% CI, bootstrap over whole regime episodes)")
ax.set_title(f"{len(d)} signal × regime splits — {n_bh} survive BH at FDR 10%", fontsize=11)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/regime_dsr.png", dpi=115)
print(f"\nwrote {OUT}/regime_dsr.png")

# IS vs OOS scatter of dSR
fig, ax = plt.subplots(figsize=(6.6, 6.2))
p2 = piv.reset_index()
p2 = p2[p2.dim.isin(PRIMARY) & ~p2.signal.str.startswith("_")]
for dim, mk in zip(PRIMARY, ["o", "s", "^", "D"]):
    q = p2[p2.dim == dim]
    ax.scatter(q["IS"], q["OOS"], marker=mk, s=42, alpha=0.85, label=dim)
lim = 1.8
ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
ax.plot([-lim, lim], [-lim, lim], ls="--", color="grey", lw=0.8)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("ΔSR in-sample (≤2018)"); ax.set_ylabel("ΔSR out-of-sample (2019+)")
agree = float((np.sign(p2["IS"]) == np.sign(p2["OOS"])).mean())
ax.set_title(f"Does a regime split persist?\nsign agreement {agree:.0%} of {len(p2)} splits "
             f"(coin flip = 50%)", fontsize=11)
ax.legend(fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/regime_is_oos.png", dpi=120)
print(f"wrote {OUT}/regime_is_oos.png")
