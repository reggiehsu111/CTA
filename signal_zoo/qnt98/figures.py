"""QNT-98 figures: argument-sweep robustness, PnL and transaction cost."""
import sys, io, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
SWEEP = "/home/ubuntu/mtx/signal_zoo/macro_windows/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))   # noqa: S102
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cta

OUT = "/home/ubuntu/mtx/signal_zoo/qnt98"
_ns = io.open(f"{OUT}/neighbourhood_sweep.py", encoding="utf-8").read()
exec(compile(_ns.split("# ── Reproduction check")[0]
             .split("import numpy as np, pandas as pd\nimport cta")[1]
             .replace("from cta.signals import _operators as ops", ""), "<palette>", "exec"))
from cta.signals import _operators as ops   # re-import after the palette exec

NB = pd.read_csv(f"{OUT}/neighbourhood_sweep.csv")
NU = pd.read_csv(f"{OUT}/null_circular_shift.csv")
C  = NB[NB.role == "candidate"].copy(); K = NB[NB.role == "control"].copy()
CL = C[C.is_claim == 1].set_index("series")
cat = cta.macro_catalog()

def raw_series(sid, kind):
    if kind == "yoy":
        per = {"M": 12, "Q": 4, "D": 252, "W": 52}.get(cat.loc[sid, "freq"], 12)
        return ctx.macro_yoy(sid, per).astype(float)
    return ctx.macro(sid).astype(float)

COSTS = {"gross": 0.0, "stub": 20.0, "real": 70.0, "real3x": 210.0}
FEES  = {"gross": 0.0, "stub": 2e-5, "real": 4e-5, "real3x": 12e-5}
ENTRY = {"c2c": A["close"].astype(float), "o2o": _prev_o_adj,
         "day": A["open"].astype(float), "ongap": A["night_close"].astype(float)}

def pnl(sig, variant, lag, sign, start, cost="real"):
    ret = _RET[variant]
    c = COSTS[cost] / (ENTRY[variant] * PV) + FEES[cost]
    pos = sig.reindex(A.index).astype(float).shift(lag) * sign
    g = (pos * ret)
    tc = pos.fillna(0).diff().abs() * c
    n = (g - tc).loc[start:].dropna()
    return n

ORDER = list(CL.index)
PN, NBR = {}, {}
for sid in ORDER:
    r = CL.loc[sid]
    kind = r["kind"]; v = r["variant"]; lag = int(r["shift"])
    conv_start = r["start_date"] if r["variant"] != "c2c" else None
    x = raw_series(sid, kind)
    s = cta.normalize_signal(OPS[r["op"]](x, int(r["window"])).replace([np.inf, -np.inf], np.nan),
                             method="tanh", window=252)
    PN[sid] = pnl(s, v, lag, int(r["sign_IS"]), r["start_date"])
    if sid == "cny_usd":
        for opn, fn in OPS.items():
            for w in WINDOWS:
                row = C[(C.series == sid) & (C.op == opn) & (C.window == w)]
                if not len(row):
                    continue
                s2 = cta.normalize_signal(fn(x, w).replace([np.inf, -np.inf], np.nan),
                                          method="tanh", window=252)
                NBR[f"{opn}|w{w}"] = pnl(s2, v, lag, int(row.sign_IS.iloc[0]), r["start_date"])

BH = A.returns.loc[str(PN["cny_usd"].index.min().date()):].dropna()
LOCC, SWAPC = "#1f77b4", "#d62728"

# ══ FIGURE 1 — argument robustness ════════════════════════════════════════
fig = plt.figure(figsize=(17, 15))
gs = GridSpec(3, 2, figure=fig, hspace=.42, wspace=.24)

# (a) window axis at the claimed operator
a = fig.add_subplot(gs[0, 0])
for sid in ORDER:
    r = CL.loc[sid]
    d = C[(C.series == sid) & (C.op == r["op"])].sort_values("window")
    flip = (d.sign_IS != r["sign_IS"]).any()
    a.plot(d.window, d.SR_net, "o-", ms=3.5, lw=1.4, alpha=.85,
           ls="--" if flip else "-",
           label=f"{sid} ({r['op']}){' ⚠sign flips' if flip else ''}")
    a.scatter([r["window"]], [r["SR_net"]], s=95, facecolors="none", edgecolors="k", zorder=5, lw=1.4)
a.set_xscale("log"); a.set_xticks(list(WINDOWS)); a.set_xticklabels(WINDOWS)
a.axhline(0, color="k", lw=.8); a.set_xlabel("window (bars)"); a.set_ylabel("SR net of real costs")
a.set_title("1. WINDOW axis — sweep w around each published cell\n"
            "black ring = the published cell. It is never the peak; 2 of 11 flip sign inside their own sweep",
            fontsize=10.5)
a.legend(fontsize=6.6, ncol=2, loc="lower right"); a.grid(alpha=.25)

# (b) operator axis
a = fig.add_subplot(gs[0, 1])
prof = C[C.window == 252].groupby(["op", "op_family"]).SR_net.median().reset_index()
prof = prof.sort_values("SR_net")
cols = [LOCC if f == "location" else SWAPC for f in prof.op_family]
a.barh(prof.op, prof.SR_net, color=cols)
a.axvline(0, color="k", lw=.8)
a.set_xlabel("median SR net across the 11 candidate series, w = 252")
a.set_title("2. OPERATOR axis — swap InstMean for InstStdev / InstSkew / …\n"
            "blue = location ops (monotone in x, the published palette);  red = genuine statistic swaps",
            fontsize=10.5)
for i, (o, v, f) in enumerate(zip(prof.op, prof.SR_net, prof.op_family)):
    a.text(v + (.015 if v >= 0 else -.015), i, f"{v:+.2f}", va="center",
           ha="left" if v >= 0 else "right", fontsize=7.5)
a.grid(alpha=.25, axis="x")
a.set_xlim(-0.12, 0.78)
a.text(.42, .12, "every dispersion / shape statistic lands at ~0:\nthe edge is in the LEVEL of the\nmacro series, nowhere else",
       transform=a.transAxes, fontsize=9, color=SWAPC,
       bbox=dict(fc="white", ec=SWAPC, alpha=.9))

# (c) heatmap for the headline candidate
a = fig.add_subplot(gs[1, 0])
h = C[C.series == "cny_usd"].pivot_table(index="op", columns="window", values="SR_net")
h = h.reindex([o for o in OPS if o in h.index])
im = a.imshow(h.values, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
a.set_xticks(range(h.shape[1])); a.set_xticklabels(h.columns)
a.set_yticks(range(h.shape[0]))
a.set_yticklabels([f"{o}" for o in h.index], fontsize=8)
for lab in a.get_yticklabels():
    lab.set_color(LOCC if FAMILY[lab.get_text()] == "location" else SWAPC)
for i in range(h.shape[0]):
    for j in range(h.shape[1]):
        if np.isfinite(h.values[i, j]):
            a.text(j, i, f"{h.values[i,j]:.2f}", ha="center", va="center", fontsize=6.2)
ri, rj = list(h.index).index("dev"), list(h.columns).index(252)
a.add_patch(plt.Rectangle((rj-.5, ri-.5), 1, 1, fill=False, ec="k", lw=2.4))
plt.colorbar(im, ax=a, fraction=.03, pad=.02)
a.set_title("3. The full 16 × 9 neighbourhood of the strongest cell\n"
            "cny_usd / ongap / dev / w252 = 1.04 (boxed) — neighbourhood median 0.34, rank 5 of 144",
            fontsize=10.5)
a.set_xlabel("window")

# (d) claim vs neighbourhood
a = fig.add_subplot(gs[1, 1])
y = np.arange(len(ORDER))
med = C.groupby("series").SR_net.median().reindex(ORDER)
for i, sid in enumerate(ORDER):
    a.plot([med[sid], CL.loc[sid, "SR_net"]], [i, i], color="#999", lw=2, zorder=1)
a.scatter(med.values, y, s=55, color="#555", label="neighbourhood median (144 cells)", zorder=3)
a.scatter(CL.SR_net.reindex(ORDER).values, y, s=70, color="#d62728",
          label="published cell", zorder=3)
for i, sid in enumerate(ORDER):
    rk = int((C[C.series == sid].SR_net > CL.loc[sid, "SR_net"]).sum()) + 1
    a.text(CL.loc[sid, "SR_net"] + .03, i, f"rank {rk}/144", fontsize=7, va="center")
a.set_yticks(y); a.set_yticklabels(ORDER, fontsize=8.5)
a.axvline(0, color="k", lw=.8); a.set_xlabel("SR net of real costs")
a.set_title("4. What the published number shrinks to once the arguments move\n"
            "median shrinkage 0.31 SR; the published cell is rank 5–47 of its own 144",
            fontsize=10.5)
a.set_xlim(-0.05, 1.32)
a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.11), ncol=2, frameon=False)
a.grid(alpha=.25, axis="x")

# (e) best-of-144 vs the circular-shift null
a = fig.add_subplot(gs[2, 0])
bn = NU.groupby(["shift", "series"]).SR_net.max()
a.hist(bn, bins=18, color="#bbb", edgecolor="w",
       label=f"circular-shift NULL, best of 144 (n={len(bn)})")
p90 = bn.quantile(.9)
a.axvline(p90, color="k", ls="--", lw=1.3, label=f"null p90 = {p90:.2f}")
br = C.groupby("series").SR_net.max(); bk = K.groupby("series").SR_net.max()
a.scatter(br.values, np.full(len(br), 3.2), s=70, color="#d62728", zorder=4, label="candidate series")
a.scatter(bk.values, np.full(len(bk), 1.9), s=70, color="#1f77b4", zorder=4, label="control series")
for s_, v_ in br.items():
    a.annotate(s_, (v_, 3.2), rotation=90, fontsize=6.2, ha="center", va="bottom")
a.set_xlabel("best SR net found by sweeping 144 arguments on ONE series")
a.set_ylabel("null draws")
a.set_title("5. THE NULL — sweep the same 144 arguments against circularly shifted returns\n"
            f"null best-of-144: median {bn.median():.2f}, p90 {p90:.2f}, max {bn.max():.2f}. "
            f"Only 4 of 11 candidates clear the p90 — as do 2 of 8 controls.", fontsize=10.5)
a.legend(fontsize=7.5); a.grid(alpha=.25)

# (f) gate pass rate + sign agreement
a = fig.add_subplot(gs[2, 1])
agree = []
for sid in ORDER:
    d = C[C.series == sid]
    agree.append(100 * (d.sign_IS == CL.loc[sid, "sign_IS"]).mean())
a2 = a.twinx()
a.bar(y - .2, agree, .4, color="#7f7f7f", label="sign agreement inside the neighbourhood (%)")
pg = [100 * (C[C.series == s].n_gates == 4).mean() for s in ORDER]
a2.bar(y + .2, pg, .4, color="#2ca02c", label="cells passing all 4 gates (%)")
a.axhline(100 * (NU.sign_IS.notna().mean() if "sign_IS" in NU else 50), color="none")
a2.axhline(100 * (NU.n_gates == 4).mean(), color="#2ca02c", ls="--", lw=1.3)
a2.set_ylim(0, 33)
a.set_xlim(-.75, len(ORDER) + 1.6)
a2.text(len(ORDER) - .2, 100 * (NU.n_gates == 4).mean() + .5,
        f"circular-shift null 4-gate rate {100*(NU.n_gates==4).mean():.1f}%",
        fontsize=8, color="#2ca02c", ha="left")
a.axhline(50, color="#7f7f7f", ls=":", lw=1.2)
a.set_xticks(y); a.set_xticklabels(ORDER, rotation=35, ha="right", fontsize=8)
a.set_ylabel("sign agreement %"); a2.set_ylabel("4-gate pass %")
a.set_ylim(0, 105)
a.set_title("6. Sign stability under the sweep, and gate pass rate vs the null\n"
            "twd_usd and kr_kospi disagree with their own published sign on >half the grid",
            fontsize=10.5)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.set_ylim(0, 150)
a.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left", ncol=2, framealpha=.95)
a.grid(alpha=.25, axis="y")

fig.suptitle("QNT-98 — robustness by sweeping the ARGUMENTS around every working macro cell "
             "(16 operators × 9 windows = 144 per series)", fontsize=13.5, y=.995)
fig.savefig(f"{OUT}/qnt98_argument_robustness.png", dpi=115, bbox_inches="tight")
print("-> qnt98_argument_robustness.png")

# ══ FIGURE 2 — PnL + transaction cost ═════════════════════════════════════
P = pd.DataFrame(PN)
NIGHT = [s_ for s_ in ORDER if s_ != "igrea"]
Pn = P[NIGHT].dropna(how="all")
CM = Pn.corr(); _ev = np.linalg.eigvalsh(CM.values); _ev = _ev[_ev > 0]
NEFF = float(_ev.sum() ** 2 / (_ev ** 2).sum())
SHARE = (Pn.loc["2025-01-01":].sum() / Pn.sum()).sort_values()

fig2 = plt.figure(figsize=(17, 16))
gs = GridSpec(3, 2, figure=fig2, hspace=.40, wspace=.22)
CMAP11 = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[[0,2,4,6,8,10,12,14,16,18,1]]

a = fig2.add_subplot(gs[0, 0])
for i, sid in enumerate(ORDER):
    a.plot(PN[sid].index, PN[sid].cumsum() * 100, lw=1.3, color=CMAP11[i],
           ls=":" if sid == "igrea" else "-",
           label=f"{sid} (SR {CL.loc[sid,'SR_net']:.2f})")
a.plot(BH.index, BH.cumsum() * 100, color="k", lw=2, ls="--", label="MTX buy & hold (roll-adj)")
a.axvspan(pd.Timestamp("2025-01-01"), PN["cny_usd"].index.max(), color="#ffd54f", alpha=.3, zorder=0)
a.set_ylabel("cumulative return, % (sum of daily)"); a.axhline(0, color="k", lw=.8)
a.set_title("1. Net PnL of the 11 published cells — real costs, sign frozen on IS\n"
            "igrea (dotted) is the only 25-year name; the other 10 start 2017-05-16. "
            "Shaded = 2025-26", fontsize=10.5)
a.legend(fontsize=7, ncol=2); a.grid(alpha=.25)

a = fig2.add_subplot(gs[0, 1])
for k, v in NBR.items():
    a.plot(v.index, v.cumsum() * 100, lw=.7, alpha=.35,
           color=LOCC if FAMILY[k.split("|")[0]] == "location" else SWAPC)
a.plot(PN["cny_usd"].index, PN["cny_usd"].cumsum() * 100, color="k", lw=2.4, label="published cell dev/w252")
a.axhline(0, color="k", lw=.8)
a.set_title("2. cny_usd — the published cell against all 144 of its own arguments\n"
            "blue = location ops, red = InstStdev/InstSkew-family swaps", fontsize=10.5)
a.set_ylabel("cumulative return, %"); a.legend(fontsize=8); a.grid(alpha=.25)

a = fig2.add_subplot(gs[1, 0])
lad = CL.loc[ORDER, ["SR_gross", "SR_stub", "SR_net", "SR_real3x"]]
xs = np.arange(len(ORDER)); w_ = .2
for i, (c_, lab) in enumerate(zip(lad.columns,
        ["gross", "stub 20+2e-5", "REAL 70+4e-5", "3× real"])):
    b = a.bar(xs + (i - 1.5) * w_, lad[c_], w_, label=lab)
a.set_xticks(xs); a.set_xticklabels(ORDER, rotation=35, ha="right", fontsize=8)
a.set_ylabel("SR"); a.grid(alpha=.25, axis="y")
a.set_title("3. Transaction-cost ladder — costs are NOT the binding constraint\n"
            "median turnover 8.6×/yr, held 99.6% of days; gross→real drag 0.025 SR", fontsize=10.5)
a.legend(fontsize=8)

a = fig2.add_subplot(gs[1, 1])
yr = pd.DataFrame({sid: PN[sid].groupby(PN[sid].index.year).apply(
        lambda s: np.sqrt(252) * s.mean() / s.std() if len(s) > 20 and s.std() > 0 else np.nan)
        for sid in ORDER}).T.reindex(ORDER)
im = a.imshow(yr.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
a.set_xticks(range(yr.shape[1])); a.set_xticklabels(yr.columns, rotation=90, fontsize=7)
a.set_yticks(range(len(ORDER))); a.set_yticklabels(ORDER, fontsize=8)
for i in range(yr.shape[0]):
    for j in range(yr.shape[1]):
        if np.isfinite(yr.values[i, j]):
            a.text(j, i, f"{yr.values[i,j]:.1f}", ha="center", va="center", fontsize=5.8)
plt.colorbar(im, ax=a, fraction=.03, pad=.02)
a.set_title("4. Per-year SR of the published cells\n"
            "the night-era names have 10 calendar years, so ≥65%-positive-years is 7 of 10",
            fontsize=10.5)

# (5) era concentration
a = fig2.add_subplot(gs[2, 0])
xs = np.arange(len(SHARE))
a.bar(xs, SHARE.values * 100, color="#ffa000")
a.axhline(20, color="k", ls="--", lw=1.2)
a.text(len(xs) - .4, 21.5, "2025-26 is 20% of the 10-year window", fontsize=8.5, ha="right")
a.set_xticks(xs); a.set_xticklabels(SHARE.index, rotation=35, ha="right", fontsize=8)
a.set_ylabel("% of the cell's whole 10-year net PnL")
for i, v in enumerate(SHARE.values):
    a.text(i, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=7.5)
a.set_title("5. WHEN the PnL was earned — median 55% of it in the last 20 months\n"
            "median SR across the 10 cells: 2017-21 +0.40 | 2022-24 +0.49 | 2025-26 +1.56",
            fontsize=10.5)
a.grid(alpha=.25, axis="y")

# (6) redundancy of the 10 'independent' cells
a = fig2.add_subplot(gs[2, 1])
im = a.imshow(CM.values, cmap="RdBu_r", vmin=-1, vmax=1)
a.set_xticks(range(len(CM))); a.set_xticklabels(CM.columns, rotation=90, fontsize=7.5)
a.set_yticks(range(len(CM))); a.set_yticklabels(CM.index, fontsize=7.5)
for i in range(len(CM)):
    for j in range(len(CM)):
        a.text(j, i, f"{CM.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
plt.colorbar(im, ax=a, fraction=.04, pad=.02)
a.set_title(f"6. The 10 night-era cells are not 10 bets\n"
            f"mean pairwise net-PnL corr +{CM.values[~np.eye(len(CM),dtype=bool)].mean():.2f}, "
            f"n_eff = {NEFF:.1f} independent sleeves", fontsize=10.5)

fig2.suptitle("QNT-98 — PnL, per-year robustness, cost and era-concentration of the published macro cells",
              fontsize=13.5, y=.99)
fig2.savefig(f"{OUT}/qnt98_pnl_costs.png", dpi=115, bbox_inches="tight")
print("-> qnt98_pnl_costs.png")
