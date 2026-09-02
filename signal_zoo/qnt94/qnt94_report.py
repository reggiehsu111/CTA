"""QNT-94: restate the QNT-78 power table in n_eff, and one figure."""
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/ubuntu/mtx/signal_zoo/qnt94"
R = json.load(open(f"{OUT}/qnt94_neff.json"))
Zc, SD = 2.80, 0.13

# n_eff / S ratio -- the usable rule of thumb
rat = {k: v["neff_eig"]/v["S"] for k,v in R.items()}
K = float(np.mean(list(rat.values())))
print("n_eff / S per grid:", {k: round(v,3) for k,v in rat.items()})
print(f"mean ratio K = {K:.3f}   ->  d_min ~ {Zc/np.sqrt(K):.2f}*sd/sqrt(S)"
      f" = {Zc/np.sqrt(K)*SD:.3f}/sqrt(S) at sd=0.13   (inflation factor {1/np.sqrt(K):.2f}x)")

print("\nRESTATED REFERENCE TABLE (sd = 0.13), n_eff = 0.42*S:")
hdr = ["S", 5, 11, 17, 20, 26, 29, 40, 60]
print("  " + " | ".join(f"{h:>7}" for h in hdr))
print("  " + " | ".join([f"{'n_eff':>7}"] + [f"{K*s:7.1f}" for s in hdr[1:]]))
print("  " + " | ".join([f"{'d(rawS)':>7}"] + [f"{Zc*SD/np.sqrt(s):7.3f}" for s in hdr[1:]]))
print("  " + " | ".join([f"{'d(neff)':>7}"] + [f"{Zc*SD/np.sqrt(K*s):7.3f}" for s in hdr[1:]]))
print("\n  S required for a target d, raw vs n_eff form:")
for d in (0.05,0.075,0.10,0.125,0.15,0.20):
    print(f"    d={d:5.3f}   S_raw={int(np.ceil((Zc*SD/d)**2)):4d}   S_true={int(np.ceil((Zc*SD/d)**2/K)):4d}")

# ── figure ────────────────────────────────────────────────────────────────
P1 = pd.read_csv(f"{OUT}/g1_series_pnl_ew.csv", index_col=0, parse_dates=True)
P2 = pd.read_csv(f"{OUT}/g2_series_pnl_c2c_ew.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
for a, P, lab, ne in ((ax[0], P1.dropna(), "GRID 1 — QNT-12, S=29", R["G1 QNT-12 (S=29), EW-of-18"]["neff_eig"]),
                      (ax[1], P2.dropna(), "GRID 2 — QNT-14, S=11", R["G2 QNT-14 (S=11), EW-of-18, c2c"]["neff_eig"])):
    C = P.corr()
    im = a.imshow(C.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    a.set_xticks(range(len(C))); a.set_yticks(range(len(C)))
    a.set_xticklabels(C.columns, rotation=90, fontsize=6); a.set_yticklabels(C.columns, fontsize=6)
    off = C.values[np.triu_indices(len(C),1)]
    a.set_title(f"{lab}\nper-series PnL corr: mean {off.mean():+.3f}   eigenvalue n_eff {ne:.2f}", fontsize=10)
    plt.colorbar(im, ax=a, fraction=0.046)

a = ax[2]
S = np.arange(3, 61)
a.plot(S, Zc*SD/np.sqrt(S), lw=2, label="QNT-78 as written:  $2.80\\,sd/\\sqrt{S}$")
a.plot(S, Zc*SD/np.sqrt(K*S), lw=2, color="crimson",
       label=f"corrected:  $2.80\\,sd/\\sqrt{{n_{{eff}}}}$,  $n_{{eff}}\\!\\approx\\!{K:.2f}S$")
for k, v in R.items():
    if "median" in k: continue
    a.scatter([v["S"]], [Zc*SD/np.sqrt(v["neff_eig"])], color="crimson", zorder=5, s=45)
    a.scatter([v["S"]], [Zc*SD/np.sqrt(v["S"])], color="tab:blue", zorder=5, s=45)
a.axhline(0.089, ls=":", color="k")
a.annotate("QNT-14's claimed dSR +0.089 (pre-floor)\npost-floor it is +0.038",
           (34, 0.093), fontsize=8, ha="right")
a.scatter([11], [0.089], marker="*", s=200, color="k", zorder=6)
a.set_xlabel("S — number of source series"); a.set_ylabel("smallest resolvable d  (SR units, sd=0.13)")
a.set_title("Power floor: raw S overstates power by 1.52$\\times$", fontsize=10)
a.legend(fontsize=8); a.grid(alpha=0.3); a.set_ylim(0, 0.30)
plt.tight_layout(); plt.savefig(f"{OUT}/qnt94_series_neff.png", dpi=110)
print(f"\nwrote {OUT}/qnt94_series_neff.png")

# QNT-14 verdict
v = R["G2 paired d(day-c2c) PnL"]
print(f"\nQNT-14 VERDICT — statistic = per-series median dSR(day-c2c), S=11")
for lab, sd_ in (("house sd 0.13", 0.13), (f"measured sd {v['sr_sd']:.3f}", v["sr_sd"])):
    f_raw, f_eig, f_deff = Zc*sd_/np.sqrt(11), Zc*sd_/np.sqrt(v["neff_eig"]), Zc*sd_/np.sqrt(v["neff_deff"])
    print(f"  [{lab:22s}] floor: raw-S {f_raw:.3f}   n_eff(eig {v['neff_eig']:.2f}) {f_eig:.3f}"
          f"   n_eff(deff {v['neff_deff']:.2f}) {f_deff:.3f}")
    print(f"      claimed +0.089 sits {0.089/f_raw:.2f}x the raw floor -> {0.089/f_eig:.2f}x / {0.089/f_deff:.2f}x the corrected floor")
    print(f"      post-floor +0.038 sits {0.038/f_raw:.2f}x -> {0.038/f_eig:.2f}x / {0.038/f_deff:.2f}x")
