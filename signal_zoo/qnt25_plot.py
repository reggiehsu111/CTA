"""QNT-25 evidence figure: effective n, the noise floor, and igrea's redundancy."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

Z = "/home/ubuntu/mtx/signal_zoo"
g1 = pd.read_csv(f"{Z}/macro_sweep/full_sweep.csv")
g2 = pd.read_csv(f"{Z}/macro_windows/window_sweep_full.csv"); g2 = g2[g2.regime == "full"]
p2 = g2.pivot_table(index=["series", "transform", "window"], columns="variant", values="SR_net").reset_index()
p2["d"] = p2["day"] - p2["c2c"]
g3 = pd.read_csv(f"{Z}/macro_windows/slow_window_sweep.csv")
p3 = g3.pivot_table(index=["series", "transform", "window"], columns="variant", values="SR_net").reset_index()
p3["d"] = p3["day"] - p3["c2c"]
red = pd.read_csv(f"{Z}/qnt25_family_redundancy.csv")
C = pd.read_csv(f"{Z}/qnt25_igrea_pnl_corr.csv", index_col=0)
fam = pd.read_csv(f"{Z}/macro_combo/combo_family_sweep.csv")

fig, ax = plt.subplots(2, 3, figsize=(19, 10))

# 1 — nominal cells vs effective tests
a = ax[0][0]
lbl = ["QNT-12\n522 cells", "QNT-14\n198 cells", "QNT-18 slow\n378 cells", "QNT-18 reg.\n11 signals"]
nom, srs, eff = [522, 198, 378, 11], [29, 11, 21, 11], [45.6, 21.0, 38.1, 11]
x = np.arange(4)
a.bar(x - .26, nom, .25, label="宣稱的 cell 數", color="tab:red", alpha=.75)
a.bar(x, eff, .25, label="ICC 有效檢定數 n_eff", color="tab:orange")
a.bar(x + .26, srs, .25, label="來源序列數 (誠實的 n)", color="tab:blue")
a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(lbl, fontsize=8)
a.set_title("每個 grid 真正有幾個獨立檢定\n18 個 transform 是同一個序列的近似複本")
a.set_ylim(6, 3000); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=.3, axis="y")
for i, (n, e) in enumerate(zip(nom, eff)):
    a.text(i - .26, n * 1.15, f"{n/e:.0f}×", ha="center", fontsize=9, color="tab:red", weight="bold")

# 2 — QNT-14 dSR: cells vs series
a = ax[0][1]
a.hist(p2.d, bins=32, color="lightsteelblue", edgecolor="w", label="198 cells")
ser = p2.groupby("series").d.median()
for i, (nm, v) in enumerate(ser.items()):
    a.plot(v, 2 + (i % 4) * 1.6, "o", color="tab:red", ms=7, zorder=5)
a.axvline(0, color="k", lw=1)
a.axvline(p2.d.median(), color="tab:blue", ls="--", lw=2, label=f"cell 中位數 +{p2.d.median():.3f} (QNT-14 標題數字)")
a.axvline(ser.median(), color="tab:red", ls="--", lw=2, label=f"per-series 中位數 +{ser.median():.3f}, n=11")
se = 0.217
a.axvspan(-se, se, color="gray", alpha=.18, label=f"單一 ΔSR 的 ±1 SE ({se:.2f})")
a.set_title(f"QNT-14 的 day vs c2c: 紅點 = 11 個來源序列\nWilcoxon p={stats.wilcoxon(ser).pvalue:.3f} — 不顯著")
a.set_xlabel("ΔSR (day@1 − c2c@2)"); a.legend(fontsize=7); a.grid(alpha=.3)

# 3 — noise floor
a = ax[0][2]
grids = ["QNT-12\n(SR_full)", "QNT-12\n(SR_of_SR)", "QNT-14\n(SR_c2c)", "QNT-18 slow\n(SR_c2c)", "QNT-18 reg.\n(SR_c2c)"]
sd = [g1.SR_full.std(), g1.SR_of_SR.std(), p2.c2c.std(), p3.c2c.std(), 0.257]
se_ = [0.200, 0.200, 0.201, 0.202, 0.210]
x = np.arange(len(grids))
a.bar(x - .18, sd, .35, label="cell 之間的 sd(SR) — 排名靠的就是這個", color="tab:blue")
a.bar(x + .18, se_, .35, label="單一 Sharpe 的 SE (~25 年)", color="tab:red", alpha=.8)
a.set_xticks(x); a.set_xticklabels(grids, fontsize=8)
a.set_title("噪音下限: 除了 registered，所有 macro grid 的\ncell 間離散度都「小於」估計誤差")
a.set_ylim(0, 0.335); a.legend(fontsize=8, loc="upper left"); a.grid(alpha=.3, axis="y")
for i, (s_, e_) in enumerate(zip(sd, se_)):
    a.text(i, max(s_, e_) + .008, f"{s_/e_:.2f}×", ha="center", fontsize=9,
           color="tab:red" if s_ / e_ < 1 else "tab:green", weight="bold")

# 4 — igrea family correlation
a = ax[1][0]
im = a.imshow(C.values, vmin=0, vmax=1, cmap="RdYlGn_r")
a.set_xticks(range(len(C))); a.set_yticks(range(len(C)))
a.set_xticklabels(C.columns, rotation=90, fontsize=6); a.set_yticklabels(C.index, fontsize=6)
off = C.values[np.triu_indices_from(C, 1)]
a.set_title(f"igrea 的 18 個 cell 之間的損益相關\n平均 {off.mean():.2f}, PC1 佔 77%, n_eff = 1.63")
plt.colorbar(im, ax=a, fraction=.046)

# 5 — sign-selection bias
a = ax[1][1]
psx = g1.groupby("series").agg(IS=("SR_IS", "median"), full=("SR_full", "median"), OOS=("SR_OOS", "median"))
a.boxplot([psx.IS, psx.full, psx.OOS], labels=["IS (符號在此擬合)", "全樣本", "OOS (符號凍結)"])
for i, c in enumerate(["IS", "full", "OOS"]):
    a.plot(np.full(len(psx), i + 1) + np.random.uniform(-.09, .09, len(psx)), psx[c], "o", ms=4, alpha=.55, color="tab:blue")
a.axhline(0, color="k", lw=1)
a.set_ylabel("每序列的中位數 SR (n=29)")
a.set_title("QNT-12 的「29 個序列有 26 個為正」是符號擬合的產物\n"
            f"IS 28/29 正 → OOS 11/29 正 (binom p={stats.binomtest(11,29,.5).pvalue:.2f})")
a.grid(alpha=.3, axis="y")

# 6 — EW3 stability gate vs its own error bar
a = ax[1][2]
srsr, se_srsr = 0.542, 0.214
a.bar(["EW3 (QNT-16)"], [srsr], yerr=[1.96 * se_srsr], capsize=12, color="tab:blue", width=.35)
a.scatter(np.full(len(fam), 0) + np.random.uniform(-.2, .2, len(fam)), fam.SR_of_SR,
          color="tab:gray", s=28, zorder=5, label="同一籃子的 18 個 transform cell")
a.axhline(0.6, color="tab:red", ls="--", lw=2, label="門檻 SR_of_SR > 0.6")
a.axhline(0, color="k", lw=1)
a.set_ylabel("SR_of_SR"); a.set_ylim(-0.1, 1.05)
a.set_title("EW3 的 0.542 與門檻 0.60 相差 0.27 SE\n95% CI 同時包含 0 與 0.6 — 分不出來")
a.legend(fontsize=8); a.grid(alpha=.3, axis="y")

plt.tight_layout()
plt.savefig(f"{Z}/qnt25_effective_n.png", dpi=105)
print("saved", f"{Z}/qnt25_effective_n.png")
