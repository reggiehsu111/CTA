"""QNT-100 — one figure: the leak, who it caught, and what the fix costs."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/home/ubuntu/mtx/signal_zoo/qnt100"
BLUE, RED, GREEN, GREY = "#4C72B0", "#C44E52", "#55A868", "#8C8C8C"

R = pd.read_csv(f"{OUT}/gate_leak_masks_v2.csv")
LO = R[R.kind == "long_only"]
GR = [("QNT-99 事件行事曆", "regated_event_calendar_sweep.csv"),
      ("QNT-99 發布驚奇",   "regated_event_surprise_full.csv"),
      ("QNT-99 put OI",     "regated_put_oi_sweep_full.csv"),
      ("QNT-98 鄰域",       "regated_neighbourhood_sweep.csv"),
      ("QNT-94 macro",      "regated_window_sweep_full.csv"),
      ("QNT-94 slow",       "regated_slow_window_sweep.csv"),
      ("QNT-94 registered", "regated_registered_window_sweep.csv")]
frames = []
for lab, f in GR:
    d = pd.read_csv(f"{OUT}/{f}")
    aw = [c for c in ("mean_abs_w", "abs_exec_w", "full_abs_exec_w") if c in d][0]
    bc = [c for c in ("beta", "full_beta") if c in d][0]
    frames.append(d.assign(abs_w=d[aw], beta_=d[bc], grid=lab,
                           old=d.gate_srsr & d.gate_posyr & d.gate_nyr & d.gate_beta_raw)
                   [["grid", "abs_w", "beta_", "beta_per_w", "old", "passes",
                     "gate_srsr", "gate_posyr", "gate_nyr"]])
A = pd.concat(frames, ignore_index=True)
base = A.gate_srsr & A.gate_posyr & A.gate_nyr

fig, ax = plt.subplots(1, 4, figsize=(22, 5.0))

# ── 1. the leak, as a function of exposure ────────────────────────────────
t = LO.groupby("frac").agg(old=("passes_old", "mean"), new=("passes_new", "mean"),
                           beta=("beta", "mean"), bpw=("beta_per_w", "mean"))
ax[0].plot(t.index * 100, t.old * 100, "o-", color=RED, lw=2, label="舊規則 |beta| < 0.15")
ax[0].plot(t.index * 100, t.new * 100, "s-", color=GREEN, lw=2, label="新規則 |beta/mean|w|| < 0.15")
ax[0].axhline(0, color="k", lw=.6)
ax[0].set_xscale("log")
ax[0].set(xlabel="持倉夜盤比例 mean|exec_w|  (%, log)", ylabel="四道門檻全過的比例 (%)",
          ylim=(-1, 20),
          title="1. 純隨機、零資訊的夜盤多頭遮罩\n"
                f"舊規則放行 {LO.passes_old.mean()*100:.1f}%（5–10% 曝險時 9–16%），新規則 "
                f"{LO.passes_new.mean()*100:.1f}%")
ax[0].legend(fontsize=8, loc="upper right")
a2 = ax[0].twinx()
a2.plot(t.index * 100, t.beta, ":", color=RED, lw=1.4)
a2.plot(t.index * 100, t.bpw, ":", color=GREEN, lw=1.4)
a2.axhline(0.15, color=GREY, ls="--", lw=1)
a2.set_ylabel("beta（虛線）", fontsize=8); a2.set_ylim(0, 1.15)
a2.text(2.2, 0.17, "門檻 0.15", fontsize=7, color=GREY)
a2.text(2.2, 0.93, "beta/|w| ≈ 1.0 各曝險皆然", fontsize=7, color=GREEN)
a2.text(30, 0.30, "beta 隨曝險縮小", fontsize=7, color=RED)

# ── 2. who the old rule let through: beta vs beta at unit exposure ────────
op = A[A.old]
sc = ax[1].scatter(op.beta_.abs(), op.beta_per_w.abs(), c=op.abs_w, cmap="viridis",
                   s=16, alpha=.75, vmin=0, vmax=1)
ax[1].axhline(0.15, color=GREEN, ls="--", lw=1.4)
ax[1].axvline(0.15, color=RED, ls="--", lw=1.4)
ax[1].set_yscale("log")
ax[1].set(xlabel="|beta|（舊規則：全數 < 0.15，故全部在左側）",
          ylabel="|beta| / mean|exec_w|（新規則，log）",
          title=f"2. 舊規則的 {len(op)} 個倖存者\n"
                f"曝險 <0.2 者 {(op.abs_w<0.2).sum()} 個：中位 |beta| 0.037，"
                f"但單位曝險 beta 0.60")
cb = plt.colorbar(sc, ax=ax[1]); cb.set_label("mean|exec_w|", fontsize=8)
ax[1].text(0.005, 0.42, "縮小曝險偽裝的指數多頭", fontsize=8, color=RED)
ax[1].text(0.078, 0.0035, "真正中性、滿倉的訊號", fontsize=8, color=GREEN)

# ── 3. what the fix costs, and the threshold choice ───────────────────────
ths = [0.15, 0.20, 0.225, 0.25, 0.30]
surv = [int((base & (A.beta_per_w.abs() < th)).sum()) for th in ths]
spar = [int((base & (A.beta_per_w.abs() < th) & (A.abs_w < 0.2)).sum()) for th in ths]
old_n = int((base & (A.beta_.abs() < 0.15)).sum())
old_s = int((base & (A.beta_.abs() < 0.15) & (A.abs_w < 0.2)).sum())
x = np.arange(len(ths) + 1)
tot = surv + [old_n]; spr = spar + [old_s]
ax[2].bar(x, np.array(tot) - np.array(spr), color=GREEN, label="曝險 ≥0.2（滿倉、真中性）")
ax[2].bar(x, spr, bottom=np.array(tot) - np.array(spr), color=RED, label="曝險 <0.2（稀疏，漏洞高風險）")
for xi, (a_, s_) in enumerate(zip(tot, spr)):
    ax[2].text(xi, a_ + 6, f"{a_}\n({s_} 稀疏)", ha="center", fontsize=8)
ax[2].set_xticks(x)
ax[2].set_xticklabels([f"|b/w|\n<{th}" for th in ths] + ["|beta|\n<0.15\n(舊)"], fontsize=8)
ax[2].axhline(old_n, color=GREY, ls=":", lw=1)
ax[2].set(ylabel="8,930 個歷史 cell 中的倖存者數", ylim=(0, 470),
          title="3. 門檻選擇：8,930 cells，7 個已發表 grid\n"
                "0.15 收緊四成；0.30 維持總數但稀疏者 92→16")
ax[2].legend(fontsize=8, loc="upper left")

# ── 4. the honest caveat: the ratio's own noise vs exposure ───────────────
E = pd.read_csv(f"{OUT}/power_informed_masks.csv")
TS = R[R.kind == "two_sided"]
k = 0.084
for d, lab, c in ((E, "有真實 edge 的稀疏雙邊書", BLUE), (TS, "純隨機雙邊書", GREY)):
    q = d.groupby("frac").beta_per_w.std()
    ax[3].plot(q.index, q.values, "o", color=c, ms=7, label=lab)
w = np.linspace(0.02, 1.0, 200)
ax[3].plot(w, k / np.sqrt(w), "-", color=RED, lw=1.6, label=r"擬合 $0.084/\sqrt{w}$")
ax[3].axhline(0.15, color=GREEN, ls="--", lw=1.4)
ax[3].axvline(0.31, color=GREEN, ls=":", lw=1.4)
ax[3].set_xscale("log"); ax[3].set_yscale("log")
ax[3].set(xlabel="mean|exec_w|（log）", ylabel="sd(beta/mean|w|)，真實 beta = 0（log）",
          title="4. 但這個比值自己也有雜訊\n"
                "曝險 <0.31 時 sd 超過 0.15 門檻本身：\n"
                "真正中性的書在 10% 曝險下有 57% 被誤殺")
ax[3].legend(fontsize=8, loc="lower left")
ax[3].text(0.33, 0.5, "0.31\n可測下限", fontsize=8, color=GREEN)
ax[3].text(0.035, 0.16, "門檻 0.15", fontsize=8, color=GREEN)

plt.tight_layout(); plt.savefig(f"{OUT}/qnt100_gate_fix.png", dpi=130, bbox_inches="tight")
print("wrote qnt100_gate_fix.png")
print(pd.DataFrame({"thresh": ths + ["old |beta|<0.15"], "survivors": tot, "sparse": spr}).to_string(index=False))
