import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0,"/home/ubuntu/mtx")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
OUT="/home/ubuntu/mtx/signal_zoo/qnt99"

# --- beta per unit of exposure, the leak metric -------------------------------
cs=pd.read_csv(f"{OUT}/event_calendar_sweep.csv")
g=cs[(cs.full_SR_of_SR>0.6)&(cs.full_positive_years>=0.65)&(cs.full_beta.abs()<0.15)&(cs.full_n_years>=5)]
es_=pd.read_csv(f"{OUT}/event_surprise_full.csv")
gs=pd.read_csv(f"{OUT}/event_surprise_gated.csv")
print("=== beta per unit of exposure (a pure index bet reads ~1.0) ===")
for nm,d in [("calendar passers",g),("surprise passers",gs)]:
    r=(d.full_beta/d.full_abs_exec_w.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna()
    print(f"  {nm:18s} n={len(d):3d}  median beta/|exec_w| = {r.median():+.2f}  "
          f"[q25 {r.quantile(.25):+.2f}, q75 {r.quantile(.75):+.2f}]  median |exec_w| {d.full_abs_exec_w.median():.3f}")

fig,ax=plt.subplots(1,3,figsize=(16.5,4.8))
# 1 event-surprise IS -> OOS
d=es_.dropna(subset=["SR_IS","SR_OOS"])
ax[0].scatter(d.SR_IS,d.SR_OOS,s=6,alpha=.25,color="#4C72B0")
per=d.groupby("event")[["SR_IS","SR_OOS"]].median()
ax[0].scatter(per.SR_IS,per.SR_OOS,s=70,color="#C44E52",zorder=3,label="per-event median")
lim=[-1.6,1.6]; ax[0].plot(lim,lim,"k--",lw=.8); ax[0].axhline(0,c="k",lw=.6); ax[0].axvline(0,c="k",lw=.6)
ax[0].set(xlim=lim,ylim=lim,xlabel="SR_net in-sample (sign fitted here)",ylabel="SR_net out-of-sample",
          title=f"發布驚奇 (release surprise) 813 cells / 15 events\nIS +{d.SR_IS.median():.3f} → OOS {d.SR_OOS.median():+.3f}, corr {d.SR_IS.corr(d.SR_OOS):+.2f}")
ax[0].legend(fontsize=8)
# 2 event-study t distribution
esd=pd.read_csv(f"{OUT}/event_study.csv")
ax[1].hist(esd.t,bins=30,density=True,color="#55A868",alpha=.75,label=f"320 event×offset×window t")
x=np.linspace(-4,4,200); ax[1].plot(x,stats.norm.pdf(x),"k-",lw=1.6,label="N(0,1)")
ax[1].axvline(1.96,c="r",ls=":"); ax[1].axvline(-1.96,c="r",ls=":")
ax[1].set(xlabel="Welch t, event days vs non-event days",ylabel="density",
          title=f"事件日報酬檢定：mean t {esd.t.mean():+.3f}, sd {esd.t.std():.2f}\n|t|>1.96: {(esd.t.abs()>1.96).sum()} vs {0.05*len(esd):.0f} expected (KS p={stats.kstest(esd.t,'norm').pvalue:.2f})")
ax[1].legend(fontsize=8)
# 3 put-OI: per-feature IC by era
di=pd.read_csv(f"{OUT}/put_oi_diagnostics.csv"); di=di[di.panel=="monthly"]
cols=[c for c in di.columns if c.startswith("IC_") and c!="IC_sign_agree"]
xs=np.arange(len(di))
for i,c in enumerate(cols):
    ax[2].bar(xs+(i-1.5)*0.2,di[c],width=0.2,label=c.replace("IC_","").replace("_","-"))
ax[2].axhline(0,c="k",lw=.8)
ax[2].set_xticks(xs); ax[2].set_xticklabels(di.feature,rotation=60,ha="right",fontsize=7)
ax[2].set(ylabel="Spearman IC vs c2c return (t+2)",
          title="賣權未平倉：每個特徵的 IC 依年代\n11 features × 144 cells × 2 panels → 0/1584 通過四道門檻")
ax[2].legend(fontsize=7,ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/qnt99_summary.png",dpi=115); print("wrote qnt99_summary.png")

# second figure: the gate leak
R=pd.read_csv(f"{OUT}/gate_leak_random_masks.csv")
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
m=R.groupby("frac")[["SR_of_SR","beta","passes"]].mean()
ax[0].plot(m.index,m.beta,"o-",label="beta vs buy&hold")
ax[0].plot(m.index,m.SR_of_SR,"s-",label="SR_of_SR")
ax[0].axhline(0.15,c="r",ls=":",label="|beta| gate = 0.15")
ax[0].axhline(0.6,c="g",ls=":",label="SR_of_SR gate = 0.6")
ax[0].set(xscale="log",xlabel="fraction of nights held long (random, no information)",
          title="門檻漏洞：beta 隨曝險縮小，SR_of_SR 不會")
ax[0].legend(fontsize=8)
ax[1].bar(["random long-night\nmask (5-10% expo)","QNT-99 event\ncalendar sweep"],
          [R[R.frac<=0.10].passes.mean()*100, 100*70/len(cs)],color=["#8172B2","#C44E52"])
ax[1].set(ylabel="% of cells passing all 4 house gates",
          title="70/800 = 8.8% 的日曆格通過門檻\n純隨機遮罩也是 ~12%")
plt.tight_layout(); plt.savefig(f"{OUT}/qnt99_gate_leak.png",dpi=115); print("wrote qnt99_gate_leak.png")
