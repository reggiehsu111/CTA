import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/home/ubuntu/mtx/signal_zoo/macro_combo"
yr=pd.read_csv(f"{OUT}/yearly_sr.csv",index_col=0)
fam=pd.read_csv(f"{OUT}/combo_family_sweep.csv")
sc=pd.read_csv(f"{OUT}/combo_scoreboard.csv")
rng=np.random.default_rng(0)
fig,ax=plt.subplots(2,2,figsize=(15,9.5))

# 1. bootstrap distribution of SR_of_SR
for c in yr.columns:
    s=yr[c].dropna().values
    v=(lambda b: b.mean(1)/b.std(1,ddof=1))(rng.choice(s,size=(20000,len(s)),replace=True))
    ax[0,0].hist(v,bins=90,histtype="step",lw=1.6 if c.startswith("EW") else 1,label=f"{c} ({s.mean()/s.std(ddof=1):.3f})")
ax[0,0].axvline(0.6,color="r",ls="--",lw=1.4); ax[0,0].text(0.61,ax[0,0].get_ylim()[1]*.85,"gate 0.6",color="r",fontsize=8)
ax[0,0].set_title("SR_of_SR 抽樣分布（逐年 bootstrap, n≈25）"); ax[0,0].legend(fontsize=7); ax[0,0].set_xlabel("SR_of_SR")

# 2. family sweep
o=fam.sort_values("SR_of_SR")
ax[0,1].barh(range(len(o)),o.SR_of_SR,color="#69c")
ax[0,1].axvline(0.6,color="r",ls="--",lw=1.4)
ax[0,1].axvline(0.494,color="k",ls=":",lw=1.2)
ax[0,1].text(0.497,0.5,"best single leg 0.494",rotation=90,fontsize=7)
ax[0,1].set_yticks(range(len(o))); ax[0,1].set_yticklabels(o.cell,fontsize=7)
ax[0,1].set_title("EW3 SR_of_SR：整個 transform×window 家族 (0/18 過關)"); ax[0,1].grid(alpha=.3,axis="x")

# 3. IS vs OOS scatter for the family
ax[1,0].scatter(fam.SR_IS,fam.SR_OOS,c=fam.window,cmap="viridis",s=55)
lim=[0,0.85]; ax[1,0].plot(lim,lim,"k--",lw=.9); ax[1,0].set_xlim(lim); ax[1,0].set_ylim(lim)
for _,r in fam.iterrows(): ax[1,0].annotate(r.cell.replace("EW3|",""),(r.SR_IS,r.SR_OOS),fontsize=6,alpha=.8)
ax[1,0].set_xlabel("SR_IS (<=2018)"); ax[1,0].set_ylabel("SR_OOS (2019+)")
ax[1,0].set_title("EW3 IS vs OOS — 9/18 在對角線上方（igrea 單腳為 18/18）"); ax[1,0].grid(alpha=.3)

# 4. gate summary
r=sc.set_index("combo").loc[["igrea","epu_global","kr_kospi","EW(igrea+epu_global+kr_kospi)"]]
x=np.arange(len(r)); ax[1,1].bar(x-0.2,r.SR_full,0.4,label="SR_net (full)",color="#888")
ax[1,1].bar(x+0.2,r.SR_of_SR,0.4,label="SR_of_SR",color="#2b7")
ax[1,1].axhline(0.6,color="r",ls="--",lw=1.2,label="SR_of_SR gate")
ax[1,1].axhline(0.703,color="b",ls=":",lw=1.2,label="買進持有 SR 0.703")
ax[1,1].set_xticks(x); ax[1,1].set_xticklabels([i.replace("EW(","EW3\n(") for i in r.index],fontsize=7)
ax[1,1].legend(fontsize=7); ax[1,1].grid(alpha=.3,axis="y"); ax[1,1].set_title("對照 house gate 與基準")
fig.suptitle("QNT-16 穩健性：分散化沒有把 SR_of_SR 推過 0.6",fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/combo_robustness.png",dpi=110)
print("wrote combo_robustness.png")
