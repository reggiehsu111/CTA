"""QNT-14 follow-through: the two full-history `day` cells that clear all four
gates, examined against the §5 untrustworthy-result checklist. Report only -
no sign is chosen for live use, no variant is recommended."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context

OUT="/home/ubuntu/mtx/signal_zoo/macro_windows"
ctx=build_context(); A=ctx.asset; PV,PPY=50.0,252
_o,_c=A["open"].astype(float),A["close"].astype(float)
RET_DAY=_c/_o-1                      # 08:45->13:45, intra-contract, no roll exposure

CANDS={"us_dgs5|rankc|w252":("us_dgs5",ops.rank_c,252),
       "us_dgs5|bdtanh|w252":("us_dgs5",ops.bd_selftanh,252),
       "us_dgs5|selfz|w252":("us_dgs5",ops.selfz,252),
       "us_dgs30|rankc|w252":("us_dgs30",ops.rank_c,252),
       "us_real_10y|signth|w120":("us_real_10y",ops.sign_thresh,120)}
COSTS={"gross":(0.0,0.0),"stub 20+2e-5":(20.,2e-5),"real 70+4e-5":(70.,4e-5),
       "2x real 140+8e-5":(140.,8e-5),"3x real 210+1.2e-4":(210.,1.2e-4)}

def pnl(sig,sign,fixed,fee,lag=1):
    pos=sig.reindex(A.index).astype(float).shift(lag)*sign
    cost=fixed/(_o*PV)+fee
    return pos*RET_DAY - pos.fillna(0).diff().abs()*cost, pos

def sr(x): x=x.dropna(); return float(np.sqrt(PPY)*x.mean()/x.std())

SIG={}
for k,(sid,tf,w) in CANDS.items():
    s=tf(ctx.macro(sid).astype(float),w).replace([np.inf,-np.inf],np.nan)
    SIG[k]=cta.normalize_signal(s,method="tanh",window=252)

# sign frozen on IS <= 2018 exactly as the sweep did
SIGN={k:(1 if sr(pnl(s,1,70.,4e-5)[0].loc[:"2018-12-31"])>=0 else -1) for k,s in SIG.items()}
print("IS-frozen signs:",SIGN)

print("\n=== COST SENSITIVITY (day window, shift 1, full history) ===")
hdr=f"{'cand':26s}"+"".join(f"{c:>20s}" for c in COSTS); print(hdr)
for k,s in SIG.items():
    print(f"{k:26s}"+"".join(f"{sr(pnl(s,SIGN[k],*COSTS[c])[0]):>20.3f}" for c in COSTS))

print("\n=== POSITION SCALE / IMPLEMENTABILITY (discrete contracts) ===")
for k,s in SIG.items():
    _,pos=pnl(s,SIGN[k],70.,4e-5); p=pos.dropna()
    for book in (5,20):
        n=(p*book).round()
        print(f"  {k:26s} book={book:3d}  mean|w| {p.abs().mean():.3f}  max|w| {p.abs().max():.3f}  "
              f"flat days {float((n==0).mean()):.1%}  SR_cont {sr(pnl(s,SIGN[k],70.,4e-5)[0]):.3f}  "
              f"SR_disc {sr((n/book)*RET_DAY-(n/book).fillna(0).diff().abs()*(70./(_o*PV)+4e-5)):.3f}")

print("\n=== PER-YEAR SR (net, real cost) ===")
tab={}
for k,s in SIG.items():
    n=pnl(s,SIGN[k],70.,4e-5)[0].dropna()
    tab[k]=n.groupby(n.index.year).apply(lambda x: sr(x) if len(x)>20 and x.std()>0 else np.nan)
T=pd.DataFrame(tab); pd.set_option("display.width",200)
print(T.round(2).to_string())
print("\npositive-year rate:",(T>0).mean().round(3).to_dict())

print("\n=== PnL CORRELATION between the candidates (are these one bet or five?) ===")
P=pd.DataFrame({k:pnl(s,SIGN[k],70.,4e-5)[0] for k,s in SIG.items()}).dropna()
print(P.corr().round(3).to_string())

print("\n=== IS / OOS with the sign FROZEN ON IS ===")
for k,s in SIG.items():
    n=pnl(s,SIGN[k],70.,4e-5)[0]
    print(f"  {k:26s} IS(<=2018) {sr(n.loc[:'2018-12-31']):+.3f}  OOS(2019+) {sr(n.loc['2019-01-01':]):+.3f}  full {sr(n):+.3f}")

# ── plots ──────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,2,figsize=(15,9))
for k,s in SIG.items():
    ax[0,0].plot(pnl(s,SIGN[k],70.,4e-5)[0].fillna(0).cumsum(),label=k,lw=1.1)
ax[0,0].plot((RET_DAY).fillna(0).cumsum(),"k--",lw=1,label="day-window buy&hold")
ax[0,0].set_title("累積損益 day 窗口 (net, 70 TWD/side + 4e-5)");ax[0,0].legend(fontsize=7);ax[0,0].grid(alpha=.3)
for k in SIG:
    ax[0,1].plot(list(COSTS),[sr(pnl(SIG[k],SIGN[k],*COSTS[c])[0]) for c in COSTS],marker="o",label=k)
ax[0,1].axhline(0,color="k",lw=.8);ax[0,1].set_title("成本敏感度 (SR vs cost model)");ax[0,1].tick_params(axis='x',rotation=20);ax[0,1].legend(fontsize=7);ax[0,1].grid(alpha=.3)
T.plot(ax=ax[1,0],marker="o",lw=1);ax[1,0].axhline(0,color="k",lw=.8);ax[1,0].set_title("逐年 SR (net)");ax[1,0].legend(fontsize=7);ax[1,0].grid(alpha=.3)
d=pd.read_csv(f"{OUT}/window_sweep_full.csv").dropna(subset=["SR_net"])
f=d[d.regime=="full"]
for v in ["c2c","o2o","day"]:
    ax[1,1].hist(f[f.variant==v].SR_net,bins=30,alpha=.5,label=f"{v} (median {f[f.variant==v].SR_net.median():.3f})")
ax[1,1].axvline(0,color="k",lw=.8);ax[1,1].set_title("全網格 SR 分布 198 cells/variant (25y, net)");ax[1,1].legend(fontsize=8);ax[1,1].grid(alpha=.3)
plt.tight_layout();plt.savefig(f"{OUT}/day_window_evidence.png",dpi=110)
print(f"\nwrote {OUT}/day_window_evidence.png")
