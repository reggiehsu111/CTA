"""QNT-16 addendum: is the EW3 SR_of_SR a real effect or one lucky cell?

Sweeps the ALREADY-DECLARED parameters (transform x window) applied uniformly to
all three legs, rebuilds the equal-weight sleeve for each, and reports the FULL
grid. Signs are refitted on IS only per cell and frozen — never on full sample.
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd, cta
from cta.signals import _operators as ops
from cta.signals._ctx import build_context

OUT="/home/ubuntu/mtx/signal_zoo/macro_combo"; IS_END,OOS="2018-12-31","2019-01-01"
REAL=dict(fixed_per_side=70.0, fee_rate=4e-5)
ctx=build_context(); ASSET=ctx.asset; cat=cta.macro_catalog()
SERIES=[("igrea","level"),("epu_global","level"),("kr_kospi","yoy")]
TF={"selfz":ops.selfz,"robustz":ops.robust_z,"bdtanh":ops.bd_selftanh,
    "rankc":ops.rank_c,"signth":ops.sign_thresh,"dev":ops.dev}
raw={}
for sid,kind in SERIES:
    raw[sid]= ctx.macro_yoy(sid,{"M":12,"Q":4,"D":252,"W":52}.get(cat.loc[sid,"freq"],12)) \
              if kind=="yoy" else ctx.macro(sid)

rows=[]
for tn,tf in TF.items():
    for w in (60,120,252):
        legs=[]
        for sid,_ in SERIES:
            s=tf(raw[sid].astype(float),w).replace([np.inf,-np.inf],np.nan)
            s=cta.normalize_signal(s,method="tanh",window=252)
            sg=int(cta.signal_stats(s,ASSET,end=IS_END,auto_flip=True,roll_adjusted=True,**REAL)["sign"])
            legs.append(s*sg)
        ew=pd.concat(legs,axis=1).mean(axis=1,skipna=False)
        f=cta.signal_stats(ew,ASSET,auto_flip=False,roll_adjusted=True,**REAL)
        i=cta.signal_stats(ew,ASSET,end=IS_END,auto_flip=False,roll_adjusted=True,**REAL)
        o=cta.signal_stats(ew,ASSET,start=OOS,auto_flip=False,roll_adjusted=True,**REAL)
        rows.append(dict(cell=f"EW3|{tn}|w{w}",transform=tn,window=w,
            SR_IS=i["SR_net"],SR_OOS=o["SR_net"],SR_full=f["SR_net"],SR_of_SR=f["SR_of_SR"],
            SR_of_SR_IS=i["SR_of_SR"],SR_of_SR_OOS=o["SR_of_SR"],
            positive_years=f["positive_years"],beta=f["beta"],n_years=f["n_years"],
            max_dd_days=f["max_dd_days"],turnover_ann=f["turnover_ann"]))
d=pd.DataFrame(rows); d["OOS_minus_IS"]=d.SR_OOS-d.SR_IS
d["gate_srsr"]=d.SR_of_SR>0.6; d["gate_posyr"]=d.positive_years>=0.65
d["gate_beta"]=d.beta.abs()<0.15; d["gate_nyr"]=d.n_years>=5
d["n_gates"]=d[["gate_srsr","gate_posyr","gate_beta","gate_nyr"]].sum(axis=1)
d.to_csv(f"{OUT}/combo_family_sweep.csv",index=False)
pd.set_option("display.width",260,"display.max_columns",40)
print("=== EW3 across the transform x window family (full grid, nothing dropped) ===")
print(d.round(3).drop(columns=["transform","window"]).to_string(index=False))
print(f"\nSR_of_SR: min {d.SR_of_SR.min():.3f} median {d.SR_of_SR.median():.3f} max {d.SR_of_SR.max():.3f}")
print(f"cells passing SR_of_SR>0.6: {int(d.gate_srsr.sum())}/{len(d)}")
print(f"cells passing ALL FOUR gates: {int((d.n_gates==4).sum())}/{len(d)}")
print(f"cells with OOS>IS: {int((d.OOS_minus_IS>0).sum())}/{len(d)}")
