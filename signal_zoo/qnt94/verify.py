"""QNT-94 validity check: does the rebuilt PnL reproduce the on-disk SR grids?"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0,"/home/ubuntu/mtx")
import numpy as np, pandas as pd
exec(open("/home/ubuntu/mtx/signal_zoo/qnt94/series_neff.py").read().split("BAR = ")[0])

Z = "/home/ubuntu/mtx/signal_zoo"
# --- GRID 2 c2c/day: rebuild per-cell SR and compare to qnt19_postfloor ---
ref = pd.read_csv(f"{Z}/qnt19_postfloor/window_sweep_full.csv")
ref = ref[ref.regime=="full"].set_index(["cand","variant"])["SR_net"]
DAILY = ["us_dxy_broad","us_real_10y","us_breakeven_10y","us_breakeven_5y5y","us_dgs5",
         "us_dgs30","us_term_premium_10y","twd_usd","krw_usd","cny_usd","wti"]
rows=[]
for v in ("c2c","day","o2o"):
    g = build_grid(DAILY, v, "G2")
    for sid,d in g.items():
        for cell in d.columns:
            k=(f"{sid}|{cell}",v)
            if k in ref.index: rows.append(dict(cand=k[0],variant=v,mine=sr(d[cell]),ref=float(ref.loc[k])))
V=pd.DataFrame(rows); V["diff"]=V.mine-V.ref
print(f"GRID 2: {len(V)} cells matched   corr(mine,ref) = {V.mine.corr(V.ref):.5f}")
print(f"  |diff|: mean {V['diff'].abs().mean():.4f}  max {V['diff'].abs().max():.4f}")
print(V.groupby("variant")["diff"].agg(["mean","std","max"]).round(4).to_string())

# --- the statistic QNT-14 quoted: per-series median dSR(day - c2c) ---
p = pd.read_csv(f"{Z}/qnt19_postfloor/window_sweep_full.csv")
p = p[p.regime=="full"].pivot_table(index=["series","transform","window"],columns="variant",values="SR_net")
d = (p["day"]-p["c2c"]).groupby("series").median()
print(f"\nQNT-14 statistic, post-floor: per-series median dSR(day-c2c) = {d.median():+.4f}  (n=11 series)")
pre = pd.read_csv(f"{Z}/macro_windows/window_sweep_full.csv")
pre = pre[pre.regime=="full"].pivot_table(index=["series","transform","window"],columns="variant",values="SR_net")
dpre = (pre["day"]-pre["c2c"]).groupby("series").median()
print(f"QNT-14 statistic, pre-floor : per-series median dSR(day-c2c) = {dpre.median():+.4f}")
V.round(4).to_csv(f"{Z}/qnt94/qnt94_verify_grid2.csv",index=False)
