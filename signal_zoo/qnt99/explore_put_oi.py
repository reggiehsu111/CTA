"""QNT-99 Part B step 1 — what IS the put-OI data? Questions before signals."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0,"/home/ubuntu/mtx")
import numpy as np, pandas as pd, cta
from db_utils import engine

OUT="/home/ubuntu/mtx/signal_zoo/qnt99"
asset = cta.load_asset("mtx","1d")
spot  = asset["close"]

# Q1 — coverage: strikes per day, total put OI, by year. Is the source complete?
q = """
select date, count(*) n_strike, sum(oi) oi, sum(volume) vol,
       count(*) filter (where oi>0) n_strike_oi
from tw_options_daily where side='put' and market_code=0 and expiry ~ '^[0-9]{6}$'
group by 1 order by 1"""
mo = pd.read_sql(q, engine, parse_dates=["date"]).set_index("date")
mo["yr"]=mo.index.year
print("=== Q1  put OI, MONTHLY expiries, regular session ===")
print(mo.groupby("yr").agg(days=("oi","size"), med_strikes=("n_strike","median"),
      med_oi=("oi","median"), med_vol=("vol","median")).to_string())

# same for ALL expiries (weeklies included)
q2 = q.replace("and expiry ~ '^[0-9]{6}$'","")
al = pd.read_sql(q2, engine, parse_dates=["date"]).set_index("date")
al["yr"]=al.index.year
print("\n=== Q1b  put OI, ALL expiries ===")
print(al.groupby("yr").agg(days=("oi","size"), med_strikes=("n_strike","median"),
      med_oi=("oi","median")).to_string())

# Q2 — is the level a market-size proxy? corr with spot level, and stationarity
for nm,s in [("monthly",mo["oi"]),("all",al["oi"])]:
    x=np.log(s.replace(0,np.nan)).dropna(); sp=np.log(spot).reindex(x.index).dropna()
    x=x.reindex(sp.index)
    print(f"\n=== Q2 {nm}: corr(log put OI, log spot) full={x.corr(sp):.3f}", end="  ")
    for a,b in [("2013","2018"),("2019","2026")]:
        m=(x.index.year>=int(a))&(x.index.year<=int(b))
        print(f"{a}-{b}={x[m].corr(sp[m]):.3f}", end="  ")
    print()

mo["oi"].to_frame("put_oi_monthly").join(al["oi"].rename("put_oi_all")).to_csv(f"{OUT}/put_oi_totals.csv")
print("\nwrote put_oi_totals.csv", mo.index.min().date(), mo.index.max().date())
