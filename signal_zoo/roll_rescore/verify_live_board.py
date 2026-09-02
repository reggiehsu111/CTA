"""Read-only: does the stored live-board pnl_1d now match the roll-adjusted legs?"""
import sys, warnings
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs"); sys.path.insert(0, "/home/ubuntu/mtx")
warnings.filterwarnings("ignore")
import pandas as pd, cta
from db_utils import engine
from cta.signals import _base as B
A = cta.load_asset("mtx","1d"); df = A.df if hasattr(A,"df") else A
roll = A.is_rollover
legs = {"c2c":B._c2c_ret(df), "o2o":B._o2o_ret(df), "noonpause":B._noonpause_ret(df),
        "day":B._day_ret(df), "ongap":B._ongap_ret(df), "night":B._night_ret(df)}
legs = {k:pd.Series(v).reindex(A.index) for k,v in legs.items()}
raw  = A["close"].pct_change()
q = """select v.date, v.signal_name, v.variant, v.position, v.pnl_1d
       from mtx_signal_values v where v.variant in ('c2c','o2o','noonpause')
         and v.position <> 0 order by v.date"""
V = pd.read_sql(q, engine, parse_dates=["date"])
print(f"rows read {len(V)}, {V.date.min().date()}..{V.date.max().date()}, "
      f"{V.signal_name.nunique()} signals")
V["is_roll"] = V.date.map(roll).fillna(False)
R = V[V.is_roll]
for var, g in R.groupby("variant"):
    fit_new = (g.pnl_1d - g.position*g.date.map(legs[var])).abs()
    fit_old = (g.pnl_1d - g.position*g.date.map(raw if var=="c2c" else legs[var])).abs()
    print(f"{var:10s} n_rollrows={len(g):5d}  residual vs NEW leg: max={fit_new.max():.3e} "
          f"mean={fit_new.mean():.3e}   (cost leg included in pnl, so nonzero = turnover cost)")
