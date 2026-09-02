"""QNT-18 part 3e: resolve the tension between the two mechanism tests.

Part 3b found that WITHIN every grid, `day` helps net-short positions and hurts
net-long ones (pooled corr(ΔSR, tilt) = -0.345), and the three grids differ
sharply in tilt composition (daily macro 61% net-short, slow macro 58% net-long,
registered 73% net-long). That would explain the whole cross-grid reversal.
But part 3c found the daily-macro gain SURVIVES de-tilting (+0.060).

Both can't be the full story. This removes the tilt from ALL THREE grids at the
matched lag (day@2 vs c2c@2) and asks whether they then agree.

Demean = subtract the full-sample mean position (the strictest possible removal;
it is look-ahead, and is a diagnostic only, never a tradeable construction).
"""
import sys, warnings, io
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd

OUT = "/home/ubuntu/mtx/signal_zoo/macro_windows"
SWEEP = f"{OUT}/macro_window_sweep.py"
_src = io.open(SWEEP, encoding="utf-8").read()
exec(compile(_src.split("# ── Build the signals once")[0], SWEEP, "exec"))   # noqa: S102
import cta
from cta.signals import SIGNAL_REGISTRY
pd.set_option("display.width", 200)

cat = cta.macro_catalog()
SLOW = [("us_semi_ip","yoy"),("us_semi_ip_nsa","yoy"),("us_semi_ppi","yoy"),
        ("us_electronics_ppi","yoy"),("kr_exports_sa","yoy"),("kr_exports","yoy"),
        ("kr_kospi","yoy"),("cn_leading_idx","level"),("cn_exports","yoy"),
        ("cn_shanghai_comp","yoy"),("us_cfnai","level"),("us_mfg_new_orders","yoy"),
        ("us_freight_tsi","yoy"),("us_retail_inv_sales","level"),
        ("us_recession_prob","level"),("us_empire_state","level"),
        ("us_philly_fed","level"),("us_stlfsi","level"),("epu_global","level"),
        ("copper","yoy"),("igrea","level")]

def macro_grid(specs, tag):
    out = {}
    for sid, kind in specs:
        x = (ctx.macro_yoy(sid, {"M":12,"Q":4,"W":52}.get(cat.loc[sid,"freq"],12))
             if kind == "yoy" else ctx.macro(sid)).astype(float)
        for tn, tf in TRANSFORMS.items():
            for w in WINDOWS:
                s = cta.normalize_signal(tf(x, w).replace([np.inf,-np.inf], np.nan),
                                         method="tanh", window=252)
                if s.dropna().empty or s.loc[:"2018-12-31"].dropna().shape[0] < 500:
                    continue
                out[f"{tag}:{sid}_{kind}|{tn}|w{w}"] = (s, None)
    return out

G = {}
G["daily macro"] = macro_grid([(s, "level") for s in DAILY], "d")
G["slow macro"]  = macro_grid(SLOW, "s")
reg = {}
for name, obj in sorted(SIGNAL_REGISTRY.items()):
    cls = type(obj)
    raw = obj.compute_raw(ctx).astype(float)
    s = raw if cls.pre_normalized else cta.normalize_signal(raw, method="tanh", window=252)
    reg[name] = ((s * cls.sign).replace([np.inf,-np.inf], np.nan), 1)   # sign already frozen
G["registered"] = reg
print({k: len(v) for k, v in G.items()})

LAG = 2
rows = []
for g, cells in G.items():
    for cand, (s, fixed_sign) in cells.items():
        ref = wstats(s, "c2c", end="2018-12-31")
        if ref is None:
            continue
        sgn = fixed_sign if fixed_sign is not None else ref["sign"]
        for how in ("raw", "detilt"):
            d = s if how == "raw" else s - s.mean()
            for v in ("c2c", "day"):
                keep, _SHIFT[v] = _SHIFT[v], LAG
                try:
                    st = wstats(d, v, sign=sgn)
                finally:
                    _SHIFT[v] = keep
                if st:
                    rows.append(dict(grid=g, cand=cand, how=how, variant=v,
                                     SR=st["SR_net"], srsr=st["SR_of_SR"],
                                     tilt=st["mean_exec_w"], beta=st["beta"]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/detilt_matched.csv", index=False)

print("\n=== day@2 − c2c@2, before and after removing the position's tilt ===")
print(f"{'grid':14s} {'position':10s} {'n':>4s} {'med|tilt|':>10s} {'medΔSR':>9s} {'meanΔSR':>9s} {'win':>7s}")
for g in G:
    for how, lab in (("raw", "raw"), ("detilt", "de-tilted")):
        p = df[(df.grid == g) & (df.how == how)].pivot_table(
            index="cand", columns="variant", values=["SR", "tilt"])
        x = (p["SR"]["day"] - p["SR"]["c2c"]).dropna()
        print(f"{g:14s} {lab:10s} {len(x):4d} {p['tilt']['c2c'].abs().median():10.4f} "
              f"{x.median():+9.4f} {x.mean():+9.4f} {(x>0).mean():7.1%}")

print("\n=== within-grid: de-tilted ΔSR split by the ORIGINAL tilt sign ===")
for g in G:
    p = df[(df.grid == g) & (df.how == "detilt")].pivot_table(index="cand", columns="variant", values="SR")
    t = df[(df.grid == g) & (df.how == "raw") & (df.variant == "c2c")].set_index("cand")["tilt"]
    x = (p["day"] - p["c2c"]).dropna(); t = t.reindex(x.index)
    print(f"  {g:14s} long-tilt n={int((t>0).sum()):4d} med {x[t>0].median():+.3f} | "
          f"short-tilt n={int((t<0).sum()):4d} med {x[t<0].median():+.3f}")
