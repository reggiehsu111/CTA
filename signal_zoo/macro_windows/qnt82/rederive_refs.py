"""QNT-82: re-derive the frozen reference constants qnt32_verify.py checks against.

Reads only. Prints, for each grid, the statistic name and the value the CSVs on
disk produce right now, in the same order qnt32_verify.py checks them — so the
refresh is a copy of this output, not a hand-typed number, and can be repeated
the next time the sweeps move.

    python3 signal_zoo/macro_windows/qnt82/rederive_refs.py
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/QuantResearch/Libs")
sys.path.insert(0, "/home/ubuntu/mtx")
import numpy as np, pandas as pd
import cta

Z = "/home/ubuntu/mtx/signal_zoo"


def stamp(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def show(name, val, dec=4):
    print(f'    check("{name}", s["?"], {val:+.{dec}f})')


# ── Grid 1 ────────────────────────────────────────────────────────────────
# QNT-92: grid 1 reads the POST-FLOOR copy. `macro_sweep/full_sweep.csv`
# stays pre-floor as the "before" side of qnt19_postfloor/compare_prefloor.py.
p = f"{Z}/qnt19_postfloor/full_sweep.csv"
print(f"GRID 1  {p}  mtime {stamp(p)}")
g1 = pd.read_csv(p)
s = cta.sweep_headline(g1, "SR_full", label="g1").stats
for k in ("icc", "n_eff", "per_series_median", "n_positive", "sd_cells", "se_sr"):
    print(f"  SR_full     {k:20s} {s[k]:+.4f}")
s2 = cta.sweep_headline(g1, "SR_of_SR", label="g1srsr").stats
for k in ("icc", "n_eff", "sd_cells", "best_cell", "expected_max_null"):
    print(f"  SR_of_SR    {k:20s} {s2[k]:+.4f}")

# ── Grid 2 ────────────────────────────────────────────────────────────────
p = f"{Z}/macro_windows/window_sweep_full.csv"
print(f"\nGRID 2  {p}  mtime {stamp(p)}")
g2 = pd.read_csv(p)
g2 = g2[g2.regime == "full"]
p2 = g2.pivot_table(index=["series", "transform", "window"],
                    columns="variant", values="SR_net").reset_index()
s = cta.paired_headline(p2["day"], p2["c2c"], series=p2["series"],
                        n_years=g2.n_years.median()).stats
for k in ("icc", "n_eff", "cell_median", "per_series_median", "n_positive",
          "wilcoxon_p", "rho", "se_delta"):
    print(f"  day-c2c     {k:20s} {s[k]:+.4f}")

# ── Grid 3 ────────────────────────────────────────────────────────────────
p = f"{Z}/macro_windows/slow_window_sweep.csv"
print(f"\nGRID 3  {p}  mtime {stamp(p)}")
g3 = pd.read_csv(p)
p3 = g3.pivot_table(index=["series", "transform", "window"],
                    columns="variant", values="SR_net").reset_index()
s = cta.paired_headline(p3["day"], p3["c2c"], series=p3["series"],
                        n_years=g3.n_years.median()).stats
for k in ("icc", "n_eff", "cell_median", "per_series_median", "wilcoxon_p"):
    print(f"  day@2-c2c@2 {k:20s} {s[k]:+.4f}")

# ── Grid 4 ────────────────────────────────────────────────────────────────
p = f"{Z}/macro_windows/registered_window_sweep.csv"
print(f"\nGRID 4  {p}  mtime {stamp(p)}")
g4 = pd.read_csv(p)
g4 = g4[g4.regime == "full"]
p4 = g4.pivot_table(index="signal", columns="variant",
                    values="SR_net").dropna(subset=["c2c", "day"])
s = cta.paired_headline(p4["day"], p4["c2c"], n_years=g4.n_years.median()).stats
for k in ("per_series_median", "n_positive", "wilcoxon_p", "n_eff"):
    print(f"  day-c2c     {k:20s} {s[k]:+.4f}")
s = cta.sweep_headline(p4.reset_index(), "c2c", series_col="signal",
                       n_years=g4.n_years.median(), label="g4").stats
for k in ("sd_cells", "se_sr", "ratio"):
    print(f"  c2c leg     {k:20s} {s[k]:+.4f}")
