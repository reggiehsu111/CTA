"""
Options factor-zoo re-derivation on CORRECTED full-strike data.

Context
-------
`tw_options_daily` was migrated 2026-08-24 from the ~20-strike 簡表
(`optDailyMarketSummaryExcel`) to the full `optDataDown` file. The 簡表
captured a fixed ~950-index-point window of strikes, recentred daily, so as
TAIEX ran 17k -> 45k its share of true open interest fell from ~45% (2022) to
7.5% (2026), and the day-to-day change in captured OI increasingly just
tracked spot: corr(dlog OI, return) was -0.264 in 2022-2026 on the old data
and is +0.014 on the corrected data.

Both live options signals had their sign, window AND variant fitted on that
contaminated series, so this treats their parameterization as unvalidated and
re-searches the space from scratch.

Two deliberate deviations from the exemplar zoo notebook, both documented in
the report:
  1. Returns are ROLL-ADJUSTED (same expiry at both ends of each window).
     Raw `close.pct_change()` books the calendar spread as P&L on the 305
     roll days in this history — mean 57bps, max 352bps.
  2. Both the project-standard cost model and a heavier realistic one are
     reported, because the standard one omits 期交稅 (see COSTS below).

Usage:  python options_rederive.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cta
import cta.backtest_live as bl
from cta.signals._operators import (
    bd_selftanh, rank_c, robust_z, selftanh, selfz, selfz_winsor, sign_thresh,
)

# ── Phase 1: setup ────────────────────────────────────────────────────────
IS_START, IS_END   = "2002-04-02", "2016-12-31"
OOS_START, OOS_END = "2017-01-01", "2026-08-24"

ASSET = cta.load_asset("mtx", "1d")
PERIODS = 252

# Roll-adjusted close-to-close return of the held contract.
_panel  = bl.load_price_panel()
_prices = bl.contract_prices(_panel, roll_lead_days=0)
_RET    = bl.scheme_returns(_prices, "close").reindex(ASSET.index)

_CLOSE = ASSET["close"].astype(float)
# Project-standard cost: 20 TWD/side + 2e-5. The 2e-5 is documented as
# slippage, which means 期交稅 (also 2e-5/side) is never charged.
_COST_STD  = 20.0 / (_CLOSE * 50.0) + 0.00002
# Realistic: commission 20 + tax 2e-5 + ~1 index point of slippage.
_COST_REAL = 20.0 / (_CLOSE * 50.0) + 0.00002 + 1.0 / _CLOSE


# ── Phase 2: the zoo ──────────────────────────────────────────────────────
def build_raw() -> dict[str, pd.Series]:
    raw: dict[str, pd.Series] = {}
    for metric in ("oi", "volume"):
        for side in ("put", "call", None):
            for ef in ("all", "monthly", "weekly", "front"):
                tag = f"{metric}_{side or 'both'}_{ef}"
                try:
                    s = cta.load_option_daily_total(metric, side, ef).astype(float)
                except Exception as e:                       # noqa: BLE001
                    print(f"  skip {tag}: {type(e).__name__} {e}")
                    continue
                s = s.replace([np.inf, -np.inf], np.nan)
                if s.notna().sum() < 300:
                    continue
                raw[tag] = s
    # Put/call ratios — a scale-free view that cancels the overall OI level
    for metric in ("oi", "volume"):
        for ef in ("all", "monthly", "weekly", "front"):
            p, c = raw.get(f"{metric}_put_{ef}"), raw.get(f"{metric}_call_{ef}")
            if p is None or c is None:
                continue
            raw[f"{metric}_pcr_{ef}"] = (p / c.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan)
    return raw


def build_zoo(raw: dict[str, pd.Series]) -> dict[str, pd.Series]:
    z: dict[str, pd.Series] = {}
    for tag, s in raw.items():
        d = s.diff()
        for w in (20, 60, 120):
            z[f"{tag}_selftanh{w}"] = bd_selftanh(s, w)
        for w in (20, 60):
            z[f"{tag}_winsor{w}"]   = selfz_winsor(s, w)
            z[f"{tag}_robust{w}"]   = selftanh(robust_z(s, w))
            z[f"{tag}_signth{w}"]   = sign_thresh(s, w)
            z[f"{tag}_chg_selftanh{w}"] = bd_selftanh(d, w)
        for w in (60, 120):
            z[f"{tag}_rank{w}"]     = rank_c(s, w)
        for w in (20, 60):
            z[f"{tag}_chgskew{w}"]  = selftanh(cta.InstSkew(w, d))
        # log-level deviation: robust to the multi-decade growth in OI
        ls = np.log(s.where(s > 0))
        for w in (60, 120):
            z[f"{tag}_logdev{w}"]   = bd_selftanh(ls, w)
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in z.items()}


# ── Phase 3/4: evaluation ─────────────────────────────────────────────────
def _stats(sig: pd.Series, start, end, sign=None, cost=None):
    cost = _COST_STD if cost is None else cost
    s = sig.reindex(ASSET.index).astype(float)
    ex = s.shift(2)
    gross = (ex * _RET)
    turn  = ex.fillna(0).diff().abs()
    tc    = turn * cost
    g = gross.loc[start:end].dropna()
    if len(g) < 200 or not np.isfinite(g.std()) or g.std() == 0:
        return None
    if sign is None:
        sign = -1 if (g.mean() < 0) else 1
    gs = g * sign
    ns = gs - tc.reindex(g.index).fillna(0)
    return {
        "sign": int(sign),
        "SR_gross": float(np.sqrt(PERIODS) * gs.mean() / gs.std()),
        "SR_net":   float(np.sqrt(PERIODS) * ns.mean() / ns.std()) if ns.std() else np.nan,
        "held_pct": float((s.loc[start:end].abs() > 0.01).mean()) * 100,
        "n_bars":   int(len(g)),
    }


def main() -> None:
    print("building raw series ...")
    raw = build_raw()
    print(f"  {len(raw)} raw series")
    zoo = build_zoo(raw)
    print(f"  {len(zoo)} signals in the zoo")

    print("in-sample sweep ...")
    rows = []
    for nm, sg in zoo.items():
        st = _stats(sg, IS_START, IS_END)
        if st:
            rows.append({"signal": nm, **st})
    tbl = pd.DataFrame(rows).set_index("signal")
    print(f"  {len(tbl)} evaluable in IS")

    print("out-of-sample (frozen IS sign) ...")
    oos = []
    for nm, r in tbl.iterrows():
        st = _stats(zoo[nm], OOS_START, OOS_END, sign=int(r["sign"]))
        if st:
            stx = _stats(zoo[nm], OOS_START, OOS_END, sign=int(r["sign"]),
                         cost=_COST_REAL)
            oos.append({"signal": nm,
                        "SR_net_oos":   st["SR_net"],
                        "SR_gross_oos": st["SR_gross"],
                        "SR_net_oos_realcost": stx["SR_net"] if stx else np.nan,
                        "n_bars_oos":   st["n_bars"]})
    comb = tbl.join(pd.DataFrame(oos).set_index("signal"), how="inner")

    OUT = Path(__file__).resolve().parent
    comb.sort_values("SR_net_oos", ascending=False).to_csv(
        OUT / "options_rederive_scoreboard.csv", encoding="utf-8-sig")

    surv = comb[(comb["SR_net"] >= 0.5) & (comb["SR_net_oos"] > 0.0)
                & (comb["held_pct"] >= 10.0) & (comb["n_bars"] >= 200)
                & (comb["n_bars_oos"] >= 200)].sort_values(
        "SR_net_oos", ascending=False)
    surv.to_csv(OUT / "options_rederive_survivors.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(f"\n=== zoo {len(comb)} evaluated | survivors {len(surv)} "
          f"({100*len(surv)/max(len(comb),1):.1f}%) ===")
    cols = ["sign", "SR_net", "SR_net_oos", "SR_net_oos_realcost", "held_pct", "n_bars_oos"]
    print("\n--- top 15 by OOS net SR ---")
    print(comb.sort_values("SR_net_oos", ascending=False)[cols].head(15).round(2).to_string())
    print("\n--- survivors ---")
    print(surv[cols].round(2).to_string() if len(surv) else "  NONE")

    # Where do the two incumbents land?
    print("\n--- incumbent parameterizations, re-measured ---")
    inc = {
        "opt_put_mo_oi_selftanh_w60":  "oi_put_monthly_selftanh60",
        "opt_call_all_oi_signth_w20":  "oi_call_all_signth20",
    }
    for live, key in inc.items():
        if key in comb.index:
            r = comb.loc[key]
            pct = 100.0 * (comb["SR_net_oos"] < r["SR_net_oos"]).mean()
            print(f"  {live:32s} -> {key}")
            print(f"      IS {r['SR_net']:+.2f} | OOS {r['SR_net_oos']:+.2f} "
                  f"| OOS@realcost {r['SR_net_oos_realcost']:+.2f} "
                  f"| pctile in zoo {pct:.0f}")
        else:
            print(f"  {live:32s} -> {key}: NOT EVALUABLE")


if __name__ == "__main__":
    main()
