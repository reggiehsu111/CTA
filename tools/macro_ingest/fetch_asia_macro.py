"""Fetchers + parsers for the Taiwan / Japan macro ingest (QNT-10).

Pure functions: every one returns a month-indexed (or date-indexed) DataFrame.
Nothing here touches the database — see ingest_asia_macro.py for that.
"""
from __future__ import annotations

import csv
import io
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "/home/ubuntu/mtx/tools/macro_ingest")
import asia_macro_sources as S


# ── helpers ─────────────────────────────────────────────────────────────────
def _period_to_ts(p: str) -> pd.Timestamp | None:
    """'1981M01' / '198201' / '2026-06' -> month-start Timestamp. Annual -> None.

    Annual rows must map to None, not to January: the DGBAS files interleave
    annual totals ('1978') with monthly rows, and silently treating an annual
    figure as January would overwrite a real month with a 12-month average.
    """
    p = str(p).strip()
    if m := re.fullmatch(r"(\d{4})M(\d{2})", p):
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    if m := re.fullmatch(r"(\d{4})-(\d{2})", p):
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    if m := re.fullmatch(r"(\d{4})(\d{2})", p):
        mo = int(m.group(2))
        return pd.Timestamp(int(m.group(1)), mo, 1) if 1 <= mo <= 12 else None
    return None


def _num(v) -> float:
    """Parse a value, mapping every 'missing' marker these feeds use to NaN.

    The feeds use '-', '', '…', '---' and '‧' for missing.  Returning 0.0 for
    any of them would be the classic confident-zero failure, so they all become
    NaN and propagate.
    """
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", "").replace("　", "")
    if s in ("", "-", "--", "---", "…", "‧", "．", "n.a.", "N/A"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _get(url: str, *, verify=True, ua="Mozilla/5.0 (X11; Linux x86_64)",
         tries=4, timeout=(15, 120)) -> bytes:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": ua}, timeout=timeout, verify=verify)
            if r.status_code == 200 and r.content:
                return r.content
            last = f"HTTP {r.status_code} bytes={len(r.content)}"
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(3 * (2 ** k))
    raise RuntimeError(f"fetch failed {url}: {last}")


def _decode(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


# ── Taiwan ──────────────────────────────────────────────────────────────────
def fetch_tw_cpi() -> pd.DataFrame:
    """總指數 + the 7 top-level COICOP groups, index level and YoY.

    NOTE: this DGBAS table (基本分類) carries no core CPI (不含蔬果及能源);
    that lives in a separate DGBAS table and is NOT ingested here.
    """
    raw = _decode(_get(S.TW_SOURCES["cpi"]["url"], verify=S.TW_CA_BUNDLE))
    obs = re.findall(
        r"<Obs><Item>(.*?)</Item><TIME_PERIOD>(.*?)</TIME_PERIOD>"
        r"<FREQ>(.*?)</FREQ><TYPE>(.*?)</TYPE>\s*<Item_VALUE>(.*?)</Item_VALUE></Obs>",
        raw, re.S)
    want = {
        "總指數": "cpi",
        "一.食物類": "cpi_food", "二.衣著類": "cpi_apparel",
        "三.居住類": "cpi_housing", "四.交通及通訊類": "cpi_transport",
        "五.醫藥保健類": "cpi_health", "六.教養娛樂類": "cpi_education",
        "七.雜項類": "cpi_misc",
    }
    rows = {}
    for item, period, freq, typ, val in obs:
        if freq != "M":
            continue
        base = item.split("(")[0].strip()
        col = want.get(base)
        if col is None:
            continue
        ts = _period_to_ts(period)
        if ts is None:
            continue
        suffix = "" if typ == "原始值" else "_yoy"
        rows.setdefault(ts, {})[col + suffix] = _num(val)
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "date"
    return df


def fetch_tw_unemployment() -> pd.DataFrame:
    raw = _decode(_get(S.TW_SOURCES["unemployment"]["url"], verify=S.TW_CA_BUNDLE))
    out = {}
    for blk in re.findall(r"<失業率>(.*?)</失業率>", raw, re.S):
        pm = re.search(r"<年月別_Year_and_month>(.*?)</年月別_Year_and_month>", blk)
        tm = re.search(r"<總計_Total_百分比>(.*?)</總計_Total_百分比>", blk)
        if not pm or not tm:
            continue
        ts = _period_to_ts(pm.group(1))
        if ts is None:          # annual row — skip, do not fold into January
            continue
        out[ts] = {"unemployment_rate": _num(tm.group(1))}
    df = pd.DataFrame.from_dict(out, orient="index").sort_index()
    df.index.name = "date"
    return df


def fetch_tw_money() -> pd.DataFrame:
    txt = _decode(_get(S.TW_SOURCES["money"]["url"], verify=S.TW_CA_BUNDLE))
    rd = list(csv.reader(io.StringIO(txt)))
    hdr, body = rd[0], rd[1:]
    # Header labels use full-width Ｍ１Ａ etc.; normalise before matching.
    def norm(h): return h.replace("　", "").replace(" ", "").translate(
        str.maketrans("ＭＡＢ１２", "MAB12"))
    want = {"貨幣總計數-M1A-原始值": "m1a", "貨幣總計數-M1A-年增率": "m1a_yoy",
            "貨幣總計數-M1B-原始值": "m1b", "貨幣總計數-M1B-年增率": "m1b_yoy",
            "貨幣總計數-M2-原始值":  "m2",  "貨幣總計數-M2-年增率":  "m2_yoy"}
    idx = {}
    for i, h in enumerate(hdr):
        n = norm(h)
        if n in want:
            idx[want[n]] = i
    missing = set(want.values()) - set(idx)
    if missing:
        raise RuntimeError(f"CBC money CSV header changed; missing {sorted(missing)}. "
                           f"Header was: {hdr}")
    out = {}
    for row in body:
        if not row:
            continue
        ts = _period_to_ts(row[0])
        if ts is None:
            continue
        out[ts] = {c: _num(row[i]) for c, i in idx.items() if i < len(row)}
    df = pd.DataFrame.from_dict(out, orient="index").sort_index()
    df.index.name = "date"
    return df


def fetch_tw_pmi() -> pd.DataFrame:
    txt = _decode(_get(S.TW_SOURCES["pmi"]["url"]))
    rd = list(csv.reader(io.StringIO(txt)))
    out = {}
    for row in rd[1:]:
        if len(row) < 3:
            continue
        ts = _period_to_ts(row[0])
        if ts is None:
            continue
        out[ts] = {"pmi": _num(row[1]), "nmi": _num(row[2])}
    df = pd.DataFrame.from_dict(out, orient="index").sort_index()
    df.index.name = "date"
    return df


def fetch_tw_cycle() -> pd.DataFrame:
    """國發會 景氣指標: composite indices, monitoring score/signal, and the
    leading/coincident component series that matter for TAIEX."""
    blob = _get(S.TW_SOURCES["cycle"]["url"])
    zf = zipfile.ZipFile(io.BytesIO(blob))
    names = {n.split("/")[-1]: n for n in zf.namelist()}

    def read(fname):
        hit = next((v for k, v in names.items() if k.startswith(fname)), None)
        if hit is None:
            raise RuntimeError(f"{fname} not in NDC zip: {list(names)}")
        return list(csv.reader(io.StringIO(_decode(zf.read(hit)))))

    frames = []

    main = read("景氣指標與燈號")
    m = {}
    for row in main[1:]:
        if len(row) < 9:
            continue
        ts = _period_to_ts(row[0])
        if ts is None:
            continue
        sig = str(row[8]).strip()
        m[ts] = {
            "leading_idx": _num(row[1]),    "leading_idx_nt": _num(row[2]),
            "coincident_idx": _num(row[3]), "coincident_idx_nt": _num(row[4]),
            "lagging_idx": _num(row[5]),    "lagging_idx_nt": _num(row[6]),
            "monitor_score": _num(row[7]),
            "monitor_signal": sig if sig not in ("-", "", "--") else None,
        }
    frames.append(pd.DataFrame.from_dict(m, orient="index"))

    # Leading components — col 1 is 外銷訂單動向指數, col 6 半導體設備進口
    lead = read("領先指標構成項目")
    l = {}
    for row in lead[1:]:
        ts = _period_to_ts(row[0]) if row else None
        if ts is None:
            continue
        l[ts] = {"export_orders_dci": _num(row[1]) if len(row) > 1 else np.nan,
                 "semi_equip_imports": _num(row[6]) if len(row) > 6 else np.nan}
    frames.append(pd.DataFrame.from_dict(l, orient="index"))

    # Coincident components — 工業生產指數, 製造業銷售量指數, 海關出口值
    coin = read("同時指標構成項目")
    c = {}
    for row in coin[1:]:
        ts = _period_to_ts(row[0]) if row else None
        if ts is None:
            continue
        c[ts] = {"industrial_production": _num(row[1]) if len(row) > 1 else np.nan,
                 "mfg_sales_idx": _num(row[3]) if len(row) > 3 else np.nan,
                 "customs_exports": _num(row[6]) if len(row) > 6 else np.nan}
    frames.append(pd.DataFrame.from_dict(c, orient="index"))

    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    return df


# ── Japan ───────────────────────────────────────────────────────────────────
def _fred(series_id: str) -> pd.Series:
    txt = _decode(_get(S.FRED_CSV.format(id=series_id), ua=S.FRED_UA))
    rows = list(csv.reader(io.StringIO(txt)))
    idx, vals = [], []
    for r in rows[1:]:
        if len(r) < 2 or r[1] in ("", "."):
            continue
        try:
            idx.append(pd.Timestamp(r[0])); vals.append(float(r[1]))
        except (ValueError, TypeError):
            continue
    if not idx:
        raise RuntimeError(f"FRED {series_id}: no parseable rows")
    return pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id).sort_index()


def fetch_jp_cpi() -> pd.DataFrame:
    """Japan CPI from the OECD SDMX COICOP-2018 dataflow (FRED's is dead)."""
    cfg = S.JP_OECD_CPI
    url = f"{cfg['base']}/{cfg['key']}?startPeriod=1970-01&format=csvfile"
    txt = _decode(_get(url, timeout=(15, 300)))
    d = pd.read_csv(io.StringIO(txt))
    d = d[d["FREQ"] == "M"]
    out = {}
    for exp, col in cfg["expenditures"].items():
        sub = d[d["EXPENDITURE"] == exp]
        # YoY growth is TRANSFORMATION=GY; the index LEVEL is TRANSFORMATION=_Z
        # with UNIT_MEASURE=IX (not TRANSFORMATION=IX — that code does not exist).
        for mask, suffix in (
            (sub["TRANSFORMATION"] == "GY", "_yoy"),
            ((sub["TRANSFORMATION"] == "_Z") & (sub["UNIT_MEASURE"] == "IX"), ""),
        ):
            s = sub[mask]
            if s.empty:
                continue
            # 'N' sorts before 'S', so non-seasonally-adjusted wins the setdefault.
            s = s.sort_values("ADJUSTMENT")
            for _, r in s.iterrows():
                ts = _period_to_ts(r["TIME_PERIOD"])
                if ts is None:
                    continue
                out.setdefault(ts, {}).setdefault(col + suffix, _num(r["OBS_VALUE"]))
    df = pd.DataFrame.from_dict(out, orient="index").sort_index()
    df.index.name = "date"
    return df


def fetch_jp_monthly() -> pd.DataFrame:
    frames = [fetch_jp_cpi()]
    cols = {}
    for col, cfg in S.JP_FRED_MONTHLY.items():
        s = _fred(cfg["id"])
        s.index = s.index.to_period("M").to_timestamp()
        cols[col] = s
    frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, axis=1).sort_index()
    df.index.name = "date"
    return df


def fetch_jp_daily() -> pd.DataFrame:
    cols = {c: _fred(cfg["id"]) for c, cfg in S.JP_FRED_DAILY.items()}
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df.dropna(how="all")
