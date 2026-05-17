"""
BaseAsset — a pd.DataFrame subclass for a single CTA instrument.

Rows  : datetime bars (the trading index)
Cols  : open, high, low, close, volume  (OHLCV)

Metadata preserved through pandas operations (via _metadata):
    symbol           ticker / contract code (e.g. 'MTX')
    time_granularity bar size: '1d', '1h', '30m', …
    start_time       first bar as pd.Timestamp
    end_time         last  bar as pd.Timestamp
"""

from __future__ import annotations

import operator

import numpy as np
import pandas as pd


_OHLCV    = ["open", "high", "low", "close", "volume"]
_EXTRA    = ["settlement", "oi", "bid", "ask"]   # optional — present when loaded from TAIFEX CSV


class BaseAsset(pd.DataFrame):
    """
    Single-instrument OHLCV container with optional extra columns.

    Required columns : open, high, low, close, volume
    Optional columns : settlement, oi (open interest), bid, ask

    Analogous to BaseMatrix in the f-package but for a single time series
    instead of a cross-sectional universe.
    """

    _metadata = ["symbol", "time_granularity", "start_time", "end_time"]

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        symbol: str = "",
        time_granularity: str = "1d",
    ):
        if data is None:
            data = pd.DataFrame(columns=_OHLCV)

        missing = [c for c in _OHLCV if c not in data.columns]
        if missing:
            raise ValueError(f"data is missing OHLCV columns: {missing}")

        super().__init__(data)

        self.symbol           = symbol
        self.time_granularity = time_granularity
        self.start_time       = pd.Timestamp(data.index[0])  if len(data) else pd.NaT
        self.end_time         = pd.Timestamp(data.index[-1]) if len(data) else pd.NaT

    # ── pandas plumbing ───────────────────────────────────────────────────────

    @property
    def _constructor(self) -> type:
        return pd.DataFrame

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        symbol: str = "",
        time_granularity: str = "1d",
    ) -> "BaseAsset":
        """Wrap a DataFrame as a BaseAsset, preserving any extra known columns."""
        keep = _OHLCV + [c for c in _EXTRA if c in df.columns]
        return cls(data=df[keep].copy(), symbol=symbol, time_granularity=time_granularity)

    # ── column accessors ──────────────────────────────────────────────────────

    @property
    def open(self) -> pd.Series:
        return self["open"]

    @property
    def high(self) -> pd.Series:
        return self["high"]

    @property
    def low(self) -> pd.Series:
        return self["low"]

    @property
    def close(self) -> pd.Series:
        return self["close"]

    @property
    def volume(self) -> pd.Series:
        return self["volume"]

    @property
    def settlement(self) -> pd.Series:
        """Daily settlement price (結算價)."""
        return self["settlement"] if "settlement" in self.columns else pd.Series(dtype=float)

    @property
    def oi(self) -> pd.Series:
        """Open interest — number of outstanding contracts (未沖銷契約數)."""
        return self["oi"] if "oi" in self.columns else pd.Series(dtype=float)

    @property
    def bid(self) -> pd.Series:
        """Closing best bid price (最後最佳買價)."""
        return self["bid"] if "bid" in self.columns else pd.Series(dtype=float)

    @property
    def ask(self) -> pd.Series:
        """Closing best ask price (最後最佳賣價)."""
        return self["ask"] if "ask" in self.columns else pd.Series(dtype=float)

    @property
    def spread(self) -> pd.Series:
        """Closing bid-ask spread in index points (ask − bid)."""
        return (self.ask - self.bid).rename("spread")

    @property
    def mid(self) -> pd.Series:
        """Closing mid price (bid + ask) / 2."""
        return ((self.bid + self.ask) / 2).rename("mid")

    # ── derived series ────────────────────────────────────────────────────────

    @property
    def returns(self) -> pd.Series:
        """Bar-to-bar percentage return of the close price."""
        return self["close"].pct_change().rename("returns")

    @property
    def log_returns(self) -> pd.Series:
        return np.log(self["close"] / self["close"].shift(1)).rename("log_returns")

    @property
    def price_change(self) -> pd.Series:
        """Absolute close-to-close change in index points."""
        return self["close"].diff().rename("price_change")

    @property
    def price_range(self) -> pd.Series:
        """Intrabar high-low range."""
        return (self["high"] - self["low"]).rename("price_range")

    @property
    def typical_price(self) -> pd.Series:
        return ((self["high"] + self["low"] + self["close"]) / 3).rename("typical_price")

    # ── slicing ───────────────────────────────────────────────────────────────

    def slice(
        self,
        start: str | pd.Timestamp | None = None,
        end:   str | pd.Timestamp | None = None,
    ) -> "BaseAsset":
        df = pd.DataFrame(self)
        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]
        return BaseAsset.from_df(df, symbol=self.symbol, time_granularity=self.time_granularity)

    # ── info ──────────────────────────────────────────────────────────────────

    def info_str(self) -> str:
        lines = [
            f"BaseAsset  {self.symbol}",
            f"  time_granularity : {self.time_granularity}",
            f"  bars             : {len(self):,}",
            f"  start            : {self.start_time.date() if pd.notna(self.start_time) else 'N/A'}",
            f"  end              : {self.end_time.date()   if pd.notna(self.end_time)   else 'N/A'}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.info_str() + "\n\n" + super().__repr__()

    # ── arithmetic (inf → NaN safety, mirrors BaseMatrix) ────────────────────

    def _arith_op(self, other, op_func) -> pd.DataFrame:
        other_val = pd.DataFrame(other) if isinstance(other, BaseAsset) else other
        return op_func(pd.DataFrame(self), other_val).replace([np.inf, -np.inf], np.nan)

    def __add__     (self, o): return self._arith_op(o, operator.add)
    def __sub__     (self, o): return self._arith_op(o, operator.sub)
    def __mul__     (self, o): return self._arith_op(o, operator.mul)
    def __truediv__ (self, o):
        denom = pd.DataFrame(o) if isinstance(o, BaseAsset) else o
        if isinstance(denom, (int, float)):
            denom = denom or np.nan
            return pd.DataFrame(self).div(denom).replace([np.inf, -np.inf], np.nan)
        denom = pd.DataFrame(denom).where(pd.DataFrame(denom) != 0)
        return pd.DataFrame(self).div(denom).replace([np.inf, -np.inf], np.nan)
    def __floordiv__(self, o): return self._arith_op(o, operator.floordiv)
    def __mod__     (self, o): return self._arith_op(o, operator.mod)
    def __pow__     (self, o): return self._arith_op(o, operator.pow)
    def __neg__     (self):    return self._arith_op(-1, operator.mul)

    def __radd__    (self, o): return self._arith_op(o, operator.add)
    def __rsub__    (self, o): return self._arith_op(o, operator.sub)
    def __rmul__    (self, o): return self._arith_op(o, operator.mul)
    def __rtruediv__(self, o): return self._arith_op(o, operator.truediv)


__all__ = ["BaseAsset"]
