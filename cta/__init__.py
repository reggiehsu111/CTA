from .asset import BaseAsset
from .simulate import (
    Simulate, simulate,                       # capital is canonical; lowercase = back-compat alias
    SimulateAll, normalize_signal,
    load_asset, available_assets,
    set_date_range, get_date_range,
)
from .simulate_dollars import simulate_by_dollars
from .simulate_intraday import SimulateIntraday, SimulateAllIntraday, SimulateIntradayCalendar
from .session_strategies import simulate_midday_short_night_long
from .decomposition import RegimeDecomposition
from .operators import (
    set_active_asset,
    Prices, Returns,
    Lag, Lead,
    Filter,
    ForwardFill,
    InstMean, InstStdev, InstSkew, InstSum, InstCorr,
    InstRank, InstZScore,
    Diff, PctChange,
    Sign, Abs,
    Event, Caar,
    load_tsmc_ea_dates,
)

__all__ = [
    "BaseAsset",
    "Simulate", "simulate", "SimulateAll", "normalize_signal",
    "load_asset", "available_assets",
    "set_date_range", "get_date_range",
    "simulate_by_dollars", "simulate_midday_short_night_long",
    "SimulateIntraday", "SimulateAllIntraday", "SimulateIntradayCalendar",
    "RegimeDecomposition",
    "set_active_asset", "Prices", "Returns",
    "Lag", "Lead",
    "Filter",
    "ForwardFill",
    "InstMean", "InstStdev", "InstSkew", "InstSum", "InstCorr",
    "InstRank", "InstZScore",
    "Diff", "PctChange",
    "Sign", "Abs",
    "Event", "Caar",
    "load_tsmc_ea_dates",
]
