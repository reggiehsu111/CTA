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
from .three_majors import (
    load_three_majors, load_three_majors_wide, align_three_majors_to_asset,
    available_products as available_three_majors_products,
    available_identities as available_three_majors_identities,
    available_metrics as available_three_majors_metrics,
    show_three_majors_catalog,
)
from .options import (
    Options,
    load_option, load_atm_option,
    option_strikes, option_expiries, front_month_expiry,
    load_option_daily_total,
    load_pcr,
    load_atm_straddle_pct,
    load_put_skew,
    load_front_share_oi,
)
from .large_trader import (
    load_large_trader, load_large_trader_wide,
    available_large_trader_commodities, available_large_trader_metrics,
    show_large_trader_catalog,
)
from .bars import load_1min, bar_coverage, resolve_symbol
from .ticks import (
    load_ticks, tick_quotes, tick_daily_summary, available_tick_days, session_date,
)
from .decomposition import RegimeDecomposition
from .signal_stats import signal_stats, batch_signal_stats, composite_score
from .gates import house_gates, beta_per_w
from .sweep_report import sweep_headline, paired_headline, se_sr, icc_neff
from .tsmc_events import load_tsmc_events, load_tsmc_event_dates
from .us_indexes import load_us_index, load_us_index_tw, available_us_tickers
from .nfci import load_nfci, load_nfci_pit, load_nfci_tw, available_nfci_fields
from .asia_macro import (
    load_tw_macro, load_tw_macro_tw, load_jp_macro, load_jp_macro_tw,
    load_jp_market, load_jp_market_tw,
    available_tw_macro_fields, available_jp_macro_fields,
)
from .us_macro import (
    load_us_price, load_us_price_tw, load_us_price_yoy_tw,
    load_us_rate,  load_us_rate_tw,
    load_us_labor_monthly, load_us_labor_monthly_tw,
    load_us_labor_weekly,  load_us_labor_weekly_tw,
    load_us_risk, load_us_risk_tw,
    available_us_price_fields, available_us_rate_fields,
    available_us_labor_monthly_fields,
    available_us_labor_weekly_fields,
    available_us_risk_fields,
)
from .global_macro import (
    load_macro, load_macro_tw, load_macro_yoy_tw,
    macro_catalog, macro_pub_lag,
    available_macro_series, show_macro_catalog,
    MACRO_MIN_PUB_LAG_DAYS,
)
from .twse_margin import (
    load_margin_summary,
    load_margin_by_stock,
    load_market_maintenance_ratio,
)
from .operators import (
    set_active_asset,
    Prices, Returns,
    Lag, Lead,
    Filter,
    ForwardFill,
    InstMean, InstStdev, InstSkew, InstSum, InstCorr,
    InstRank, InstZScore,
    Diff, PctChange,
    Sign, Abs, Date,
    Event, Caar, Casr,
    EventFFill, EventRollingFFill,
    load_tsmc_ea_dates,
)

__all__ = [
    "BaseAsset",
    "Simulate", "simulate", "SimulateAll", "normalize_signal",
    "load_asset", "available_assets",
    "set_date_range", "get_date_range",
    "simulate_by_dollars", "simulate_midday_short_night_long",
    "load_three_majors", "load_three_majors_wide", "align_three_majors_to_asset",
    "available_three_majors_products",
    "available_three_majors_identities",
    "available_three_majors_metrics",
    "show_three_majors_catalog",
    "Options",
    "load_option", "load_atm_option",
    "option_strikes", "option_expiries", "front_month_expiry",
    "load_option_daily_total",
    "load_pcr", "load_atm_straddle_pct", "load_put_skew", "load_front_share_oi",
    "load_1min", "bar_coverage",
    "load_ticks", "tick_quotes", "tick_daily_summary", "available_tick_days",
    "session_date",
    "signal_stats", "batch_signal_stats", "composite_score",
    "house_gates", "beta_per_w",
    "sweep_headline", "paired_headline", "se_sr", "icc_neff",
    "load_tsmc_events", "load_tsmc_event_dates",
    "load_us_index", "load_us_index_tw", "available_us_tickers",
    "load_large_trader", "load_large_trader_wide",
    "available_large_trader_commodities", "available_large_trader_metrics",
    "show_large_trader_catalog",
    "SimulateIntraday", "SimulateAllIntraday", "SimulateIntradayCalendar",
    "RegimeDecomposition",
    "set_active_asset", "Prices", "Returns",
    "Lag", "Lead",
    "Filter",
    "ForwardFill",
    "InstMean", "InstStdev", "InstSkew", "InstSum", "InstCorr",
    "InstRank", "InstZScore",
    "Diff", "PctChange",
    "Sign", "Abs", "Date",
    "Event", "Caar", "Casr",
    "EventFFill", "EventRollingFFill",
    "load_tsmc_ea_dates",
]
