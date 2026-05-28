import argparse
from dataclasses import dataclass
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np

from product.product_universe import load_product_universe

# ----------------------------
# CONFIG
# ----------------------------
UNIVERSE_CONFIG = load_product_universe()
SECTORS = UNIVERSE_CONFIG["sectors"]
COMMODITIES = UNIVERSE_CONFIG["commodities"]
tickers = UNIVERSE_CONFIG["tickers"]

benchmark = "SPY"
macro_tickers = {
    "VIX": "^VIX",
    "TNX": "^TNX",
    "WTI": "CL=F",
}

INITIAL_CAPITAL = 100000
START_DATE = "2007-01-01"
MIN_HISTORY_DAYS_FOR_RANKING = 20

OUTPUT_DIR = Path(__file__).resolve().parent

PRODUCT_DESCRIPTIONS = UNIVERSE_CONFIG["product_descriptions"]
LEVERAGED_TICKERS = UNIVERSE_CONFIG["leveraged_tickers"]


@dataclass
class BacktestRunResult:
    baseline_equity: pd.Series
    enhanced_equity: pd.Series
    spy_equity: pd.Series
    baseline_trades: pd.DataFrame
    enhanced_trades: pd.DataFrame
    baseline_state: dict
    enhanced_state: dict
    baseline_score: pd.DataFrame
    enhanced_score: pd.DataFrame
    regime: pd.Series
    feature_details: dict
    data: pd.DataFrame
    ohlc_data: dict
    baseline_signal_date: object
    baseline_signal_regime: object
    baseline_signal_holdings: list
    enhanced_signal_date: object
    enhanced_signal_regime: object
    enhanced_signal_holdings: list

# ----------------------------
# DATA DOWNLOAD
# ----------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(description="US sector rotation backtest")
    parser.add_argument(
        "--use-macro-overlay",
        action="store_true",
        help="Apply a simple macro overlay to sector scores before ranking.",
    )
    parser.add_argument(
        "--stoploss-pct",
        type=float,
        default=None,
        help="Optional stop loss percentage from the highest close since entry. Example: 5 means exit after a 5%% drop from the highest close and keep that weight in cash until the next rebalance.",
    )
    parser.add_argument(
        "--stoploss-scope",
        choices=["all", "leveraged"],
        default="all",
        help="Apply stop loss to all positions or only to leveraged trades.",
    )
    parser.add_argument(
        "--stoploss-after-exit",
        choices=["cash", "redistribute"],
        default="cash",
        help="After a stop loss, keep stopped allocation in cash or redistribute it across remaining positions until the next rebalance.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        choices=["regime", "monthly", "daily"],
        default="monthly",
        help="Rebalance daily, on the first trading day of each month by default, or only when market regime changes.",
    )
    parser.add_argument(
        "--trailing-stop-pct",
        type=float,
        default=None,
        help="Optional trailing stop percentage from the highest close since entry.",
    )
    parser.add_argument(
        "--trailing-stop-scope",
        choices=["all", "leveraged"],
        default="all",
        help="Apply trailing stop to all positions or only leveraged trades.",
    )
    parser.add_argument(
        "--enable-velocity-rotation",
        action="store_true",
        help="Allow an early rebalance when held ETFs lose short-term momentum and higher-momentum alternatives are available.",
    )
    parser.add_argument(
        "--velocity-window",
        type=int,
        default=3,
        help="Trading-day window used to compare recent momentum for velocity rotation.",
    )
    parser.add_argument(
        "--velocity-drop-pct",
        type=float,
        default=5.0,
        help="Minimum drop in momentum, in percentage points, required to trigger a velocity rebalance.",
    )
    parser.add_argument(
        "--velocity-candidate-buffer-pct",
        type=float,
        default=0.0,
        help="Require alternative ETFs to beat fading holdings by at least this many percentage points of short-term momentum.",
    )
    parser.add_argument(
        "--compare-stoploss-pct",
        type=float,
        default=None,
        help="Run the backtest twice, once without stop loss and once with this stop loss percentage, then print a side-by-side comparison.",
    )
    parser.add_argument(
        "--compare-stoploss-scope",
        choices=["all", "leveraged"],
        default="all",
        help="Scope to use with --compare-stoploss-pct.",
    )
    parser.add_argument(
        "--log-ranked-scores",
        action="store_true",
        help="Print the latest ranked score breakdown for debugging.",
    )
    return parser


def parse_args(argv=None):
    return build_arg_parser().parse_args(argv)

def normalize_download_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df

def safe_download(ticker):
    df = yf.download(ticker, start=START_DATE, progress=False, threads=False)

    if df is None or df.empty:
        return None, None

    df = normalize_download_columns(df)

    if "Adj Close" in df.columns:
        series = df["Adj Close"]
    elif "Close" in df.columns:
        series = df["Close"]
    else:
        return None, None

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    if not isinstance(series, pd.Series):
        return None, None

    series = series.dropna()
    if series.empty:
        return None, None

    ohlc = None
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        ohlc = df[["Open", "High", "Low", "Close"]].copy()

        # Adjust OHLC for splits/dividends so stop logic matches adjusted-close returns.
        if "Adj Close" in df.columns:
            close_for_factor = df["Close"].replace(0, np.nan)
            adjustment_factor = (df["Adj Close"] / close_for_factor).replace([np.inf, -np.inf], np.nan)
            for col in ["Open", "High", "Low", "Close"]:
                ohlc[col] = ohlc[col] * adjustment_factor

        ohlc = ohlc.dropna(how="all")

    return series.rename(ticker), ohlc

def get_data():
    dfs = []
    ohlc_data = {}
    for t in tickers + [benchmark] + list(macro_tickers.values()):
        print(f"Downloading {t}...")
        series, ohlc = safe_download(t)
        if series is not None:
            dfs.append(series)
        if t in tickers and ohlc is not None:
            ohlc_data[t] = ohlc

    data = pd.concat(dfs, axis=1).sort_index().ffill()
    required_columns = [benchmark] + list(macro_tickers.values())
    data = data.dropna(subset=required_columns)
    return data, ohlc_data

# ----------------------------
# FEATURES
# ----------------------------
def apply_macro_overlay(score_baseline, score_enhanced, data):
    spy = data[benchmark]
    vix = data["^VIX"]
    tnx = data["^TNX"]
    oil = data["CL=F"]

    spy_ma200 = spy.rolling(200).mean()
    risk_on_market = (spy > spy_ma200).astype(float)
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()

    tnx_3m_change = tnx.diff(63)
    oil_3m_mom = oil.pct_change(63, fill_method=None)
    oil_1m_mom = oil.pct_change(21, fill_method=None)

    tech_tilt = (
        (tnx_3m_change < 0).astype(float)
        + (vix_z < 0.5).astype(float)
        + risk_on_market
    ) / 3 * 2 - 1

    energy_tilt = (
        (oil_3m_mom > 0).astype(float)
        + (oil_1m_mom > 0).astype(float)
        + (tnx_3m_change > 0).astype(float)
    ) / 3 * 2 - 1

    utilities_tilt = (
        (tnx_3m_change < 0).astype(float)
        + (vix_z > 0.5).astype(float)
        + (risk_on_market == 0).astype(float)
    ) / 3 * 2 - 1

    materials_tilt = (
        (oil_3m_mom > 0).astype(float)
        + (tnx_3m_change > 0).astype(float)
        + risk_on_market
    ) / 3 * 2 - 1

    baseline_overlay = pd.DataFrame(0.0, index=score_baseline.index, columns=score_baseline.columns)
    enhanced_overlay = pd.DataFrame(0.0, index=score_enhanced.index, columns=score_enhanced.columns)

    baseline_overlay["XLK"] += 0.12 * tech_tilt
    baseline_overlay["SMH"] += 0.15 * tech_tilt
    baseline_overlay["TQQQ"] += 0.10 * tech_tilt
    baseline_overlay["XLE"] += 0.15 * energy_tilt
    baseline_overlay["NLR"] += 0.15 * energy_tilt
    baseline_overlay["XLB"] += 0.08 * materials_tilt
    baseline_overlay["XLU"] += 0.10 * utilities_tilt

    enhanced_overlay["XLK"] += 0.15 * tech_tilt
    enhanced_overlay["SMH"] += 0.18 * tech_tilt
    enhanced_overlay["TQQQ"] += 0.12 * tech_tilt
    enhanced_overlay["XLE"] += 0.18 * energy_tilt
    enhanced_overlay["NLR"] += 0.18 * energy_tilt
    enhanced_overlay["XLB"] += 0.10 * materials_tilt
    enhanced_overlay["XLU"] += 0.12 * utilities_tilt

    macro_details = {
        "tech_tilt": tech_tilt,
        "energy_tilt": energy_tilt,
        "utilities_tilt": utilities_tilt,
        "materials_tilt": materials_tilt,
        "tnx_3m_change": tnx_3m_change,
        "oil_3m_mom": oil_3m_mom,
        "oil_1m_mom": oil_1m_mom,
        "vix_z": vix_z,
    }

    return score_baseline.add(baseline_overlay, fill_value=0), score_enhanced.add(enhanced_overlay, fill_value=0), macro_details

def compute_features(data, use_macro_overlay=False):

    etf = data.reindex(columns=tickers)
    spy = data[benchmark]
    history_days = etf.notna().astype(int).cumsum()

    # returns
    ret_1m = etf.pct_change(21, fill_method=None)
    ret_3m = etf.pct_change(63, fill_method=None)
    ret_6m = etf.pct_change(126, fill_method=None)
    spy_ret_1m = spy.pct_change(21, fill_method=None)
    spy_ret_3m = spy.pct_change(63, fill_method=None)

    rs_1m = ret_1m.sub(spy_ret_1m, axis=0)
    rs_3m = ret_3m.sub(spy_ret_3m, axis=0)

    # trend
    ma50 = etf.rolling(50, min_periods=20).mean()
    ma200 = etf.rolling(200, min_periods=50).mean()

    trend_vs_ma50 = (etf > ma50).where(ma50.notna()).astype(float)
    trend_vs_ma200 = (etf > ma200).where(ma200.notna()).astype(float)
    trend_ma_stack = (ma50 > ma200).where(ma50.notna() & ma200.notna()).astype(float)
    trend_component_count = (
        trend_vs_ma50.notna().astype(float)
        + trend_vs_ma200.notna().astype(float)
        + trend_ma_stack.notna().astype(float)
    )
    trend = (
        trend_vs_ma50.fillna(0)
        + trend_vs_ma200.fillna(0)
        + trend_ma_stack.fillna(0)
    ).div(trend_component_count.where(trend_component_count > 0))

    # volatility
    vol = etf.pct_change(fill_method=None).rolling(20, min_periods=10).std()

    # new features
    roc_1m = etf.pct_change(21)
    breakout_high = etf.rolling(50, min_periods=20).max()
    breakout = (etf >= breakout_high).where(breakout_high.notna()).astype(float)
    vol_signal_threshold = vol.rolling(50, min_periods=20).mean()
    vol_signal = (vol > vol_signal_threshold).where(vol_signal_threshold.notna()).astype(float)

    def zscore(df):
        return (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)

    def weighted_score(components):
        weighted_sum = None
        available_weight = None
        for weight, values in components:
            current_sum = values.fillna(0) * weight
            current_weight = values.notna().astype(float) * weight
            weighted_sum = current_sum if weighted_sum is None else weighted_sum.add(current_sum, fill_value=0)
            available_weight = current_weight if available_weight is None else available_weight.add(current_weight, fill_value=0)
        return weighted_sum.div(available_weight.where(available_weight > 0))

    z_rs_1m = zscore(rs_1m)
    z_rs_3m = zscore(rs_3m)
    z_ret_1m = zscore(ret_1m)
    z_ret_3m = zscore(ret_3m)
    z_ret_6m = zscore(ret_6m)
    z_vol = zscore(vol)
    z_roc_1m = zscore(roc_1m)
    z_vol_signal = zscore(vol_signal)

    # Let younger ETFs use the best available momentum horizon instead of
    # disappearing from the ranking until they accumulate 3M/6M history.
    z_rs_effective = z_rs_3m.where(rs_3m.notna(), z_rs_1m)
    z_ret_long_effective = z_ret_6m.where(ret_6m.notna(), z_ret_3m.where(ret_3m.notna(), z_ret_1m))

    # ----------------------------
    # BASELINE SCORE
    # ----------------------------
    score_baseline = weighted_score(
        [
            (0.30, z_rs_effective),
            (0.25, z_ret_1m),
            (0.20, trend),
            (0.15, z_ret_long_effective),
            (0.10, -z_vol),
        ]
    ).rolling(3).mean()
    score_baseline = score_baseline.where(history_days >= MIN_HISTORY_DAYS_FOR_RANKING)

    # ----------------------------
    # ENHANCED SCORE
    # ----------------------------
    score_enhanced = weighted_score(
        [
            (0.30, z_rs_effective),
            (0.25, z_roc_1m),
            (0.20, breakout),
            (0.15, z_vol_signal),
            (0.10, trend),
        ]
    ).rolling(3).mean()
    score_enhanced = score_enhanced.where(history_days >= MIN_HISTORY_DAYS_FOR_RANKING)

    # ----------------------------
    # REGIME
    # ----------------------------
    vix = data["^VIX"]
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()

    spy_ma200 = spy.rolling(200).mean()

    regime = pd.Series(index=data.index, dtype="object")

    for i in range(len(data)):
        if spy.iloc[i] > spy_ma200.iloc[i] and vix_z.iloc[i] < 0.5:
            regime.iloc[i] = "RISK_ON"
        elif vix_z.iloc[i] > 1.5 or spy.iloc[i] < spy_ma200.iloc[i]:
            regime.iloc[i] = "RISK_OFF"
        else:
            regime.iloc[i] = "NEUTRAL"

    feature_details = {
        "baseline": {
            "rs_1m": rs_1m,
            "rs_3m": rs_3m,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "vol": vol,
            "trend": trend,
            "z_rs_1m": z_rs_1m,
            "z_rs_3m": z_rs_3m,
            "z_ret_1m": z_ret_1m,
            "z_ret_3m": z_ret_3m,
            "z_ret_6m": z_ret_6m,
            "z_rs_effective": z_rs_effective,
            "z_ret_long_effective": z_ret_long_effective,
            "z_vol": z_vol,
            "score": score_baseline,
        },
        "enhanced": {
            "rs_1m": rs_1m,
            "rs_3m": rs_3m,
            "roc_1m": roc_1m,
            "breakout": breakout,
            "vol_signal": vol_signal,
            "trend": trend,
            "z_rs_1m": z_rs_1m,
            "z_rs_3m": z_rs_3m,
            "z_rs_effective": z_rs_effective,
            "z_roc_1m": z_roc_1m,
            "z_vol_signal": z_vol_signal,
            "score": score_enhanced,
        },
    }

    macro_details = None
    if use_macro_overlay:
        score_baseline, score_enhanced, macro_details = apply_macro_overlay(score_baseline, score_enhanced, data)

    feature_details["macro"] = macro_details

    return score_baseline, score_enhanced, regime, feature_details

def latest_available_date(series):
    valid_index = series.dropna().index
    if len(valid_index) == 0:
        return "N/A"
    return valid_index[-1].date()

def latest_signal_positions(score, regime):
    latest_date = score.dropna(how="all").index[-1]
    latest_regime = regime.loc[latest_date]
    ranked = score.loc[latest_date].dropna().sort_values(ascending=False)

    if latest_regime == "RISK_ON":
        positions = list(ranked.index[:3])
    elif latest_regime == "NEUTRAL":
        positions = list(ranked.index[:2])
    else:
        defensive = ["XLP","XLU","XLV","IAU", "REMX"]
        positions = [x for x in ranked.index if x in defensive][:2]

    return latest_date.date(), latest_regime, positions

def select_positions(ranked, current_regime):
    if current_regime == "RISK_ON":
        return list(ranked.index[:3]), [0.5, 0.3, 0.2]
    if current_regime == "NEUTRAL":
        positions = list(ranked.index[:2])
        return positions, [0.5, 0.5]

    defensive = ["XLP","XLU","XLV","IAU", "REMX"]
    positions = [x for x in ranked.index if x in defensive][:2]
    return positions, [0.5, 0.5]

def stoploss_applies_to_ticker(ticker, stoploss_scope):
    if stoploss_scope == "all":
        return True
    return ticker in LEVERAGED_TICKERS

def scope_applies_to_ticker(ticker, scope):
    if scope == "all":
        return True
    return ticker in LEVERAGED_TICKERS

def should_trigger_velocity_rebalance(
    i,
    score,
    regime,
    current_positions,
    velocity_returns,
    velocity_window,
    velocity_drop_pct,
    velocity_candidate_buffer_pct,
):
    if not current_positions or i < velocity_window * 2:
        return False

    current_regime = regime.iloc[i]
    ranked = score.iloc[i].dropna().sort_values(ascending=False)
    desired_positions, _ = select_positions(ranked, current_regime)

    if set(desired_positions) == set(current_positions):
        return False

    current_velocity = velocity_returns.iloc[i]
    previous_velocity = velocity_returns.shift(velocity_window).iloc[i]

    fading_scores = []
    for ticker in current_positions:
        curr = current_velocity.get(ticker, np.nan)
        prev = previous_velocity.get(ticker, np.nan)
        if pd.notna(curr) and pd.notna(prev) and prev > 0 and curr < 0 and (prev - curr) >= velocity_drop_pct:
            fading_scores.append(curr)

    if not fading_scores:
        return False

    alternative_scores = [
        current_velocity.get(ticker, np.nan)
        for ticker in desired_positions
        if ticker not in current_positions and pd.notna(current_velocity.get(ticker, np.nan))
    ]
    if not alternative_scores:
        return False

    return max(alternative_scores) >= (min(fading_scores) + velocity_candidate_buffer_pct)

def format_percent(value):
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}%"

def format_price(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"

def format_gain_pct(value):
    if pd.isna(value):
        return "N/A"
    return format_percent(value)

def calculate_gain_pct(current_price, entry_price):
    if (
        pd.isna(current_price)
        or entry_price in (None, 0)
        or pd.isna(entry_price)
    ):
        return np.nan
    return ((current_price / entry_price) - 1) * 100

def build_position_snapshot(date, positions, weights, price_data, entry_prices):
    if not positions:
        return "CASH"

    snapshot_parts = []
    for ticker, weight in zip(positions, weights):
        close_price = price_data.at[date, ticker] if ticker in price_data.columns else np.nan
        gain_pct = calculate_gain_pct(close_price, entry_prices.get(ticker))
        snapshot_parts.append(
            f"{ticker} = {format_percent(weight * 100)}, {format_price(close_price)}, P/L {format_gain_pct(gain_pct)}"
        )
    return "; ".join(snapshot_parts)

def append_position_row(
    trades,
    date,
    positions,
    weights,
    price_data,
    entry_prices,
    portfolio_value,
    daily_pnl,
    daily_return_pct,
    cumulative_pnl,
):
    trades.append(
        {
            "Date": date,
            "Action": "POSITION",
            "Ticker": "",
            "CapitalPct": np.nan,
            "TradePrice": np.nan,
            "GainPctSinceBuy": np.nan,
            "PositionSnapshot": build_position_snapshot(date, positions, weights, price_data, entry_prices),
            "PortfolioValue": portfolio_value,
            "DailyPnL": daily_pnl,
            "DailyReturnPct": daily_return_pct,
            "CumulativePnL": cumulative_pnl,
        }
    )

def add_product_descriptions(trades_df):
    descriptions = trades_df["Ticker"].map(PRODUCT_DESCRIPTIONS)
    empty_ticker = trades_df["Ticker"].isna() | trades_df["Ticker"].eq("")
    product_descriptions = descriptions.where(descriptions.notna(), np.where(empty_ticker, "", "Unknown product"))
    return trades_df.assign(ProductDescription=product_descriptions)

# ----------------------------
# STRATEGY WITH TRADES
# ----------------------------
def run_strategy(
    data,
    ohlc_data,
    score,
    regime,
    stoploss_pct=None,
    stoploss_scope="all",
    stoploss_after_exit="cash",
    rebalance_frequency="regime",
    trailing_stop_pct=None,
    trailing_stop_scope="all",
    enable_velocity_rotation=False,
    velocity_window=3,
    velocity_drop_pct=5.0,
    velocity_candidate_buffer_pct=0.0,
):

    capital = INITIAL_CAPITAL
    values = []
    dates = []

    current_positions = []
    current_regime = None
    weights = []
    entry_prices = {}
    entry_dates = {}
    open_buy_rows = {}
    high_water_prices = {}

    trades = []

    etf = data.reindex(columns=tickers)
    daily_returns = etf.pct_change(fill_method=None)
    velocity_returns = etf.pct_change(velocity_window, fill_method=None) * 100

    for i in range(252, len(data)):

        date = data.index[i]
        capital_at_day_start = capital

        regime_changed = regime.iloc[i] != current_regime
        month_changed = data.index[i].month != data.index[i - 1].month
        scheduled_rebalance = (
            rebalance_frequency == "daily"
            or (rebalance_frequency == "monthly" and month_changed)
        )
        velocity_rebalance = (
            enable_velocity_rotation
            and not regime_changed
            and not scheduled_rebalance
            and should_trigger_velocity_rebalance(
                i,
                score,
                regime,
                current_positions,
                velocity_returns,
                velocity_window,
                velocity_drop_pct,
                velocity_candidate_buffer_pct,
            )
        )

        if regime_changed or scheduled_rebalance or velocity_rebalance:
            current_regime = regime.iloc[i]
            ranked = score.iloc[i].dropna().sort_values(ascending=False)
            next_positions, next_weights = select_positions(ranked, current_regime)

            current_set = set(current_positions)
            next_set = set(next_positions)

            # Keep overlapping tickers in place so we don't log avoidable sell/buy churn.
            kept_positions = current_set.intersection(next_set)
            to_sell = [p for p in current_positions if p not in kept_positions]
            to_buy = [(p, w) for p, w in zip(next_positions, next_weights) if p not in kept_positions]

            for p in to_sell:
                sell_price = etf.at[date, p]
                buy_price = entry_prices.get(p)
                gain_pct = np.nan

                if pd.notna(sell_price) and buy_price not in (None, 0) and pd.notna(buy_price):
                    gain_pct = ((sell_price / buy_price) - 1) * 100

                trades.append(
                    {
                        "Date": date,
                        "Action": "SELL",
                        "Ticker": p,
                        "CapitalPct": np.nan,
                        "TradePrice": sell_price,
                        "GainPctSinceBuy": gain_pct,
                        "PositionSnapshot": "",
                    }
                )
                entry_prices.pop(p, None)
                entry_dates.pop(p, None)
                open_buy_rows.pop(p, None)
                high_water_prices.pop(p, None)

            for p, w in to_buy:
                buy_price = etf.at[date, p]
                entry_prices[p] = buy_price
                entry_dates[p] = date
                high_water_prices[p] = buy_price
                trades.append(
                    {
                        "Date": date,
                        "Action": "BUY",
                        "Ticker": p,
                        "CapitalPct": w * 100,
                        "TradePrice": buy_price,
                        "GainPctSinceBuy": np.nan,
                        "PositionSnapshot": "",
                    }
                )
                open_buy_rows[p] = len(trades) - 1

            current_positions = next_positions
            weights = next_weights

        daily_ret = 0.0
        stoploss_events = []
        for w, t in zip(weights, current_positions):
            asset_return = daily_returns.at[date, t]

            stop_event = None
            if (
                entry_dates.get(t) != date
                and t in ohlc_data
                and date in ohlc_data[t].index
            ):
                buy_price = entry_prices.get(t)
                day_open = ohlc_data[t].at[date, "Open"]
                day_low = ohlc_data[t].at[date, "Low"]
                prev_close = etf.at[data.index[i - 1], t] if i > 0 else np.nan
                stop_candidates = []

                if (
                    stoploss_pct is not None
                    and stoploss_applies_to_ticker(t, stoploss_scope)
                    and high_water_prices.get(t) not in (None, 0)
                    and pd.notna(high_water_prices.get(t))
                ):
                    stoploss_anchor = high_water_prices[t]
                    stop_candidates.append(
                        ("STOPLOSS_SELL", stoploss_anchor * (1 - abs(stoploss_pct) / 100))
                    )

                trailing_high = high_water_prices.get(t)
                if (
                    trailing_stop_pct is not None
                    and scope_applies_to_ticker(t, trailing_stop_scope)
                    and trailing_high not in (None, 0)
                    and pd.notna(trailing_high)
                ):
                    stop_candidates.append(
                        ("TRAILING_STOP_SELL", trailing_high * (1 - abs(trailing_stop_pct) / 100))
                    )

                valid_candidates = [
                    (action, stop_price)
                    for action, stop_price in stop_candidates
                    if pd.notna(stop_price)
                ]
                if valid_candidates and pd.notna(day_low):
                    triggered_candidates = [
                        (action, stop_price)
                        for action, stop_price in valid_candidates
                        if day_low <= stop_price
                    ]
                    if triggered_candidates:
                        action, stop_price = max(triggered_candidates, key=lambda item: item[1])
                        exit_price = stop_price
                        if pd.notna(day_open) and day_open < stop_price:
                            exit_price = day_open

                        if pd.notna(prev_close) and prev_close not in (None, 0):
                            asset_return = (exit_price / prev_close) - 1

                        gain_pct = calculate_gain_pct(exit_price, buy_price)
                        stop_event = {
                            "action": action,
                            "ticker": t,
                            "weight": w,
                            "exit_price": exit_price,
                            "gain_pct": gain_pct,
                        }

            if pd.notna(asset_return):
                daily_ret += w * asset_return

            if stop_event is not None:
                stoploss_events.append(stop_event)

        capital *= (1 + daily_ret)
        daily_pnl = capital - capital_at_day_start
        daily_return_pct = daily_ret * 100

        if stoploss_events:
            stopped_tickers = {event["ticker"] for event in stoploss_events}
            remaining_positions = []
            remaining_weights = []

            for t, w in zip(current_positions, weights):
                if t not in stopped_tickers:
                    remaining_positions.append(t)
                    remaining_weights.append(w)

            for event in stoploss_events:
                t = event["ticker"]
                trades.append(
                    {
                        "Date": date,
                        "Action": event["action"],
                        "Ticker": t,
                        "CapitalPct": np.nan,
                        "TradePrice": event["exit_price"],
                        "GainPctSinceBuy": event["gain_pct"],
                        "PositionSnapshot": "",
                    }
                )
                entry_prices.pop(t, None)
                entry_dates.pop(t, None)
                open_buy_rows.pop(t, None)
                high_water_prices.pop(t, None)

            current_positions = remaining_positions
            weights = remaining_weights
            if stoploss_after_exit == "redistribute" and weights:
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]

        append_position_row(
            trades,
            date,
            current_positions,
            weights,
            etf,
            entry_prices,
            portfolio_value=capital,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            cumulative_pnl=capital - INITIAL_CAPITAL,
        )

        for ticker in current_positions:
            close_price = etf.at[date, ticker]
            if pd.notna(close_price):
                high_water_prices[ticker] = max(high_water_prices.get(ticker, close_price), close_price)

        values.append(capital)
        dates.append(date)

    if len(data.index) > 0:
        last_date = data.index[-1]
        for ticker, trade_idx in open_buy_rows.items():
            buy_price = entry_prices.get(ticker)
            last_price = etf.at[last_date, ticker]
            if pd.notna(last_price) and buy_price not in (None, 0) and pd.notna(buy_price):
                trades[trade_idx]["GainPctSinceBuy"] = ((last_price / buy_price) - 1) * 100

    current_holdings = []
    for ticker, weight in zip(current_positions, weights):
        trade_idx = open_buy_rows.get(ticker)
        trade_row = trades[trade_idx] if trade_idx is not None else {}
        entry_date = entry_dates.get(ticker)
        current_holdings.append(
            {
                "Ticker": ticker,
                "CapitalPct": weight * 100,
                "EntryDate": entry_date.date() if pd.notna(entry_date) else pd.NaT,
                "EntryPrice": trade_row.get("TradePrice", np.nan),
                "GainPctSinceBuy": trade_row.get("GainPctSinceBuy", np.nan),
            }
        )

    final_state = {
        "as_of_date": data.index[-1].date() if len(data.index) > 0 else None,
        "regime": current_regime,
        "holdings": list(current_positions),
        "weights": list(weights),
        "holdings_table": pd.DataFrame(current_holdings),
    }

    return pd.Series(values, index=dates), pd.DataFrame(trades), final_state

# ----------------------------
# BENCHMARK
# ----------------------------
def spy_benchmark(data, index):
    spy = data["SPY"]
    returns = spy.pct_change().fillna(0)
    equity = (1 + returns).cumprod() * INITIAL_CAPITAL
    return equity.loc[index]

# ----------------------------
# PERFORMANCE
# ----------------------------
def calculate_cagr(series):
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1

def performance_metrics(series):
    returns = series.pct_change().dropna()
    std = returns.std()
    sharpe = np.nan
    if pd.notna(std) and std != 0:
        sharpe = (returns.mean() / std) * np.sqrt(252)

    return {
        "cagr": calculate_cagr(series),
        "sharpe": sharpe,
        "maxdd": (series / series.cummax() - 1).min(),
        "final_value": series.iloc[-1],
    }

def performance(name, series, spy):

    metrics = performance_metrics(series)

    print(f"\n==== {name} ====")
    print("CAGR:", round(metrics["cagr"]*100,2), "%")
    print("Sharpe:", round(metrics["sharpe"],2))
    print("MaxDD:", round(metrics["maxdd"]*100,2), "%")

def print_extrapolation(name, series):
    final_value = series.iloc[-1]
    multiple = final_value / INITIAL_CAPITAL

    print(
        f"{name}: ${INITIAL_CAPITAL:,.2f} -> ${final_value:,.2f} "
        f"({multiple:.2f}x)"
    )

def save_trade_logs(base_trades, enh_trades, base_state, enh_state):
    baseline_path = OUTPUT_DIR / "baseline_trades_log.csv"
    enhanced_path = OUTPUT_DIR / "enhanced_trades_log.csv"
    comparison_path = OUTPUT_DIR / "trades_comparison_log.csv"
    holdings_path = OUTPUT_DIR / "current_holdings_snapshot.csv"

    base_trades = add_product_descriptions(base_trades)
    enh_trades = add_product_descriptions(enh_trades)

    base_trades.to_csv(baseline_path, index=False)
    enh_trades.to_csv(enhanced_path, index=False)

    comparison = pd.concat(
        [
            base_trades.assign(Model="Baseline"),
            enh_trades.assign(Model="Enhanced"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(comparison_path, index=False)

    holdings_snapshot = pd.concat(
        [
            base_state["holdings_table"].assign(
                Model="Baseline",
                AsOfDate=base_state["as_of_date"],
                Regime=base_state["regime"],
            ),
            enh_state["holdings_table"].assign(
                Model="Enhanced",
                AsOfDate=enh_state["as_of_date"],
                Regime=enh_state["regime"],
            ),
        ],
        ignore_index=True,
    )
    if not holdings_snapshot.empty:
        holdings_snapshot = holdings_snapshot.assign(
            ProductDescription=holdings_snapshot["Ticker"].map(PRODUCT_DESCRIPTIONS).fillna("Unknown product")
        )
    holdings_snapshot.to_csv(holdings_path, index=False)

    print("\nSaved trade logs:")
    print(f"Baseline: {baseline_path}")
    print(f"Enhanced: {enhanced_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Current holdings snapshot: {holdings_path}")

# ----------------------------
# YEARLY
# ----------------------------
def yearly(series):
    return series.resample("Y").last().pct_change()

def yearly_capital(series):
    yearly_values = series.resample("Y").last()
    yearly_values.index = yearly_values.index.year
    return yearly_values

def print_stoploss_comparison(no_stop_eq, stop_eq, label, stoploss_pct):
    no_stop = performance_metrics(no_stop_eq)
    with_stop = performance_metrics(stop_eq)

    comparison = pd.DataFrame(
        {
            "NoStop": {
                "CAGR%": no_stop["cagr"] * 100,
                "Sharpe": no_stop["sharpe"],
                "MaxDD%": no_stop["maxdd"] * 100,
                "FinalValue": no_stop["final_value"],
            },
            f"Stop{stoploss_pct:g}%": {
                "CAGR%": with_stop["cagr"] * 100,
                "Sharpe": with_stop["sharpe"],
                "MaxDD%": with_stop["maxdd"] * 100,
                "FinalValue": with_stop["final_value"],
            },
        }
    )
    comparison["Delta"] = comparison.iloc[:, 1] - comparison.iloc[:, 0]

    print(f"\n{label} stop loss comparison ({stoploss_pct:g}%):")
    print(comparison.round(2).to_string())

def print_latest_score_breakdown(model_name, score, regime, feature_details):
    latest_date = score.dropna(how="all").index[-1]
    latest_regime = regime.loc[latest_date]
    ranked = score.loc[latest_date].dropna().sort_values(ascending=False)
    selected_positions, selected_weights = select_positions(ranked, latest_regime)

    tracked = []
    for ticker in selected_positions + ["XLE", "XLK", "SMH", "TQQQ"]:
        if ticker in ranked.index and ticker not in tracked:
            tracked.append(ticker)

    print(f"\nLatest {model_name} ranking breakdown on {latest_date.date()} ({latest_regime}):")
    print("Selected holdings:", ", ".join(f"{t}: {w*100:.0f}%" for t, w in zip(selected_positions, selected_weights)))
    print("Ranked scores:")
    for rank, (ticker, value) in enumerate(ranked.head(8).items(), start=1):
        print(f"  {rank}. {ticker}: {value:.4f}")

    if model_name == "BASELINE":
        detail_table = pd.DataFrame({
            "Score": feature_details["score"].loc[latest_date, tracked],
            "RS_3M": feature_details["rs_3m"].loc[latest_date, tracked],
            "Ret_1M": feature_details["ret_1m"].loc[latest_date, tracked],
            "Ret_6M": feature_details["ret_6m"].loc[latest_date, tracked],
            "Trend": feature_details["trend"].loc[latest_date, tracked],
            "Vol_20D": feature_details["vol"].loc[latest_date, tracked],
            "Z_RS_3M": feature_details["z_rs_3m"].loc[latest_date, tracked],
            "Z_Ret_1M": feature_details["z_ret_1m"].loc[latest_date, tracked],
            "Z_Ret_6M": feature_details["z_ret_6m"].loc[latest_date, tracked],
            "Z_Vol": feature_details["z_vol"].loc[latest_date, tracked],
        })
    else:
        detail_table = pd.DataFrame({
            "Score": feature_details["score"].loc[latest_date, tracked],
            "RS_3M": feature_details["rs_3m"].loc[latest_date, tracked],
            "ROC_1M": feature_details["roc_1m"].loc[latest_date, tracked],
            "Breakout": feature_details["breakout"].loc[latest_date, tracked],
            "VolSignal": feature_details["vol_signal"].loc[latest_date, tracked],
            "Trend": feature_details["trend"].loc[latest_date, tracked],
            "Z_RS_3M": feature_details["z_rs_3m"].loc[latest_date, tracked],
            "Z_ROC_1M": feature_details["z_roc_1m"].loc[latest_date, tracked],
            "Z_VolSignal": feature_details["z_vol_signal"].loc[latest_date, tracked],
        })

    detail_table["Rank"] = ranked.rank(ascending=False, method="min").reindex(tracked)
    detail_table = detail_table[["Rank"] + [c for c in detail_table.columns if c != "Rank"]]
    print(detail_table.round(4).to_string())

    if "TQQQ" in ranked.index:
        tqqq_rank = int(ranked.rank(ascending=False, method="min").loc["TQQQ"])
        if "TQQQ" not in selected_positions:
            print(f"TQQQ was not included because it ranked #{tqqq_rank}, outside the selected set for {latest_regime}.")
        else:
            print(f"TQQQ was included because it ranked #{tqqq_rank}.")

    if "NASA" in ranked.index:
        nasa_rank = int(ranked.rank(ascending=False, method="min").loc["NASA"])
        if "NASA" not in selected_positions:
            print(f"NASA was not included because it ranked #{nasa_rank}, outside the selected set for {latest_regime}.")
        else:
            print(f"NASA was included because it ranked #{nasa_rank}.")
    else:
        print("NASA is not in the ranked scores.")
        
    macro_details = feature_details.get("macro")
    if macro_details is not None:
        print("Macro overlay snapshot:")
        print(
            f"  Tech tilt: {macro_details['tech_tilt'].loc[latest_date]:.4f}, "
            f"Energy tilt: {macro_details['energy_tilt'].loc[latest_date]:.4f}, "
            f"Utilities tilt: {macro_details['utilities_tilt'].loc[latest_date]:.4f}, "
            f"Materials tilt: {macro_details['materials_tilt'].loc[latest_date]:.4f}"
        )
        print(
            f"  10Y yield 3M change: {macro_details['tnx_3m_change'].loc[latest_date]:.4f}, "
            f"WTI 3M momentum: {macro_details['oil_3m_mom'].loc[latest_date]:.4f}, "
            f"WTI 1M momentum: {macro_details['oil_1m_mom'].loc[latest_date]:.4f}, "
            f"VIX z-score: {macro_details['vix_z'].loc[latest_date]:.4f}"
        )

# ----------------------------
# LIBRARY ENTRYPOINTS
# ----------------------------
def run_backtest_suite(
    *,
    use_macro_overlay=False,
    stoploss_pct=None,
    stoploss_scope="all",
    stoploss_after_exit="cash",
    rebalance_frequency="monthly",
    trailing_stop_pct=None,
    trailing_stop_scope="all",
    enable_velocity_rotation=False,
    velocity_window=3,
    velocity_drop_pct=5.0,
    velocity_candidate_buffer_pct=0.0,
):
    data, ohlc_data = get_data()

    score_b, score_e, regime, feature_details = compute_features(
        data,
        use_macro_overlay=use_macro_overlay,
    )
    baseline_signal_date, baseline_signal_regime, baseline_signal_holdings = latest_signal_positions(score_b, regime)
    enhanced_signal_date, enhanced_signal_regime, enhanced_signal_holdings = latest_signal_positions(score_e, regime)

    print("\nRunning Baseline...")
    base_eq, base_trades, base_state = run_strategy(
        data,
        ohlc_data,
        score_b,
        regime,
        stoploss_pct=stoploss_pct,
        stoploss_scope=stoploss_scope,
        stoploss_after_exit=stoploss_after_exit,
        rebalance_frequency=rebalance_frequency,
        trailing_stop_pct=trailing_stop_pct,
        trailing_stop_scope=trailing_stop_scope,
        enable_velocity_rotation=enable_velocity_rotation,
        velocity_window=velocity_window,
        velocity_drop_pct=velocity_drop_pct,
        velocity_candidate_buffer_pct=velocity_candidate_buffer_pct,
    )

    print("\nRunning Enhanced...")
    enh_eq, enh_trades, enh_state = run_strategy(
        data,
        ohlc_data,
        score_e,
        regime,
        stoploss_pct=stoploss_pct,
        stoploss_scope=stoploss_scope,
        stoploss_after_exit=stoploss_after_exit,
        rebalance_frequency=rebalance_frequency,
        trailing_stop_pct=trailing_stop_pct,
        trailing_stop_scope=trailing_stop_scope,
        enable_velocity_rotation=enable_velocity_rotation,
        velocity_window=velocity_window,
        velocity_drop_pct=velocity_drop_pct,
        velocity_candidate_buffer_pct=velocity_candidate_buffer_pct,
    )

    spy_eq = spy_benchmark(data, base_eq.index)

    return BacktestRunResult(
        baseline_equity=base_eq,
        enhanced_equity=enh_eq,
        spy_equity=spy_eq,
        baseline_trades=base_trades,
        enhanced_trades=enh_trades,
        baseline_state=base_state,
        enhanced_state=enh_state,
        baseline_score=score_b,
        enhanced_score=score_e,
        regime=regime,
        feature_details=feature_details,
        data=data,
        ohlc_data=ohlc_data,
        baseline_signal_date=baseline_signal_date,
        baseline_signal_regime=baseline_signal_regime,
        baseline_signal_holdings=baseline_signal_holdings,
        enhanced_signal_date=enhanced_signal_date,
        enhanced_signal_regime=enhanced_signal_regime,
        enhanced_signal_holdings=enhanced_signal_holdings,
    )


def run_stoploss_comparison(compare_stoploss_pct, compare_stoploss_scope, run_result, args):
    print("\nRunning stop loss comparison variants...")
    base_eq_no_stop, _, _ = run_strategy(
        run_result.data,
        run_result.ohlc_data,
        run_result.baseline_score,
        run_result.regime,
        stoploss_pct=None,
    )
    base_eq_with_stop, _, _ = run_strategy(
        run_result.data,
        run_result.ohlc_data,
        run_result.baseline_score,
        run_result.regime,
        stoploss_pct=compare_stoploss_pct,
        stoploss_scope=compare_stoploss_scope,
        stoploss_after_exit=args.stoploss_after_exit,
        rebalance_frequency=args.rebalance_frequency,
        trailing_stop_pct=args.trailing_stop_pct,
        trailing_stop_scope=args.trailing_stop_scope,
        enable_velocity_rotation=args.enable_velocity_rotation,
        velocity_window=args.velocity_window,
        velocity_drop_pct=args.velocity_drop_pct,
        velocity_candidate_buffer_pct=args.velocity_candidate_buffer_pct,
    )
    enh_eq_no_stop, _, _ = run_strategy(
        run_result.data,
        run_result.ohlc_data,
        run_result.enhanced_score,
        run_result.regime,
        stoploss_pct=None,
    )
    enh_eq_with_stop, _, _ = run_strategy(
        run_result.data,
        run_result.ohlc_data,
        run_result.enhanced_score,
        run_result.regime,
        stoploss_pct=compare_stoploss_pct,
        stoploss_scope=compare_stoploss_scope,
        stoploss_after_exit=args.stoploss_after_exit,
        rebalance_frequency=args.rebalance_frequency,
        trailing_stop_pct=args.trailing_stop_pct,
        trailing_stop_scope=args.trailing_stop_scope,
        enable_velocity_rotation=args.enable_velocity_rotation,
        velocity_window=args.velocity_window,
        velocity_drop_pct=args.velocity_drop_pct,
        velocity_candidate_buffer_pct=args.velocity_candidate_buffer_pct,
    )

    print_stoploss_comparison(base_eq_no_stop, base_eq_with_stop, "Baseline", compare_stoploss_pct)
    print_stoploss_comparison(enh_eq_no_stop, enh_eq_with_stop, "Enhanced", compare_stoploss_pct)


def print_cli_summary(run_result, args):
    base_eq = run_result.baseline_equity
    enh_eq = run_result.enhanced_equity
    spy_eq = run_result.spy_equity

    performance("BASELINE", base_eq, spy_eq)
    performance("ENHANCED", enh_eq, spy_eq)
    performance("SPY", spy_eq, spy_eq)

    print(f"\n Growth of {INITIAL_CAPITAL=}:")
    print_extrapolation("Baseline", base_eq)
    print_extrapolation("Enhanced", enh_eq)
    print_extrapolation("SPY", spy_eq)

    print("\nTrade Count:")
    print("Baseline:", len(run_result.baseline_trades))
    print("Enhanced:", len(run_result.enhanced_trades))

    print("\nConfiguration:")
    print(f"Universe source: {UNIVERSE_CONFIG['source']}")
    print(f"Macro overlay enabled: {args.use_macro_overlay}")
    print(f"Stop loss enabled: {args.stoploss_pct is not None}")
    print(f"Stop loss pct: {args.stoploss_pct if args.stoploss_pct is not None else 'N/A'}")
    print(f"Stop loss scope: {args.stoploss_scope}")
    print(f"Stop loss after exit: {args.stoploss_after_exit}")
    print(f"Rebalance frequency: {args.rebalance_frequency}")
    print(f"Trailing stop pct: {args.trailing_stop_pct if args.trailing_stop_pct is not None else 'N/A'}")
    print(f"Trailing stop scope: {args.trailing_stop_scope}")
    print(f"Velocity rotation enabled: {args.enable_velocity_rotation}")
    print(f"Velocity window: {args.velocity_window}")
    print(f"Velocity drop pct: {args.velocity_drop_pct}")
    print(f"Velocity candidate buffer pct: {args.velocity_candidate_buffer_pct}")
    print(f"Stop loss comparison mode: {args.compare_stoploss_pct is not None}")
    print("Run with macro overlay: /opt/anaconda3/bin/python sector_rotation_backtest6.py --use-macro-overlay")
    print("Run with trailing stop: /opt/anaconda3/bin/python sector_rotation_backtest6.py --trailing-stop-pct 8 --trailing-stop-scope all")
    print("Run with velocity rotation: /opt/anaconda3/bin/python sector_rotation_backtest6.py --enable-velocity-rotation --velocity-window 3 --velocity-drop-pct 5")
    print("Run with both new controls: /opt/anaconda3/bin/python sector_rotation_backtest6.py --trailing-stop-pct 8 --trailing-stop-scope all --enable-velocity-rotation --velocity-window 3 --velocity-drop-pct 5")
    print("Run with leveraged-only hard stop plus trailing stop: /opt/anaconda3/bin/python sector_rotation_backtest6.py --stoploss-pct 5 --stoploss-scope leveraged --trailing-stop-pct 8 --trailing-stop-scope all")
    print("Run with daily rebalance and new controls: /opt/anaconda3/bin/python sector_rotation_backtest6.py --rebalance-frequency daily --trailing-stop-pct 8 --enable-velocity-rotation --velocity-window 3 --velocity-drop-pct 5")
    print("Compare no stop vs stop loss: /opt/anaconda3/bin/python sector_rotation_backtest6.py --compare-stoploss-pct 5")
    print("Compare no stop vs leveraged-only stop loss: /opt/anaconda3/bin/python sector_rotation_backtest6.py --compare-stoploss-pct 5 --compare-stoploss-scope leveraged")

    print("\nData Freshness Check:")
    print(f"Latest dataset date: {run_result.data.index.max().date()}")
    print(f"Latest SPY date: {latest_available_date(run_result.data['SPY'])}, Latest ^VIX date: {latest_available_date(run_result.data['^VIX'])}")
    print(f"Baseline latest signal date: {run_result.baseline_signal_date}, Signal regime: {run_result.baseline_signal_regime}, Signal picks: {run_result.baseline_signal_holdings}")
    print(f"Baseline executed holdings as of {run_result.baseline_state['as_of_date']}: regime {run_result.baseline_state['regime']}, holdings {run_result.baseline_state['holdings']}")
    print(f"Enhanced latest signal date: {run_result.enhanced_signal_date}, Signal regime: {run_result.enhanced_signal_regime}, Signal picks: {run_result.enhanced_signal_holdings}")
    print(f"Enhanced executed holdings as of {run_result.enhanced_state['as_of_date']}: regime {run_result.enhanced_state['regime']}, holdings {run_result.enhanced_state['holdings']}")
    if args.log_ranked_scores:
        print_latest_score_breakdown("BASELINE", run_result.baseline_score, run_result.regime, run_result.feature_details["baseline"])
        print_latest_score_breakdown("ENHANCED", run_result.enhanced_score, run_result.regime, run_result.feature_details["enhanced"])

    if args.compare_stoploss_pct is not None:
        run_stoploss_comparison(args.compare_stoploss_pct, args.compare_stoploss_scope, run_result, args)

    save_trade_logs(
        run_result.baseline_trades,
        run_result.enhanced_trades,
        run_result.baseline_state,
        run_result.enhanced_state,
    )

    print("\nYearly Capital:")
    print(pd.concat({
        "Baseline": yearly_capital(base_eq),
        "Enhanced": yearly_capital(enh_eq),
        "SPY": yearly_capital(spy_eq)
    }, axis=1).round(2))

    print("\nYearly Returns:")
    print(pd.concat({
        "Baseline": yearly(base_eq),
        "Enhanced": yearly(enh_eq),
        "SPY": yearly(spy_eq)
    }, axis=1))


def main(argv=None):
    args = parse_args(argv)
    run_result = run_backtest_suite(
        use_macro_overlay=args.use_macro_overlay,
        stoploss_pct=args.stoploss_pct,
        stoploss_scope=args.stoploss_scope,
        stoploss_after_exit=args.stoploss_after_exit,
        rebalance_frequency=args.rebalance_frequency,
        trailing_stop_pct=args.trailing_stop_pct,
        trailing_stop_scope=args.trailing_stop_scope,
        enable_velocity_rotation=args.enable_velocity_rotation,
        velocity_window=args.velocity_window,
        velocity_drop_pct=args.velocity_drop_pct,
        velocity_candidate_buffer_pct=args.velocity_candidate_buffer_pct,
    )
    print_cli_summary(run_result, args)
    return run_result


# ----------------------------
# MAIN / LIBRARY SMOKE TEST
# ----------------------------
if __name__ == "__main__":
    main()
