import numpy as np
from pathlib import Path

import library.sector_rotation_lib as base


OUTPUT_DIR = Path(__file__).resolve().parent


def compute_active_stop_levels(
    date,
    current_positions,
    entry_dates,
    high_water_prices,
    stoploss_pct,
    stoploss_scope,
    trailing_stop_pct,
    trailing_stop_scope,
):
    levels = {}

    for ticker in current_positions:
        if entry_dates.get(ticker) == date:
            continue

        candidates = []
        high_water_price = high_water_prices.get(ticker)

        if (
            stoploss_pct is not None
            and base.stoploss_applies_to_ticker(ticker, stoploss_scope)
            and high_water_price not in (None, 0)
            and not np.isnan(high_water_price)
        ):
            candidates.append(high_water_price * (1 - abs(stoploss_pct) / 100))

        if (
            trailing_stop_pct is not None
            and base.scope_applies_to_ticker(ticker, trailing_stop_scope)
            and high_water_price not in (None, 0)
            and not np.isnan(high_water_price)
        ):
            candidates.append(high_water_price * (1 - abs(trailing_stop_pct) / 100))

        valid_candidates = [price for price in candidates if not np.isnan(price)]
        if valid_candidates:
            levels[ticker] = max(valid_candidates)

    return levels


def compute_stoploss_update_line(
    date,
    current_positions,
    entry_dates,
    entry_prices,
    high_water_prices,
    ohlc_data,
    stoploss_pct,
    stoploss_scope,
    trailing_stop_pct,
    trailing_stop_scope,
):
    if not current_positions:
        return f"{date.date()} : CASH"

    pairs = []
    for ticker in current_positions:
        current_anchor = high_water_prices.get(ticker)
        day_high = np.nan
        if ticker in ohlc_data and date in ohlc_data[ticker].index:
            day_high = ohlc_data[ticker].at[date, "High"]

        candidate_anchor = current_anchor
        if not np.isnan(day_high):
            if candidate_anchor in (None, 0) or np.isnan(candidate_anchor):
                candidate_anchor = day_high
            else:
                candidate_anchor = max(candidate_anchor, day_high)

        stop_candidates = []
        if (
            stoploss_pct is not None
            and base.stoploss_applies_to_ticker(ticker, stoploss_scope)
            and candidate_anchor not in (None, 0)
            and not np.isnan(candidate_anchor)
        ):
            stop_candidates.append(candidate_anchor * (1 - abs(stoploss_pct) / 100))

        if (
            trailing_stop_pct is not None
            and base.scope_applies_to_ticker(ticker, trailing_stop_scope)
            and candidate_anchor not in (None, 0)
            and not np.isnan(candidate_anchor)
        ):
            stop_candidates.append(candidate_anchor * (1 - abs(trailing_stop_pct) / 100))

        active_stop = max(stop_candidates) if stop_candidates else np.nan
        is_new_position = entry_dates.get(ticker) == date
        has_new_high = (
            candidate_anchor not in (None, 0)
            and not np.isnan(candidate_anchor)
            and (
                current_anchor in (None, 0)
                or np.isnan(current_anchor)
                or candidate_anchor > current_anchor
            )
        )

        if (is_new_position or has_new_high) and not np.isnan(active_stop):
            pairs.append(f"<{ticker},{base.format_price(active_stop)}>")
        else:
            pairs.append(f"<{ticker},NA>")

    return f"{date.date()} : {','.join(pairs)}"


def save_daily_stoploss_logs(baseline_lines, enhanced_lines):
    baseline_path = OUTPUT_DIR / "baseline_daily_stoploss_levels.txt"
    enhanced_path = OUTPUT_DIR / "enhanced_daily_stoploss_levels.txt"

    baseline_path.write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")
    enhanced_path.write_text("\n".join(enhanced_lines) + "\n", encoding="utf-8")

    print("\nSaved daily stop-loss logs:")
    print(f"Baseline: {baseline_path}")
    print(f"Enhanced: {enhanced_path}")
    print("Each line shows stop-price updates from new highs; unchanged symbols are marked as NA.")
    print("The stop anchor is the highest traded High since entry.")
    print("If the market opens below an active stop level, the strategy exits at the opening price.")


def run_strategy_with_daily_stoploss(
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
    capital = base.INITIAL_CAPITAL
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
    stoploss_lines = []

    etf = data.reindex(columns=base.tickers)
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
            and base.should_trigger_velocity_rebalance(
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
            next_positions, next_weights = base.select_positions(ranked, current_regime)

            current_set = set(current_positions)
            next_set = set(next_positions)

            kept_positions = current_set.intersection(next_set)
            to_sell = [position for position in current_positions if position not in kept_positions]
            to_buy = [
                (position, weight)
                for position, weight in zip(next_positions, next_weights)
                if position not in kept_positions
            ]

            for position in to_sell:
                sell_price = etf.at[date, position]
                buy_price = entry_prices.get(position)
                gain_pct = np.nan

                if (
                    not np.isnan(sell_price)
                    and buy_price not in (None, 0)
                    and not np.isnan(buy_price)
                ):
                    gain_pct = ((sell_price / buy_price) - 1) * 100

                trades.append(
                    {
                        "Date": date,
                        "Action": "SELL",
                        "Ticker": position,
                        "CapitalPct": np.nan,
                        "TradePrice": sell_price,
                        "GainPctSinceBuy": gain_pct,
                        "PositionSnapshot": "",
                    }
                )
                entry_prices.pop(position, None)
                entry_dates.pop(position, None)
                open_buy_rows.pop(position, None)
                high_water_prices.pop(position, None)

            for position, weight in to_buy:
                buy_price = etf.at[date, position]
                entry_prices[position] = buy_price
                entry_dates[position] = date
                high_water_prices[position] = buy_price
                trades.append(
                    {
                        "Date": date,
                        "Action": "BUY",
                        "Ticker": position,
                        "CapitalPct": weight * 100,
                        "TradePrice": buy_price,
                        "GainPctSinceBuy": np.nan,
                        "PositionSnapshot": "",
                    }
                )
                open_buy_rows[position] = len(trades) - 1

            current_positions = next_positions
            weights = next_weights

        daily_ret = 0.0
        stoploss_events = []
        for weight, ticker in zip(weights, current_positions):
            asset_return = daily_returns.at[date, ticker]

            stop_event = None
            if (
                entry_dates.get(ticker) != date
                and ticker in ohlc_data
                and date in ohlc_data[ticker].index
            ):
                buy_price = entry_prices.get(ticker)
                day_open = ohlc_data[ticker].at[date, "Open"]
                day_low = ohlc_data[ticker].at[date, "Low"]
                prev_close = etf.at[data.index[i - 1], ticker] if i > 0 else np.nan
                stop_candidates = []

                if (
                    stoploss_pct is not None
                    and base.stoploss_applies_to_ticker(ticker, stoploss_scope)
                    and high_water_prices.get(ticker) not in (None, 0)
                    and not np.isnan(high_water_prices.get(ticker))
                ):
                    stoploss_anchor = high_water_prices[ticker]
                    stop_candidates.append(
                        ("STOPLOSS_SELL", stoploss_anchor * (1 - abs(stoploss_pct) / 100))
                    )

                trailing_high = high_water_prices.get(ticker)
                if (
                    trailing_stop_pct is not None
                    and base.scope_applies_to_ticker(ticker, trailing_stop_scope)
                    and trailing_high not in (None, 0)
                    and not np.isnan(trailing_high)
                ):
                    stop_candidates.append(
                        ("TRAILING_STOP_SELL", trailing_high * (1 - abs(trailing_stop_pct) / 100))
                    )

                valid_candidates = [
                    (action, stop_price)
                    for action, stop_price in stop_candidates
                    if not np.isnan(stop_price)
                ]
                if valid_candidates and not np.isnan(day_low):
                    triggered_candidates = [
                        (action, stop_price)
                        for action, stop_price in valid_candidates
                        if day_low <= stop_price
                    ]
                    if triggered_candidates:
                        action, stop_price = max(triggered_candidates, key=lambda item: item[1])
                        exit_price = stop_price
                        if not np.isnan(day_open) and day_open < stop_price:
                            exit_price = day_open

                        if not np.isnan(prev_close) and prev_close not in (None, 0):
                            asset_return = (exit_price / prev_close) - 1

                        gain_pct = base.calculate_gain_pct(exit_price, buy_price)
                        stop_event = {
                            "action": action,
                            "ticker": ticker,
                            "weight": weight,
                            "exit_price": exit_price,
                            "gain_pct": gain_pct,
                        }

            if not np.isnan(asset_return):
                daily_ret += weight * asset_return

            if stop_event is not None:
                stoploss_events.append(stop_event)

        capital *= (1 + daily_ret)
        daily_pnl = capital - capital_at_day_start
        daily_return_pct = daily_ret * 100

        if stoploss_events:
            stopped_tickers = {event["ticker"] for event in stoploss_events}
            remaining_positions = []
            remaining_weights = []

            for ticker, weight in zip(current_positions, weights):
                if ticker not in stopped_tickers:
                    remaining_positions.append(ticker)
                    remaining_weights.append(weight)

            for event in stoploss_events:
                ticker = event["ticker"]
                trades.append(
                    {
                        "Date": date,
                        "Action": event["action"],
                        "Ticker": ticker,
                        "CapitalPct": np.nan,
                        "TradePrice": event["exit_price"],
                        "GainPctSinceBuy": event["gain_pct"],
                        "PositionSnapshot": "",
                    }
                )
                entry_prices.pop(ticker, None)
                entry_dates.pop(ticker, None)
                open_buy_rows.pop(ticker, None)
                high_water_prices.pop(ticker, None)

            current_positions = remaining_positions
            weights = remaining_weights
            if stoploss_after_exit == "redistribute" and weights:
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [weight / total_weight for weight in weights]

        base.append_position_row(
            trades,
            date,
            current_positions,
            weights,
            etf,
            entry_prices,
            portfolio_value=capital,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            cumulative_pnl=capital - base.INITIAL_CAPITAL,
        )

        stoploss_lines.append(
            compute_stoploss_update_line(
                date,
                current_positions,
                entry_dates,
                entry_prices,
                high_water_prices,
                ohlc_data,
                stoploss_pct,
                stoploss_scope,
                trailing_stop_pct,
                trailing_stop_scope,
            )
        )

        for ticker in current_positions:
            day_high = np.nan
            if ticker in ohlc_data and date in ohlc_data[ticker].index:
                day_high = ohlc_data[ticker].at[date, "High"]

            if not np.isnan(day_high):
                prior_anchor = high_water_prices.get(ticker)
                if prior_anchor in (None, 0) or np.isnan(prior_anchor):
                    high_water_prices[ticker] = day_high
                else:
                    high_water_prices[ticker] = max(prior_anchor, day_high)
            else:
                close_price = etf.at[date, ticker]
                if not np.isnan(close_price):
                    prior_anchor = high_water_prices.get(ticker)
                    if prior_anchor in (None, 0) or np.isnan(prior_anchor):
                        high_water_prices[ticker] = close_price
                    else:
                        high_water_prices[ticker] = max(prior_anchor, close_price)

        values.append(capital)
        dates.append(date)

    if len(data.index) > 0:
        last_date = data.index[-1]
        for ticker, trade_idx in open_buy_rows.items():
            buy_price = entry_prices.get(ticker)
            last_price = etf.at[last_date, ticker]
            if (
                not np.isnan(last_price)
                and buy_price not in (None, 0)
                and not np.isnan(buy_price)
            ):
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
                "EntryDate": entry_date.date() if entry_date is not None else np.nan,
                "EntryPrice": trade_row.get("TradePrice", np.nan),
                "GainPctSinceBuy": trade_row.get("GainPctSinceBuy", np.nan),
            }
        )

    final_state = {
        "as_of_date": data.index[-1].date() if len(data.index) > 0 else None,
        "regime": current_regime,
        "holdings": list(current_positions),
        "weights": list(weights),
        "holdings_table": base.pd.DataFrame(current_holdings),
    }

    return base.pd.Series(values, index=dates), base.pd.DataFrame(trades), final_state, stoploss_lines


if __name__ == "__main__":
    args = base.parse_args()

    data, ohlc_data = base.get_data()

    score_b, score_e, regime, feature_details = base.compute_features(
        data,
        use_macro_overlay=args.use_macro_overlay,
    )
    baseline_signal_date, baseline_signal_regime, baseline_signal_holdings = base.latest_signal_positions(score_b, regime)
    enhanced_signal_date, enhanced_signal_regime, enhanced_signal_holdings = base.latest_signal_positions(score_e, regime)

    active_stoploss_pct = args.stoploss_pct
    active_stoploss_scope = args.stoploss_scope
    active_stoploss_after_exit = args.stoploss_after_exit
    active_rebalance_frequency = args.rebalance_frequency
    active_trailing_stop_pct = args.trailing_stop_pct
    active_trailing_stop_scope = args.trailing_stop_scope
    active_enable_velocity_rotation = args.enable_velocity_rotation
    active_velocity_window = args.velocity_window
    active_velocity_drop_pct = args.velocity_drop_pct
    active_velocity_candidate_buffer_pct = args.velocity_candidate_buffer_pct
    compare_stoploss_pct = args.compare_stoploss_pct
    compare_stoploss_scope = args.compare_stoploss_scope

    print("\nRunning Baseline...")
    base_eq, base_trades, base_state, base_stoploss_lines = run_strategy_with_daily_stoploss(
        data,
        ohlc_data,
        score_b,
        regime,
        stoploss_pct=active_stoploss_pct,
        stoploss_scope=active_stoploss_scope,
        stoploss_after_exit=active_stoploss_after_exit,
        rebalance_frequency=active_rebalance_frequency,
        trailing_stop_pct=active_trailing_stop_pct,
        trailing_stop_scope=active_trailing_stop_scope,
        enable_velocity_rotation=active_enable_velocity_rotation,
        velocity_window=active_velocity_window,
        velocity_drop_pct=active_velocity_drop_pct,
        velocity_candidate_buffer_pct=active_velocity_candidate_buffer_pct,
    )

    print("\nRunning Enhanced...")
    enh_eq, enh_trades, enh_state, enh_stoploss_lines = run_strategy_with_daily_stoploss(
        data,
        ohlc_data,
        score_e,
        regime,
        stoploss_pct=active_stoploss_pct,
        stoploss_scope=active_stoploss_scope,
        stoploss_after_exit=active_stoploss_after_exit,
        rebalance_frequency=active_rebalance_frequency,
        trailing_stop_pct=active_trailing_stop_pct,
        trailing_stop_scope=active_trailing_stop_scope,
        enable_velocity_rotation=active_enable_velocity_rotation,
        velocity_window=active_velocity_window,
        velocity_drop_pct=active_velocity_drop_pct,
        velocity_candidate_buffer_pct=active_velocity_candidate_buffer_pct,
    )

    spy_eq = base.spy_benchmark(data, base_eq.index)

    base.performance("BASELINE", base_eq, spy_eq)
    base.performance("ENHANCED", enh_eq, spy_eq)
    base.performance("SPY", spy_eq, spy_eq)

    print(f"\n Growth of {base.INITIAL_CAPITAL=}:")
    base.print_extrapolation("Baseline", base_eq)
    base.print_extrapolation("Enhanced", enh_eq)
    base.print_extrapolation("SPY", spy_eq)

    print("\nTrade Count:")
    print("Baseline:", len(base_trades))
    print("Enhanced:", len(enh_trades))

    print("\nConfiguration:")
    print(f"Macro overlay enabled: {args.use_macro_overlay}")
    print(f"Stop loss enabled: {active_stoploss_pct is not None}")
    print(f"Stop loss pct: {active_stoploss_pct if active_stoploss_pct is not None else 'N/A'}")
    print(f"Stop loss scope: {active_stoploss_scope}")
    print(f"Stop loss after exit: {active_stoploss_after_exit}")
    print(f"Rebalance frequency: {active_rebalance_frequency}")
    print(f"Trailing stop pct: {active_trailing_stop_pct if active_trailing_stop_pct is not None else 'N/A'}")
    print(f"Trailing stop scope: {active_trailing_stop_scope}")
    print(f"Velocity rotation enabled: {active_enable_velocity_rotation}")
    print(f"Velocity window: {active_velocity_window}")
    print(f"Velocity drop pct: {active_velocity_drop_pct}")
    print(f"Velocity candidate buffer pct: {active_velocity_candidate_buffer_pct}")
    print(f"Stop loss comparison mode: {compare_stoploss_pct is not None}")
    print("Run with macro overlay: /opt/anaconda3/bin/python sector_rotation_backtest_with_dl_sl.py --use-macro-overlay")
    print("Run with trailing stop: /opt/anaconda3/bin/python sector_rotation_backtest_with_dl_sl.py --trailing-stop-pct 8 --trailing-stop-scope all")
    print("Run with velocity rotation: /opt/anaconda3/bin/python sector_rotation_backtest_with_dl_sl.py --enable-velocity-rotation --velocity-window 3 --velocity-drop-pct 5")
    print("Run with daily stop-loss file: /opt/anaconda3/bin/python sector_rotation_backtest_with_dl_sl.py --stoploss-pct 5 --stoploss-scope all --rebalance-frequency daily --stoploss-after-exit redistribute")

    print("\nData Freshness Check:")
    print(f"Latest dataset date: {data.index.max().date()}")
    print(f"Latest SPY date: {base.latest_available_date(data['SPY'])}, Latest ^VIX date: {base.latest_available_date(data['^VIX'])}")
    print(f"Baseline latest signal date: {baseline_signal_date}, Signal regime: {baseline_signal_regime}, Signal picks: {baseline_signal_holdings}")
    print(f"Baseline executed holdings as of {base_state['as_of_date']}: regime {base_state['regime']}, holdings {base_state['holdings']}")
    print(f"Enhanced latest signal date: {enhanced_signal_date}, Signal regime: {enhanced_signal_regime}, Signal picks: {enhanced_signal_holdings}")
    print(f"Enhanced executed holdings as of {enh_state['as_of_date']}: regime {enh_state['regime']}, holdings {enh_state['holdings']}")
    if args.log_ranked_scores:
        base.print_latest_score_breakdown("BASELINE", score_b, regime, feature_details["baseline"])
        base.print_latest_score_breakdown("ENHANCED", score_e, regime, feature_details["enhanced"])

    if compare_stoploss_pct is not None:
        print("\nRunning stop loss comparison variants...")
        base_eq_no_stop, _, _ = base.run_strategy(data, ohlc_data, score_b, regime, stoploss_pct=None)
        base_eq_with_stop, _, _ = base.run_strategy(
            data,
            ohlc_data,
            score_b,
            regime,
            stoploss_pct=compare_stoploss_pct,
            stoploss_scope=compare_stoploss_scope,
            stoploss_after_exit=active_stoploss_after_exit,
            rebalance_frequency=active_rebalance_frequency,
            trailing_stop_pct=active_trailing_stop_pct,
            trailing_stop_scope=active_trailing_stop_scope,
            enable_velocity_rotation=active_enable_velocity_rotation,
            velocity_window=active_velocity_window,
            velocity_drop_pct=active_velocity_drop_pct,
            velocity_candidate_buffer_pct=active_velocity_candidate_buffer_pct,
        )
        enh_eq_no_stop, _, _ = base.run_strategy(data, ohlc_data, score_e, regime, stoploss_pct=None)
        enh_eq_with_stop, _, _ = base.run_strategy(
            data,
            ohlc_data,
            score_e,
            regime,
            stoploss_pct=compare_stoploss_pct,
            stoploss_scope=compare_stoploss_scope,
            stoploss_after_exit=active_stoploss_after_exit,
            rebalance_frequency=active_rebalance_frequency,
            trailing_stop_pct=active_trailing_stop_pct,
            trailing_stop_scope=active_trailing_stop_scope,
            enable_velocity_rotation=active_enable_velocity_rotation,
            velocity_window=active_velocity_window,
            velocity_drop_pct=active_velocity_drop_pct,
            velocity_candidate_buffer_pct=active_velocity_candidate_buffer_pct,
        )

        base.print_stoploss_comparison(base_eq_no_stop, base_eq_with_stop, "Baseline", compare_stoploss_pct)
        base.print_stoploss_comparison(enh_eq_no_stop, enh_eq_with_stop, "Enhanced", compare_stoploss_pct)

    base.save_trade_logs(base_trades, enh_trades, base_state, enh_state)
    save_daily_stoploss_logs(base_stoploss_lines, enh_stoploss_lines)

    print("\nYearly Capital:")
    print(base.pd.concat({
        "Baseline": base.yearly_capital(base_eq),
        "Enhanced": base.yearly_capital(enh_eq),
        "SPY": base.yearly_capital(spy_eq)
    }, axis=1).round(2))

    print("\nYearly Returns:")
    print(base.pd.concat({
        "Baseline": base.yearly(base_eq),
        "Enhanced": base.yearly(enh_eq),
        "SPY": base.yearly(spy_eq)
    }, axis=1))
