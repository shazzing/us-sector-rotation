import argparse
import csv
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

import library.sector_rotation_lib as base


CASH_LIKE_SYMBOLS = {"FZDXX"}
DEFAULT_STOP_CONFIG_PATH = Path(__file__).resolve().parent / "symbol_stop_config.csv"
CANONICAL_COLUMN_ALIASES = {
    "symbol": ["symbol", "ticker"],
    "quantity": ["quantity", "qty", "shares", "units"],
    "last_price": ["last", "last price", "last_price", "price", "current price", "current_price", "market price"],
    "average_price": ["$ avg cost", "avg cost", "average cost", "average_price", "average price", "avgprice", "avg_price"],
    "trade_date": ["entrydate", "entry date", "purchase date", "acquired date", "trade date", "trade_date", "buy date"],
    "profit_dollar": ["$ total g/l", "total g/l", "total_gl", "profit", "unrealized p/l", "unrealized pl", "pnl"],
    "profit_pct": ["% total g/l", "total g/l %", "profit %", "profit_pct", "unrealized return", "pnl %"],
    "value": ["value", "market value", "market_value", "position value"],
    "basis": ["basis", "cost basis", "cost_basis"],
    "weight_pct": ["% of acct", "%ofacct", "allocation", "weight", "portfolio pct"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Review an allocation file for stoploss and momentum.")
    parser.add_argument(
        "--allocation-path",
        required=True,
        help="Path to the asset allocation spreadsheet.",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Optional worksheet name for Excel files. Ignored for CSV files.",
    )
    parser.add_argument(
        "--stop-config-path",
        default=str(DEFAULT_STOP_CONFIG_PATH),
        help="Path to per-symbol stop configuration CSV.",
    )
    parser.add_argument(
        "--stoploss-pct",
        type=float,
        default=5.0,
        help="Fallback stop loss percentage for peak-stop symbols missing an explicit config value.",
    )
    parser.add_argument(
        "--use-macro-overlay",
        action="store_true",
        default=True,
        help="Use the macro overlay when ranking momentum candidates.",
    )
    return parser.parse_args()


def clean_numeric(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "--"}:
        return np.nan
    text = text.replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_matching_column(columns, aliases):
    normalized_map = {normalize_column_name(column): column for column in columns}
    for alias in aliases:
        match = normalized_map.get(normalize_column_name(alias))
        if match is not None:
            return match
    return None


def detect_asof_date(workbook_path, sheet_name):
    wb = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
    ws = wb[sheet_name]
    title_value = ws["A1"].value
    wb.close()

    if not title_value:
        return None

    match = re.search(r"as of (\d{2}/\d{2}/\d{4})", str(title_value), flags=re.IGNORECASE)
    if not match:
        return None

    return pd.to_datetime(match.group(1), format="%m/%d/%Y").date()


def detect_asof_date_from_text(values):
    for value in values:
        if value is None:
            continue
        match = re.search(r"as of (\d{2}/\d{2}/\d{4})", str(value), flags=re.IGNORECASE)
        if match:
            return pd.to_datetime(match.group(1), format="%m/%d/%Y").date()
    return None


def detect_header_row(raw_df):
    for idx in range(min(len(raw_df), 10)):
        row_values = [
            normalize_column_name(value)
            for value in raw_df.iloc[idx].tolist()
            if pd.notna(value)
        ]
        if any(value in {"symbol", "ticker"} for value in row_values):
            return idx
    return 0


def load_raw_allocation(allocation_path, sheet_name=None):
    suffix = allocation_path.suffix.lower()

    if suffix == ".csv":
        with allocation_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        max_columns = max((len(row) for row in rows), default=0)
        padded_rows = [row + [None] * (max_columns - len(row)) for row in rows]
        raw_df = pd.DataFrame(padded_rows)
        selected_sheet = "CSV"
        asof_date = detect_asof_date_from_text(raw_df.head(10).fillna("").to_numpy().flatten().tolist())
        return raw_df, selected_sheet, asof_date

    if suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else None
        xls = pd.ExcelFile(allocation_path, engine=engine)
        selected_sheet = sheet_name or xls.sheet_names[0]
        asof_date = detect_asof_date(allocation_path, selected_sheet)
        raw_df = pd.read_excel(
            allocation_path,
            sheet_name=selected_sheet,
            header=None,
            dtype=str,
            engine=engine,
        )
        return raw_df, selected_sheet, asof_date

    raise ValueError(f"Unsupported allocation file extension: {suffix}")


def load_allocation(allocation_path, sheet_name=None):
    raw_df, selected_sheet, asof_date = load_raw_allocation(allocation_path, sheet_name)
    header_row = detect_header_row(raw_df)
    header_values = raw_df.iloc[header_row].fillna("").tolist()
    df = raw_df.iloc[header_row + 1 :].copy()
    df.columns = [str(value).strip() if value is not None else "" for value in header_values]
    df = df.rename(columns=lambda col: str(col).strip() if col is not None else col)
    df = df.dropna(how="all")
    canonical_columns = {
        field: find_matching_column(df.columns, aliases)
        for field, aliases in CANONICAL_COLUMN_ALIASES.items()
    }
    symbol_column = canonical_columns["symbol"]
    if symbol_column is None:
        raise ValueError("Could not find a symbol/ticker column in the allocation file.")

    df["Symbol"] = df[symbol_column].astype(str).str.strip()
    df = df[df["Symbol"].notna() & df["Symbol"].ne("") & df["Symbol"].ne("nan")]

    numeric_field_map = {
        "Quantity": canonical_columns["quantity"],
        "Last": canonical_columns["last_price"],
        "AveragePrice": canonical_columns["average_price"],
        "ProfitDollar": canonical_columns["profit_dollar"],
        "ProfitPct": canonical_columns["profit_pct"],
        "Value": canonical_columns["value"],
        "Basis": canonical_columns["basis"],
        "WeightPct": canonical_columns["weight_pct"],
    }
    for output_column, source_column in numeric_field_map.items():
        if source_column is not None:
            df[output_column] = df[source_column].map(clean_numeric)
        else:
            df[output_column] = np.nan

    position_like_mask = (
        df["Symbol"].str.match(r"^[A-Za-z][A-Za-z0-9.\-() ]{0,20}$", na=False)
        & df["Quantity"].notna()
        & (df["Last"].notna() | df["AveragePrice"].notna() | df["Value"].notna())
    )
    df = df[position_like_mask].copy()

    trade_date_column = canonical_columns["trade_date"]
    if trade_date_column:
        df["EffectiveEntryDate"] = pd.to_datetime(df[trade_date_column], errors="coerce").dt.date
        df["EntryDateSource"] = trade_date_column
    else:
        df["EffectiveEntryDate"] = asof_date
        df["EntryDateSource"] = "AsOfDateFallback"

    df["IsCashLike"] = df["Symbol"].str.contains("cash", case=False, na=False) | df["Symbol"].isin(CASH_LIKE_SYMBOLS)
    return df, selected_sheet, asof_date


def build_default_stop_config_rows():
    broad_index_tickers = {"SPY", "VOO"}
    sector_tickers = {"XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLC", "VOX", "VGT"}
    thematic_tickers = {"REMX", "NLR", "ITB", "LIT", "QTUM", "NASA", "DRAM"}
    semiconductor_tickers = {"SMH"}
    high_vol_leveraged_tickers = {"TQQQ"}
    ultra_vol_leveraged_tickers = {"SOXL", "AGQ", "UGL"}

    rows = []
    for symbol in sorted(set(base.tickers)):
        if symbol in broad_index_tickers:
            stop_pct = 4.0
        elif symbol in sector_tickers:
            stop_pct = 5.0
        elif symbol in semiconductor_tickers:
            stop_pct = 7.0
        elif symbol in thematic_tickers:
            stop_pct = 8.0
        elif symbol in high_vol_leveraged_tickers:
            stop_pct = 10.0
        elif symbol in ultra_vol_leveraged_tickers:
            stop_pct = 12.0
        else:
            stop_pct = 6.0

        rows.append(
            {
                "Symbol": symbol,
                "ProductDescription": base.PRODUCT_DESCRIPTIONS.get(symbol, symbol),
                "StopMode": "peak_stop",
                "StopPct": stop_pct,
                "BrokerTrailPct": np.nan,
                "Notes": "",
            }
        )
    return pd.DataFrame(rows)


def ensure_stop_config_exists(stop_config_path):
    if stop_config_path.exists():
        return
    default_df = build_default_stop_config_rows()
    default_df.to_csv(stop_config_path, index=False)


def load_stop_config(stop_config_path, fallback_stoploss_pct):
    ensure_stop_config_exists(stop_config_path)
    config_df = pd.read_csv(stop_config_path)
    if config_df.empty:
        config_df = pd.DataFrame(columns=["Symbol", "StopMode", "StopPct", "BrokerTrailPct", "Notes"])

    config_df = config_df.rename(columns=lambda col: str(col).strip())
    if "Symbol" not in config_df.columns:
        raise ValueError("Stop config file must contain a Symbol column.")

    config_df["Symbol"] = config_df["Symbol"].astype(str).str.strip().str.upper()
    config_df = config_df[config_df["Symbol"].ne("") & config_df["Symbol"].ne("NAN")].copy()

    if "StopMode" not in config_df.columns:
        config_df["StopMode"] = "peak_stop"
    config_df["StopMode"] = config_df["StopMode"].fillna("peak_stop").astype(str).str.strip().str.lower()
    config_df.loc[~config_df["StopMode"].isin({"peak_stop", "broker_trailing"}), "StopMode"] = "peak_stop"

    if "StopPct" not in config_df.columns:
        config_df["StopPct"] = np.nan
    config_df["StopPct"] = pd.to_numeric(config_df["StopPct"], errors="coerce")

    if "BrokerTrailPct" not in config_df.columns:
        config_df["BrokerTrailPct"] = np.nan
    config_df["BrokerTrailPct"] = pd.to_numeric(config_df["BrokerTrailPct"], errors="coerce")

    if "Notes" not in config_df.columns:
        config_df["Notes"] = ""
    config_df["Notes"] = config_df["Notes"].fillna("")

    config_df = config_df.drop_duplicates(subset=["Symbol"], keep="last")
    config_df["EffectiveStopPct"] = config_df["StopPct"].fillna(float(fallback_stoploss_pct))
    return config_df


def apply_stop_config(allocation_df, stop_config_df, fallback_stoploss_pct):
    merged = allocation_df.merge(
        stop_config_df[["Symbol", "StopMode", "StopPct", "BrokerTrailPct", "Notes", "EffectiveStopPct"]],
        on="Symbol",
        how="left",
    )
    merged["StopMode"] = merged["StopMode"].fillna("peak_stop")
    merged["StopPct"] = merged["StopPct"].fillna(float(fallback_stoploss_pct))
    merged["EffectiveStopPct"] = merged["EffectiveStopPct"].fillna(float(fallback_stoploss_pct))
    merged["BrokerTrailPct"] = merged["BrokerTrailPct"].astype(float)
    merged["StopNotes"] = merged["Notes"].fillna("")
    merged["StopConfigSource"] = np.where(
        merged["Notes"].isna() & merged["StopMode"].eq("peak_stop") & merged["StopPct"].eq(float(fallback_stoploss_pct)),
        "FallbackDefault",
        "ConfigFile",
    )
    merged = merged.drop(columns=["Notes"])
    return merged


def download_review_data(held_symbols):
    tickers = list(dict.fromkeys(base.tickers + held_symbols))

    dfs = []
    ohlc_data = {}
    for ticker in tickers + [base.benchmark] + list(base.macro_tickers.values()):
        print(f"Downloading {ticker}...")
        series, ohlc = base.safe_download(ticker)
        if series is not None:
            dfs.append(series)
        if ticker in tickers and ohlc is not None:
            ohlc_data[ticker] = ohlc

    data = pd.concat(dfs, axis=1).sort_index().ffill()
    required_columns = [base.benchmark] + list(base.macro_tickers.values())
    data = data.dropna(subset=required_columns)
    return data, ohlc_data


def build_stoploss_history(allocation_df, data, ohlc_data):
    detail_columns = [
        "Date",
        "Symbol",
        "StopMode",
        "ConfiguredStopPct",
        "BrokerTrailPct",
        "EntryDate",
        "EntryDateSource",
        "HighestHighSinceEntry",
        "StopLossPrice",
        "UpdatedToday",
        "LineValue",
    ]
    line_columns = ["Date", "StopLossLine"]
    detail_rows = []
    lines_by_date = {}
    orders_today_rows = []

    held_df = allocation_df[~allocation_df["IsCashLike"]].copy()

    for row in held_df.itertuples(index=False):
        symbol = row.Symbol
        stop_mode = getattr(row, "StopMode", "peak_stop")
        stop_pct = float(getattr(row, "EffectiveStopPct", np.nan))
        broker_trail_pct = getattr(row, "BrokerTrailPct", np.nan)

        if stop_mode == "broker_trailing":
            orders_today_rows.append(
                {
                    "Date": data.index.max().date(),
                    "Symbol": symbol,
                    "StopMode": stop_mode,
                    "Quantity": getattr(row, "Quantity", np.nan),
                    "Last": getattr(row, "Last", np.nan),
                    "EntryDate": getattr(row, "EffectiveEntryDate", pd.NaT),
                    "EntryDateSource": getattr(row, "EntryDateSource", ""),
                    "ConfiguredStopPct": np.nan,
                    "BrokerTrailPct": broker_trail_pct,
                    "CurrentPeakStopPrice": np.nan,
                    "Action": "USE_BROKER_TRAILING",
                    "OrderInstruction": (
                        f"Maintain Fidelity trailing stop at {broker_trail_pct:.2f}%."
                        if pd.notna(broker_trail_pct)
                        else "Set a Fidelity trailing stop using the configured broker trail percent."
                    ),
                }
            )
            continue

        if symbol not in ohlc_data:
            orders_today_rows.append(
                {
                    "Date": data.index.max().date(),
                    "Symbol": symbol,
                    "StopMode": stop_mode,
                    "Quantity": getattr(row, "Quantity", np.nan),
                    "Last": getattr(row, "Last", np.nan),
                    "EntryDate": getattr(row, "EffectiveEntryDate", pd.NaT),
                    "EntryDateSource": getattr(row, "EntryDateSource", ""),
                    "ConfiguredStopPct": stop_pct,
                    "BrokerTrailPct": broker_trail_pct,
                    "CurrentPeakStopPrice": np.nan,
                    "Action": "NO_PRICE_DATA",
                    "OrderInstruction": "No OHLC data available for stop calculation.",
                }
            )
            continue

        symbol_ohlc = ohlc_data[symbol].copy().sort_index()
        latest_symbol_timestamp = symbol_ohlc.index.max()

        entry_date = row.EffectiveEntryDate
        if pd.isna(entry_date):
            # When the allocation file has no entry date, fall back to the
            # latest tradable bar for that symbol so we still emit a current stop.
            entry_timestamp = latest_symbol_timestamp
        else:
            entry_timestamp = min(pd.Timestamp(entry_date), latest_symbol_timestamp)

        symbol_ohlc = symbol_ohlc.loc[symbol_ohlc.index >= entry_timestamp]
        if symbol_ohlc.empty:
            orders_today_rows.append(
                {
                    "Date": data.index.max().date(),
                    "Symbol": symbol,
                    "StopMode": stop_mode,
                    "Quantity": getattr(row, "Quantity", np.nan),
                    "Last": getattr(row, "Last", np.nan),
                    "EntryDate": entry_timestamp.date(),
                    "EntryDateSource": getattr(row, "EntryDateSource", ""),
                    "ConfiguredStopPct": stop_pct,
                    "BrokerTrailPct": broker_trail_pct,
                    "CurrentPeakStopPrice": np.nan,
                    "Action": "NO_HISTORY_AFTER_ENTRY",
                    "OrderInstruction": "No OHLC bars available on or after the effective entry date.",
                }
            )
            continue

        running_high = symbol_ohlc["High"].cummax()
        stoploss = running_high * (1 - abs(stop_pct) / 100)
        new_high = running_high.ne(running_high.shift(1)).fillna(True)

        for date, high_price, stop_price, is_new_high in zip(symbol_ohlc.index, running_high, stoploss, new_high):
            line_value = base.format_price(stop_price) if is_new_high else "NA"
            detail_rows.append(
                {
                    "Date": date.date(),
                    "Symbol": symbol,
                    "StopMode": stop_mode,
                    "ConfiguredStopPct": stop_pct,
                    "BrokerTrailPct": broker_trail_pct,
                    "EntryDate": entry_timestamp.date(),
                    "EntryDateSource": row.EntryDateSource,
                    "HighestHighSinceEntry": high_price,
                    "StopLossPrice": stop_price,
                    "UpdatedToday": bool(is_new_high),
                    "LineValue": line_value,
                }
            )
            lines_by_date.setdefault(date.date(), []).append((symbol, line_value))

        current_stop_price = float(stoploss.iloc[-1])
        orders_today_rows.append(
            {
                "Date": symbol_ohlc.index[-1].date(),
                "Symbol": symbol,
                "StopMode": stop_mode,
                "Quantity": getattr(row, "Quantity", np.nan),
                "Last": getattr(row, "Last", np.nan),
                "EntryDate": entry_timestamp.date(),
                "EntryDateSource": getattr(row, "EntryDateSource", ""),
                "ConfiguredStopPct": stop_pct,
                "BrokerTrailPct": broker_trail_pct,
                "CurrentPeakStopPrice": current_stop_price,
                "Action": "SET_PEAK_STOP",
                "OrderInstruction": f"Set or replace stop-loss at {base.format_price(current_stop_price)}.",
            }
        )

    line_rows = []
    for date in sorted(lines_by_date):
        joined = ",".join(f"<{symbol},{value}>" for symbol, value in lines_by_date[date])
        line_rows.append({"Date": date, "StopLossLine": f"{date} : {joined}"})

    return (
        pd.DataFrame(detail_rows, columns=detail_columns),
        pd.DataFrame(line_rows, columns=line_columns),
        pd.DataFrame(
            orders_today_rows,
            columns=[
                "Date",
                "Symbol",
                "StopMode",
                "Quantity",
                "Last",
                "EntryDate",
                "EntryDateSource",
                "ConfiguredStopPct",
                "BrokerTrailPct",
                "CurrentPeakStopPrice",
                "Action",
                "OrderInstruction",
            ],
        ),
    )


def build_momentum_review(allocation_df, score, regime):
    latest_date = score.dropna(how="all").index[-1]
    latest_regime = regime.loc[latest_date]
    ranked = score.loc[latest_date].dropna().sort_values(ascending=False)
    selected_positions, _ = base.select_positions(ranked, latest_regime)
    top_candidates = list(ranked.index[:10])

    held_symbols = set(allocation_df.loc[~allocation_df["IsCashLike"], "Symbol"])
    replacement_pool = [ticker for ticker in top_candidates if ticker not in held_symbols]

    rows = []
    for _, row in allocation_df.iterrows():
        symbol = row["Symbol"]
        in_universe = symbol in score.columns
        model_score = ranked.get(symbol, np.nan) if in_universe else np.nan
        model_rank = ranked.rank(ascending=False, method="min").get(symbol, np.nan) if in_universe else np.nan
        has_momentum = bool(symbol in selected_positions) if in_universe else False

        suggestions = replacement_pool[:3] if (not has_momentum and not row["IsCashLike"]) else []
        suggestion_text = ", ".join(
            f"{ticker} ({base.PRODUCT_DESCRIPTIONS.get(ticker, ticker)})"
            for ticker in suggestions
        )

        rows.append(
            {
                "AsOfDate": latest_date.date(),
                "Regime": latest_regime,
                "Symbol": symbol,
                "ProductDescription": base.PRODUCT_DESCRIPTIONS.get(symbol, symbol),
                "Quantity": row.get("Quantity", np.nan),
                "Last": row.get("Last", np.nan),
                "AveragePrice": row.get("AveragePrice", np.nan),
                "Value": row.get("Value", np.nan),
                "ProfitDollar": row.get("ProfitDollar", np.nan),
                "ProfitPct": row.get("ProfitPct", np.nan),
                "Basis": row.get("Basis", np.nan),
                "WeightPct": row.get("WeightPct", np.nan),
                "EntryDate": row["EffectiveEntryDate"],
                "EntryDateSource": row["EntryDateSource"],
                "IsCashLike": row["IsCashLike"],
                "InModelUniverse": in_universe,
                "HasMomentum": has_momentum if not row["IsCashLike"] else np.nan,
                "ModelScore": model_score,
                "ModelRank": model_rank,
                "CurrentTopSelections": ", ".join(selected_positions),
                "SuggestedMomentumAlternatives": suggestion_text,
            }
        )

    candidates_rows = []
    for ticker, score_value in ranked.head(10).items():
        candidates_rows.append(
            {
                "AsOfDate": latest_date.date(),
                "Regime": latest_regime,
                "Ticker": ticker,
                "ProductDescription": base.PRODUCT_DESCRIPTIONS.get(ticker, ticker),
                "Score": score_value,
                "SelectedNow": ticker in selected_positions,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(candidates_rows)


def get_attr(row, name):
    return row[name] if name in row else np.nan


def save_review_outputs(output_prefix, stoploss_detail, stoploss_lines, orders_today, momentum_review, momentum_candidates):
    output_dir = Path(__file__).resolve().parent
    momentum_review_path = output_dir / f"{output_prefix}_momentum_review.csv"
    momentum_candidates_path = output_dir / f"{output_prefix}_momentum_candidates.csv"
    stoploss_detail_path = output_dir / f"{output_prefix}_daily_stoploss_detail.csv"
    stoploss_lines_path = output_dir / f"{output_prefix}_daily_stoploss_lines.csv"
    stoploss_text_path = output_dir / f"{output_prefix}_allocation_stoploss_lines.txt"
    fidelity_orders_path = output_dir / f"{output_prefix}_fidelity_orders_today.csv"

    stoploss_text_path.write_text(
        "\n".join(stoploss_lines["StopLossLine"].tolist()) + "\n",
        encoding="utf-8",
    )
    momentum_review.to_csv(momentum_review_path, index=False)
    momentum_candidates.to_csv(momentum_candidates_path, index=False)
    stoploss_detail.to_csv(stoploss_detail_path, index=False)
    stoploss_lines.to_csv(stoploss_lines_path, index=False)
    orders_today.to_csv(fidelity_orders_path, index=False)

    return (
        momentum_review_path,
        momentum_candidates_path,
        stoploss_detail_path,
        stoploss_lines_path,
        stoploss_text_path,
        fidelity_orders_path,
    )


def main():
    args = parse_args()
    allocation_path = Path(args.allocation_path).expanduser()
    stop_config_path = Path(args.stop_config_path).expanduser()

    allocation_df, sheet_name, asof_date = load_allocation(allocation_path, args.sheet_name)
    stop_config_df = load_stop_config(stop_config_path, args.stoploss_pct)
    allocation_df = apply_stop_config(allocation_df, stop_config_df, args.stoploss_pct)
    held_symbols = allocation_df.loc[~allocation_df["IsCashLike"], "Symbol"].dropna().unique().tolist()

    data, ohlc_data = download_review_data(held_symbols)
    score_b, score_e, regime, _ = base.compute_features(
        data,
        use_macro_overlay=args.use_macro_overlay,
    )

    stoploss_detail, stoploss_lines, orders_today = build_stoploss_history(
        allocation_df,
        data,
        ohlc_data,
    )
    momentum_review, momentum_candidates = build_momentum_review(
        allocation_df,
        score_e,
        regime,
    )

    output_prefix = sanitize_name(allocation_path.stem)
    (
        momentum_review_path,
        momentum_candidates_path,
        stoploss_detail_path,
        stoploss_lines_path,
        stoploss_text_path,
        fidelity_orders_path,
    ) = save_review_outputs(
        output_prefix,
        stoploss_detail,
        stoploss_lines,
        orders_today,
        momentum_review,
        momentum_candidates,
    )

    print("\nAllocation review complete.")
    print(f"Source workbook: {allocation_path}")
    print(f"Source sheet: {sheet_name}")
    print(f"As-of date detected: {asof_date}")
    print(f"Momentum review CSV: {momentum_review_path}")
    print(f"Momentum candidates CSV: {momentum_candidates_path}")
    print(f"Daily stoploss detail CSV: {stoploss_detail_path}")
    print(f"Daily stoploss lines CSV: {stoploss_lines_path}")
    print(f"Stoploss text file: {stoploss_text_path}")
    print(f"Fidelity orders CSV: {fidelity_orders_path}")
    print(f"Stop config CSV: {stop_config_path}")
    fallback_count = int((allocation_df["EntryDateSource"] == "AsOfDateFallback").sum())
    if fallback_count > 0:
        print(
            f"Entry date fallback used for {fallback_count} rows because the spreadsheet has no explicit entry-date column."
        )
        print("That means stoploss history starts from the workbook as-of date for those rows.")


if __name__ == "__main__":
    main()
