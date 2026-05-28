from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PRODUCT_UNIVERSE_CSV_PATH = ROOT_DIR / "ProductUniverse.csv"


HARDCODED_PRODUCTS = [
    {"Symbol": "XLK", "AssetGroup": "sector", "ProductDescription": "Tech", "Leveraged": False},
    {"Symbol": "XLF", "AssetGroup": "sector", "ProductDescription": "Financials", "Leveraged": False},
    {"Symbol": "XLE", "AssetGroup": "sector", "ProductDescription": "Energy", "Leveraged": False},
    {"Symbol": "XLV", "AssetGroup": "sector", "ProductDescription": "Health Care", "Leveraged": False},
    {"Symbol": "XLY", "AssetGroup": "sector", "ProductDescription": "Consumer Discretionary", "Leveraged": False},
    {"Symbol": "XLP", "AssetGroup": "sector", "ProductDescription": "Consumer Staples", "Leveraged": False},
    {"Symbol": "XLI", "AssetGroup": "sector", "ProductDescription": "Industrials", "Leveraged": False},
    {"Symbol": "XLB", "AssetGroup": "sector", "ProductDescription": "Materials", "Leveraged": False},
    {"Symbol": "XLU", "AssetGroup": "sector", "ProductDescription": "Utilities", "Leveraged": False},
    {"Symbol": "XLC", "AssetGroup": "sector", "ProductDescription": "Communications", "Leveraged": False},
    {"Symbol": "REMX", "AssetGroup": "sector", "ProductDescription": "Rare Earths", "Leveraged": False},
    {"Symbol": "SMH", "AssetGroup": "sector", "ProductDescription": "Semiconductors", "Leveraged": False},
    {"Symbol": "TQQQ", "AssetGroup": "sector", "ProductDescription": "Leveraged Nasdaq 100", "Leveraged": True},
    {"Symbol": "NLR", "AssetGroup": "sector", "ProductDescription": "VANECK ETF TRUST URANIUM AND NUCL", "Leveraged": False},
    {"Symbol": "VOO", "AssetGroup": "sector", "ProductDescription": "Vanguard S&P 500 ETF", "Leveraged": False},
    {"Symbol": "VOX", "AssetGroup": "sector", "ProductDescription": "Vanguard Communication Services ETF", "Leveraged": False},
    {"Symbol": "VGT", "AssetGroup": "sector", "ProductDescription": "Vanguard Information Technology ETF", "Leveraged": False},
    {"Symbol": "ITB", "AssetGroup": "sector", "ProductDescription": "iShares U.S. Home Construction ETF", "Leveraged": False},
    {"Symbol": "LIT", "AssetGroup": "sector", "ProductDescription": "Global Lithium & Battery Tech ETF", "Leveraged": False},
    {"Symbol": "QTUM", "AssetGroup": "sector", "ProductDescription": "Defiance Quantum ETF", "Leveraged": False},
    {"Symbol": "NASA", "AssetGroup": "sector", "ProductDescription": "Tema Space Innovators ETF", "Leveraged": False},
    {"Symbol": "DRAM", "AssetGroup": "sector", "ProductDescription": "VanEck Vectors Semiconductor DRAM ETF", "Leveraged": False},
    {"Symbol": "UGL", "AssetGroup": "commodity", "ProductDescription": "Leveraged Gold", "Leveraged": True},
    {"Symbol": "AGQ", "AssetGroup": "commodity", "ProductDescription": "Leveraged Silver", "Leveraged": True},
]

EXTRA_PRODUCT_DESCRIPTIONS = {
    "SPY": "S&P 500",
    "IAU": "Gold",
    "SLV": "Silver",
    "SOXL": "Direxion Daily Semiconductor Bull 3X Shares",
}


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _load_products_from_csv(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No hardcoded product universe is defined and {csv_path} was not found."
        )

    universe_df = pd.read_csv(csv_path)
    required_columns = {"Symbol", "AssetGroup"}
    missing_columns = required_columns.difference(universe_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{csv_path.name} is missing required columns: {missing}. "
            "Expected at least Symbol and AssetGroup."
        )

    products = []
    for _, row in universe_df.iterrows():
        symbol = str(row["Symbol"]).strip().upper()
        asset_group = str(row["AssetGroup"]).strip().lower()
        if not symbol:
            continue
        products.append(
            {
                "Symbol": symbol,
                "AssetGroup": asset_group,
                "ProductDescription": str(row.get("ProductDescription", symbol)).strip() or symbol,
                "Leveraged": _normalize_bool(row.get("Leveraged", False)),
            }
        )
    return products


def load_product_universe(csv_path=None, hardcoded_products=None):
    products = hardcoded_products if hardcoded_products is not None else HARDCODED_PRODUCTS
    if products:
        source = "hardcoded"
        normalized_products = products
    else:
        csv_path = Path(csv_path) if csv_path is not None else PRODUCT_UNIVERSE_CSV_PATH
        normalized_products = _load_products_from_csv(csv_path)
        source = str(csv_path)

    sectors = [item["Symbol"] for item in normalized_products if item["AssetGroup"].lower() != "commodity"]
    commodities = [item["Symbol"] for item in normalized_products if item["AssetGroup"].lower() == "commodity"]
    tickers = sectors + commodities
    product_descriptions = {item["Symbol"]: item["ProductDescription"] for item in normalized_products}
    product_descriptions.update(EXTRA_PRODUCT_DESCRIPTIONS)
    leveraged_tickers = {item["Symbol"] for item in normalized_products if _normalize_bool(item.get("Leveraged", False))}

    return {
        "source": source,
        "products": normalized_products,
        "sectors": sectors,
        "commodities": commodities,
        "tickers": tickers,
        "product_descriptions": product_descriptions,
        "leveraged_tickers": leveraged_tickers,
    }
