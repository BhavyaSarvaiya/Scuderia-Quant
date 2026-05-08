"""
Download Historical Commodity Data
Downloads daily historical commodity data from Yahoo Finance
and saves a validated combined dataset to CSV.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# =============================
# Configuration
# =============================

START_DATE = "2010-01-01"
OUTPUT_FILE = Path("data/historical/historical_data.csv")

COMMODITIES: Dict[str, str] = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "Crude Oil",
    "NG=F": "Natural Gas",
    "ZW=F": "Wheat",
    "HG=F": "Copper",
}

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close"]

EXPECTED_COLUMN_ORDER = [
    "Date",
    "Commodity",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


# =============================
# Logging Configuration
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# =============================
# Helper Functions
# =============================

def get_end_date() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten and standardize column names from yfinance.
    """

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[1] == "" else col[0]
            for col in df.columns
        ]

    # Remove any ticker suffixes like _GC=F
    cleaned_columns = []
    for col in df.columns:
        if "_" in col:
            col = col.split("_")[0]
        cleaned_columns.append(col.strip())

    df.columns = cleaned_columns

    return df


def download_single_commodity(
    ticker: str,
    name: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:

    logger.info(f"Downloading {name} ({ticker})")

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
        )

        if data.empty:
            logger.warning(f"No data returned for {name}")
            return None

        data = data.reset_index()
        data = flatten_columns(data)

        # Rename first column to Date (handles Date/Datetime)
        first_col = data.columns[0]
        data = data.rename(columns={first_col: "Date"})

        data["Date"] = pd.to_datetime(data["Date"])
        data["Commodity"] = name

        logger.info(f"{name}: {len(data):,} rows downloaded")
        return data

    except Exception as e:
        logger.error(f"Error downloading {name}: {e}")
        return None


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Validating combined dataset...")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=["Date", "Close"])
    df = df.drop_duplicates(subset=["Date", "Commodity"])

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by=["Commodity", "Date"]).reset_index(drop=True)

    logger.info("Validation complete")
    return df


def enforce_column_order(df: pd.DataFrame) -> pd.DataFrame:

    existing_cols = [col for col in EXPECTED_COLUMN_ORDER if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_cols]

    return df[existing_cols + remaining_cols]


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    file_size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved to {output_path}")
    logger.info(f"File size: {file_size_kb:.2f} KB")


def log_summary(df: pd.DataFrame) -> None:

    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Total rows: {len(df):,}")

    min_date = df["Date"].min().strftime("%Y-%m-%d")
    max_date = df["Date"].max().strftime("%Y-%m-%d")

    logger.info(f"Date range: {min_date} to {max_date}")

    logger.info("Rows per commodity:")
    for commodity, count in df.groupby("Commodity").size().items():
        logger.info(f"  {commodity}: {count:,}")

    logger.info("=" * 60)


# =============================
# Main Execution
# =============================

def main() -> int:

    logger.info("=" * 60)
    logger.info("Starting Historical Data Download")
    logger.info("=" * 60)

    end_date = get_end_date()

    all_data: List[pd.DataFrame] = []

    for ticker, name in COMMODITIES.items():
        df = download_single_commodity(ticker, name, START_DATE, end_date)
        if df is not None:
            all_data.append(df)

    if not all_data:
        logger.error("No data downloaded successfully.")
        return 1

    combined_df = pd.concat(all_data, ignore_index=True)

    try:
        validated_df = validate_dataframe(combined_df)
    except ValueError as e:
        logger.error(str(e))
        return 1

    validated_df = enforce_column_order(validated_df)

    save_dataframe(validated_df, OUTPUT_FILE)
    log_summary(validated_df)

    logger.info("Historical data download completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())