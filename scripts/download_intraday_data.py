"""
Download Intraday Commodity Data
Downloads 15-minute interval commodity data from Yahoo Finance
for the last 60 days and saves a validated combined dataset to CSV.

Note: yfinance only supports 15-min intraday data for the last 60 days.
This dataset is kept separate from historical (daily) data and is used
for short-term / recent market behavior analysis.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# =============================
# Configuration
# =============================

INTRADAY_INTERVAL = "15m"
LOOKBACK_DAYS = 59          # yfinance hard limit is 60 days for 15m data
OUTPUT_FILE = Path("data/intraday/intraday_data.csv")

COMMODITIES: Dict[str, str] = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "Crude Oil",
    "NG=F": "Natural Gas",
    "ZW=F": "Wheat",
    "HG=F": "Copper",
}

REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close"]

EXPECTED_COLUMN_ORDER = [
    "Datetime",
    "Date",
    "Time",
    "Commodity",
    "Open",
    "High",
    "Low",
    "Close",
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

def get_date_range():
    """Returns start and end date for the last 60 days."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    return (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten and standardize column names from yfinance."""

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[1] == "" else col[0]
            for col in df.columns
        ]

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

    logger.info(f"Downloading {name} ({ticker}) | 15-min intraday")

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=INTRADAY_INTERVAL,
            progress=False,
        )

        if data.empty:
            logger.warning(f"No intraday data returned for {name}")
            return None

        data = data.reset_index()
        data = flatten_columns(data)

        # Rename first column to Datetime (yfinance returns Datetime for intraday)
        first_col = data.columns[0]
        data = data.rename(columns={first_col: "Datetime"})

        # Parse Datetime and extract Date + Time as separate columns
        data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
        data["Datetime"] = data["Datetime"].dt.tz_convert("Asia/Kolkata")  # IST
        data["Date"] = data["Datetime"].dt.date
        data["Time"] = data["Datetime"].dt.strftime("%H:%M")

        data["Commodity"] = name

        logger.info(f"{name}: {len(data):,} rows downloaded ({INTRADAY_INTERVAL} interval)")
        return data

    except Exception as e:
        logger.error(f"Error downloading {name}: {e}")
        return None


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Validating combined intraday dataset...")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows missing critical fields
    df = df.dropna(subset=["Datetime", "Close"])
    df = df.drop_duplicates(subset=["Datetime", "Commodity"])

    # Ensure numeric types
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by commodity and datetime
    df = df.sort_values(by=["Commodity", "Datetime"]).reset_index(drop=True)

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


def log_summary(df: pd.DataFrame, start_date: str, end_date: str) -> None:

    logger.info("=" * 60)
    logger.info("INTRADAY DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Interval       : {INTRADAY_INTERVAL}")
    logger.info(f"Date range     : {start_date} to {end_date}")
    logger.info(f"Total rows     : {len(df):,}")
    logger.info(f"Timezone       : IST (Asia/Kolkata)")
    logger.info("Rows per commodity:")
    for commodity, count in df.groupby("Commodity").size().items():
        logger.info(f"  {commodity}: {count:,}")
    logger.info("=" * 60)


# =============================
# Main Execution
# =============================

def main() -> int:

    logger.info("=" * 60)
    logger.info("Starting Intraday Data Download")
    logger.info("=" * 60)

    start_date, end_date = get_date_range()
    logger.info(f"Fetching last {LOOKBACK_DAYS} days: {start_date} to {end_date}")

    all_data: List[pd.DataFrame] = []

    for ticker, name in COMMODITIES.items():
        df = download_single_commodity(ticker, name, start_date, end_date)
        if df is not None:
            all_data.append(df)

    if not all_data:
        logger.error("No intraday data downloaded successfully.")
        return 1

    combined_df = pd.concat(all_data, ignore_index=True)

    try:
        validated_df = validate_dataframe(combined_df)
    except ValueError as e:
        logger.error(str(e))
        return 1

    validated_df = enforce_column_order(validated_df)

    save_dataframe(validated_df, OUTPUT_FILE)
    log_summary(validated_df, start_date, end_date)

    logger.info("Intraday data download completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())