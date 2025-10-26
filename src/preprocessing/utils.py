"""
Utility functions for data preprocessing in sales forecasting project.
Includes data validation, anonymization, and helper functions.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def anonymize_customer_names(
    names: pd.Series, prefix: str = "CLIENTE_", hash_length: int = 4
) -> pd.Series:
    """
    Anonymize customer names using consistent hash-based approach.

    Args:
        names: Series of customer names
        prefix: Prefix for anonymized names
        hash_length: Length of hash suffix

    Returns:
        Series with anonymized names
    """

    def hash_name(name: str) -> str:
        if pd.isna(name):
            return f"{prefix}0000"

        # Create consistent hash
        hash_obj = hashlib.md5(str(name).encode("utf-8"))
        hash_hex = hash_obj.hexdigest()
        # Take first hash_length characters and convert to numeric
        hash_numeric = str(int(hash_hex[:8], 16))[-hash_length:].zfill(hash_length)
        return f"{prefix}{hash_numeric}"

    return names.apply(hash_name)


def validate_data_quality(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data quality and return summary statistics.

    Args:
        df: Input DataFrame
        config: Data configuration parameters

    Returns:
        Dictionary with data quality metrics
    """
    quality_metrics = {
        "total_records": len(df),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_records": df.duplicated().sum(),
        "date_range": None,
        "value_statistics": None,
        "operation_counts": None,
    }

    # Date range analysis
    if config["date_column"] in df.columns:
        date_col = df[config["date_column"]]
        if not date_col.empty:
            quality_metrics["date_range"] = {
                "min_date": str(date_col.min()),
                "max_date": str(date_col.max()),
                "unique_dates": date_col.nunique(),
            }

    # Value statistics
    if config["value_column"] in df.columns:
        value_col = df[config["value_column"]]
        quality_metrics["value_statistics"] = {
            "mean": float(value_col.mean()),
            "median": float(value_col.median()),
            "std": float(value_col.std()),
            "min": float(value_col.min()),
            "max": float(value_col.max()),
            "zero_values": int((value_col == 0).sum()),
            "negative_values": int((value_col < 0).sum()),
        }

    # Operation counts (removed - data pre-filtered by GERA_COBRANCA = 1)
    if "operation_column" in config and config["operation_column"] in df.columns:
        quality_metrics["operation_counts"] = (
            df[config["operation_column"]].value_counts().to_dict()
        )

    return quality_metrics


def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    Handle missing values based on column types and strategy.

    Args:
        df: Input DataFrame
        strategy: Missing value handling strategy

    Returns:
        DataFrame with missing values handled
    """
    df_processed = df.copy()

    for column in df_processed.columns:
        if df_processed[column].isnull().any():
            if df_processed[column].dtype == "object":
                # Categorical columns
                if strategy["categorical"] == "constant":
                    df_processed[column].fillna("Desconhecido", inplace=True)
            else:
                # Numerical columns
                if strategy["numerical"] == "zero":
                    df_processed[column].fillna(0, inplace=True)
                elif strategy["numerical"] == "mean":
                    df_processed[column].fillna(df_processed[column].mean(), inplace=True)
                elif strategy["numerical"] == "median":
                    df_processed[column].fillna(df_processed[column].median(), inplace=True)

    return df_processed


def remove_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Remove duplicate records from DataFrame.

    Args:
        df: Input DataFrame
        keep: Which duplicate to keep ('first', 'last', False)

    Returns:
        DataFrame without duplicates
    """
    initial_count = len(df)
    df_clean = df.drop_duplicates(keep=keep)
    removed_count = initial_count - len(df_clean)

    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate records")

    return df_clean


def filter_valid_sales(
    df: pd.DataFrame, value_column: str, min_value: float = 0.01
) -> pd.DataFrame:
    """
    Filter to keep only valid sales transactions.

    Args:
        df: Input DataFrame
        value_column: Name of value column
        min_value: Minimum valid sales value

    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)

    # Filter positive values above minimum
    df_filtered = df[df[value_column] >= min_value].copy()

    removed_count = initial_count - len(df_filtered)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} records with invalid sales values")

    return df_filtered


def create_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create temporal features from date column.

    Args:
        df: Input DataFrame
        date_column: Name of date column

    Returns:
        DataFrame with temporal features added
    """
    df_features = df.copy()
    date_series = pd.to_datetime(df_features[date_column])

    # Linear temporal features
    df_features["year"] = date_series.dt.year
    df_features["month"] = date_series.dt.month
    df_features["day"] = date_series.dt.day
    df_features["day_of_week"] = date_series.dt.dayofweek
    df_features["quarter"] = date_series.dt.quarter
    df_features["day_of_year"] = date_series.dt.dayofyear
    df_features["week_of_year"] = date_series.dt.isocalendar().week

    # Cyclical features using sine/cosine transformation
    df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
    df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)
    df_features["day_of_week_sin"] = np.sin(2 * np.pi * df_features["day_of_week"] / 7)
    df_features["day_of_week_cos"] = np.cos(2 * np.pi * df_features["day_of_week"] / 7)
    df_features["quarter_sin"] = np.sin(2 * np.pi * df_features["quarter"] / 4)
    df_features["quarter_cos"] = np.cos(2 * np.pi * df_features["quarter"] / 4)

    return df_features


def aggregate_to_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str = "M",
    aggregation_method: str = "sum",
) -> pd.DataFrame:
    """
    Aggregate transaction data to time series format.

    Args:
        df: Input DataFrame
        date_column: Name of date column
        value_column: Name of value column
        frequency: Aggregation frequency ('D', 'W', 'M', 'Q', 'Y')
        aggregation_method: Method to aggregate ('sum', 'mean', 'count')

    Returns:
        Aggregated time series DataFrame
    """
    # Ensure date column is datetime
    df_agg = df.copy()
    df_agg[date_column] = pd.to_datetime(df_agg[date_column])

    # Set date as index
    df_agg.set_index(date_column, inplace=True)

    # Aggregate based on method
    if aggregation_method == "sum":
        ts_data = df_agg[value_column].resample(frequency).sum()
    elif aggregation_method == "mean":
        ts_data = df_agg[value_column].resample(frequency).mean()
    elif aggregation_method == "count":
        ts_data = df_agg[value_column].resample(frequency).count()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    # Convert back to DataFrame
    result_df = ts_data.reset_index()
    result_df.columns = [date_column, value_column]

    # Fill missing dates with zeros if requested
    if frequency in ["D", "W", "M", "Q", "Y"]:
        # Create complete date range
        date_range = pd.date_range(
            start=result_df[date_column].min(), end=result_df[date_column].max(), freq=frequency
        )

        # Reindex and fill missing values
        result_df = result_df.set_index(date_column).reindex(date_range, fill_value=0)
        result_df = result_df.reset_index()
        result_df.columns = [date_column, value_column]

    return result_df


def save_processed_data(data: pd.DataFrame, filepath: Path, format_type: str = "parquet") -> None:
    """
    Save processed data to file.

    Args:
        data: DataFrame to save
        filepath: Output file path
        format_type: File format ('parquet', 'csv')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "parquet":
        data.to_parquet(filepath, compression="snappy")
    elif format_type == "csv":
        data.to_csv(filepath, index=False, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    logger.info(f"Saved processed data to {filepath}")


def load_processed_data(filepath: Path, format_type: str = "parquet") -> pd.DataFrame:
    """
    Load processed data from file.

    Args:
        filepath: Input file path
        format_type: File format ('parquet', 'csv')

    Returns:
        Loaded DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if format_type == "parquet":
        return pd.read_parquet(filepath)
    elif format_type == "csv":
        return pd.read_csv(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {format_type}")


if __name__ == "__main__":
    # Test functions
    print("Utility functions loaded successfully!")
