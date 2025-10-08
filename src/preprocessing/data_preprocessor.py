"""
Core data preprocessor for sales forecasting project.
Implements the MLDataPreprocessor class with 9-step pipeline (no outlier treatment).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

from .utils import (
    anonymize_customer_names,
    create_temporal_features,
    filter_valid_sales,
    handle_missing_values,
    remove_duplicates,
    validate_data_quality,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLDataPreprocessor:
    """
    Main data preprocessor class for sales forecasting models.
    Implements a 9-step preprocessing pipeline without outlier treatment.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary with all parameters
        """
        self.config = config
        self.data_quality_report = {}
        self.scaler = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.feature_columns = []

    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load raw sales data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}")

        try:
            df = pd.read_csv(
                filepath,
                sep=self.config["data"]["csv_separator"],
                encoding=self.config["data"]["encoding"],
            )

            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

            # Initial data validation
            self.data_quality_report["initial"] = validate_data_quality(df, self.config["data"])

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def step_1_create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Create target variable (sales aggregation).
        For this dataset, target is already the VALOR_LIQ column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with target variable
        """
        logger.info("Step 1: Creating target variable")

        df_processed = df.copy()

        # Filter only sales operations
        if self.config["data"]["operation_column"] in df_processed.columns:
            target_operation = self.config["data"]["target_operation"]
            df_processed = df_processed[
                df_processed[self.config["data"]["operation_column"]] == target_operation
            ].copy()
            logger.info(f"Filtered to {len(df_processed)} {target_operation} records")

        # Set target variable (sales value is already our target)
        df_processed["target"] = df_processed[self.config["data"]["value_column"]]

        return df_processed

    def step_2_create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Create temporal features from date column.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with temporal features
        """
        logger.info("Step 2: Creating temporal features")

        # Parse date column
        date_column = self.config["data"]["date_column"]
        df_processed = df.copy()

        # Convert date column to datetime
        df_processed[date_column] = pd.to_datetime(
            df_processed[date_column], format=self.config["data"]["date_format"]
        )

        # Create temporal features
        if self.config["feature_engineering"]["create_temporal_features"]:
            df_processed = create_temporal_features(df_processed, date_column)
            logger.info("Created linear temporal features")

        return df_processed

    def step_3_handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Handle missing values with type-specific strategies.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Step 3: Handling missing values")

        missing_strategy = self.config["data_quality"]["missing_value_strategy"]
        df_processed = handle_missing_values(df, missing_strategy)

        # Log missing value treatment
        missing_after = df_processed.isnull().sum().sum()
        logger.info(f"Missing values after treatment: {missing_after}")

        return df_processed

    def step_4_remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Remove duplicate records.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame without duplicates
        """
        logger.info("Step 4: Removing duplicate records")

        if self.config["data_quality"]["remove_duplicates"]:
            df_processed = remove_duplicates(df, keep="first")
        else:
            df_processed = df.copy()

        return df_processed

    def step_5_create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Create aggregated features for ML models.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with aggregated features
        """
        logger.info("Step 5: Creating aggregated features")

        df_processed = df.copy()

        if not self.config["feature_engineering"]["create_aggregated_features"]:
            return df_processed

        # Customer-based aggregated features
        customer_col = self.config["data"]["customer_column"]
        if customer_col in df_processed.columns:
            customer_stats = (
                df_processed.groupby(customer_col)["target"]
                .agg(["mean", "std", "count", "sum"])
                .add_prefix("customer_")
            )

            # Merge back to main dataframe
            df_processed = df_processed.merge(
                customer_stats, left_on=customer_col, right_index=True, how="left"
            )

            # Fill NaN values in aggregated features
            agg_columns = [col for col in df_processed.columns if col.startswith("customer_")]
            df_processed[agg_columns] = df_processed[agg_columns].fillna(0)

            logger.info(f"Created {len(agg_columns)} customer aggregated features")

        # Time-based aggregated features (if requested)
        if self.config["feature_engineering"]["create_lag_features"]:
            # Sort by date for lag features
            date_col = self.config["data"]["date_column"]
            df_processed = df_processed.sort_values(date_col)

            # Create lag features
            lag_periods = self.config["feature_engineering"]["lag_periods"]
            for lag in lag_periods:
                df_processed[f"target_lag_{lag}"] = df_processed["target"].shift(lag)

            logger.info(f"Created lag features for periods: {lag_periods}")

        # Moving averages (if requested)
        if self.config["feature_engineering"]["create_moving_averages"]:
            ma_windows = self.config["feature_engineering"]["ma_windows"]
            for window in ma_windows:
                df_processed[f"target_ma_{window}"] = (
                    df_processed["target"].rolling(window=window, min_periods=1).mean()
                )

            logger.info(f"Created moving averages for windows: {ma_windows}")

        return df_processed

    def step_6_encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6: Encode categorical variables using different strategies.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Step 6: Encoding categorical variables")

        df_processed = df.copy()

        # Anonymize customer names first
        customer_col = self.config["data"]["customer_column"]
        if customer_col in df_processed.columns:
            anonymization_config = self.config["anonymization"]
            df_processed[customer_col] = anonymize_customer_names(
                df_processed[customer_col],
                prefix=anonymization_config["customer_prefix"],
                hash_length=anonymization_config["hash_length"],
            )
            logger.info("Anonymized customer names")

        # Identify categorical columns
        categorical_columns = df_processed.select_dtypes(include=["object"]).columns.tolist()

        # Remove date column from categorical encoding
        date_col = self.config["data"]["date_column"]
        if date_col in categorical_columns:
            categorical_columns.remove(date_col)

        # Encode categorical variables
        xgb_config = self.config["xgboost"]["categorical_encoding"]
        high_cardinality_threshold = xgb_config["high_cardinality_threshold"]

        for col in categorical_columns:
            unique_values = df_processed[col].nunique()

            if unique_values <= high_cardinality_threshold:
                # One-hot encoding for low cardinality
                if xgb_config["low_cardinality_method"] == "onehot":
                    dummies = pd.get_dummies(df_processed[col], prefix=col, dummy_na=False)
                    df_processed = pd.concat([df_processed, dummies], axis=1)
                    df_processed.drop(col, axis=1, inplace=True)
                    logger.info(f"One-hot encoded {col} ({unique_values} categories)")
            else:
                # Label encoding for high cardinality
                if xgb_config["high_cardinality_method"] == "label":
                    le = LabelEncoder()
                    df_processed[f"{col}_encoded"] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                    df_processed.drop(col, axis=1, inplace=True)
                    logger.info(f"Label encoded {col} ({unique_values} categories)")

        return df_processed

    def step_7_remove_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 7: Remove redundant or irrelevant columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with relevant columns only
        """
        logger.info("Step 7: Removing irrelevant columns")

        df_processed = df.copy()

        # Columns to remove
        columns_to_remove = []

        # Original date column (replaced by temporal features)
        date_col = self.config["data"]["date_column"]
        if date_col in df_processed.columns:
            columns_to_remove.append(date_col)

        # Original value column (replaced by target)
        value_col = self.config["data"]["value_column"]
        if value_col in df_processed.columns and value_col != "target":
            columns_to_remove.append(value_col)

        # Operation column (already filtered)
        operation_col = self.config["data"]["operation_column"]
        if operation_col in df_processed.columns:
            columns_to_remove.append(operation_col)

        # Remove identified columns
        columns_to_remove = [col for col in columns_to_remove if col in df_processed.columns]
        if columns_to_remove:
            df_processed.drop(columns_to_remove, axis=1, inplace=True)
            logger.info(f"Removed columns: {columns_to_remove}")

        return df_processed

    def step_8_apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 8: Apply robust scaling to numerical variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with scaled features
        """
        logger.info("Step 8: Applying robust scaling")

        df_processed = df.copy()

        # Identify numerical columns to scale
        numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude certain columns from scaling
        exclude_columns = self.config["scaling"]["exclude_columns"]
        columns_to_scale = [col for col in numerical_columns if col not in exclude_columns]

        if columns_to_scale:
            # Initialize and fit scaler
            self.scaler = RobustScaler()
            df_processed[columns_to_scale] = self.scaler.fit_transform(
                df_processed[columns_to_scale]
            )

            logger.info(f"Scaled {len(columns_to_scale)} numerical columns using RobustScaler")

        return df_processed

    def step_9_consolidate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 9: Final data consolidation and validation.

        Args:
            df: Input DataFrame

        Returns:
            Final processed DataFrame
        """
        logger.info("Step 9: Consolidating processed data")

        df_processed = df.copy()

        # Handle any remaining missing values
        if df_processed.isnull().any().any():
            df_processed = df_processed.fillna(0)
            logger.info("Filled remaining missing values with 0")

        # Store feature columns for future reference
        self.feature_columns = [col for col in df_processed.columns if col != "target"]

        # Final data quality check
        self.data_quality_report["final"] = {
            "total_records": len(df_processed),
            "total_features": len(self.feature_columns),
            "target_column": "target",
            "missing_values": df_processed.isnull().sum().sum(),
            "memory_usage_mb": df_processed.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        logger.info(
            f"Final dataset: {len(df_processed)} records, {len(self.feature_columns)} features"
        )

        return df_processed

    def process_data(self, filepath: Path) -> pd.DataFrame:
        """
        Execute the complete 9-step preprocessing pipeline.

        Args:
            filepath: Path to raw data file

        Returns:
            Fully processed DataFrame
        """
        logger.info("Starting data preprocessing pipeline")

        # Load raw data
        df = self.load_data(filepath)

        # Execute 9-step pipeline
        df = self.step_1_create_target_variable(df)
        df = self.step_2_create_temporal_features(df)
        df = self.step_3_handle_missing_values(df)
        df = self.step_4_remove_duplicates(df)
        df = self.step_5_create_aggregated_features(df)
        df = self.step_6_encode_categorical_variables(df)
        df = self.step_7_remove_irrelevant_columns(df)
        df = self.step_8_apply_scaling(df)
        df = self.step_9_consolidate_data(df)

        # Filter valid sales (after all processing)
        if self.config["data_quality"]["filter_zero_values"]:
            min_value = self.config["data_quality"]["min_sales_value"]
            df = filter_valid_sales(df, "target", min_value)

        logger.info("Data preprocessing pipeline completed successfully")

        return df

    def save_quality_report(self, filepath: Path) -> None:
        """
        Save data quality report to JSON file.

        Args:
            filepath: Output file path
        """

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, "item"):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        converted_report = convert_numpy_types(self.data_quality_report)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(converted_report, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved data quality report to {filepath}")


if __name__ == "__main__":
    print("MLDataPreprocessor class loaded successfully!")
