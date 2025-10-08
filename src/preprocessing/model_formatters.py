"""
Model-specific data formatters for different forecasting approaches.
Converts processed data to formats required by each model type.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Darts imports - required for time series forecasting
from darts import TimeSeries

from .utils import aggregate_to_time_series, save_processed_data

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesFormatter:
    """
    Formatter for time series models (ARIMA, Theta, Exponential Smoothing).
    Converts tabular data to Darts TimeSeries format.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize formatter with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def format_for_time_series_models(self, df: pd.DataFrame) -> Tuple[TimeSeries, pd.DataFrame]:
        """
        Convert processed tabular data to time series format.

        Args:
            df: Processed DataFrame from MLDataPreprocessor

        Returns:
            Tuple of (TimeSeries object, aggregated DataFrame)
        """
        logger.info("Formatting data for time series models")

        # Reconstruct date information from temporal features
        if all(col in df.columns for col in ["year", "month", "day"]):
            df_ts = df.copy()
            df_ts["date"] = pd.to_datetime(df_ts[["year", "month", "day"]])
        else:
            raise ValueError("Missing temporal features (year, month, day) in processed data")

        # Aggregate to time series
        ts_config = self.config["time_series"]
        df_aggregated = aggregate_to_time_series(
            df_ts,
            date_column="date",
            value_column="target",
            frequency=ts_config["frequency"],
            aggregation_method=ts_config["aggregation_method"],
        )

        # Handle missing dates in time series
        df_aggregated = self._fill_missing_dates(df_aggregated, ts_config["frequency"])

        # Convert to Darts TimeSeries
        series = TimeSeries.from_dataframe(df_aggregated, time_col="date", value_cols="target")

        # Convert to float32 for performance
        series = series.astype(np.float32)

        # Apply Darts preprocessing if configured
        series = self._apply_darts_preprocessing(series)

        logger.info(f"Created TimeSeries with {len(series)} time points")
        logger.info(f"Date range: {series.start_time()} to {series.end_time()}")

        return series, df_aggregated

    def prepare_for_arima(self, series: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
        """
        Prepare TimeSeries for ARIMA model (train/test split).

        Args:
            series: Input TimeSeries

        Returns:
            Tuple of (train_series, test_series)
        """
        logger.info("Preparing data for ARIMA model")

        # Split into train/test (80/20)
        split_point = int(len(series) * 0.8)
        train_series = series[:split_point]
        test_series = series[split_point:]

        logger.info(f"ARIMA split: {len(train_series)} train, {len(test_series)} test points")

        return train_series, test_series

    def prepare_for_theta(self, series: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
        """
        Prepare TimeSeries for Theta model (train/test split).

        Args:
            series: Input TimeSeries

        Returns:
            Tuple of (train_series, test_series)
        """
        logger.info("Preparing data for Theta model")

        # Same split as ARIMA
        return self.prepare_for_arima(series)

    def prepare_for_exponential_smoothing(
        self, series: TimeSeries
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Prepare TimeSeries for Exponential Smoothing model (train/test split).

        Args:
            series: Input TimeSeries

        Returns:
            Tuple of (train_series, test_series)
        """
        logger.info("Preparing data for Exponential Smoothing model")

        # Same split as ARIMA
        return self.prepare_for_arima(series)

    def save_time_series_data(
        self, series: TimeSeries, df_aggregated: pd.DataFrame, output_dir: Path
    ) -> None:
        """
        Save time series data for future use.

        Args:
            series: TimeSeries object
            df_aggregated: Aggregated DataFrame
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregated DataFrame
        save_processed_data(df_aggregated, output_dir / "time_series_aggregated.parquet")

        # Save TimeSeries as CSV for compatibility
        df_ts = series.to_dataframe()
        df_ts.to_csv(output_dir / "time_series_darts.csv")

        logger.info(f"Saved time series data to {output_dir}")

    def _fill_missing_dates(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Fill missing dates in time series to ensure continuous sequence.

        Args:
            df: DataFrame with date and target columns
            frequency: Frequency string (e.g., 'D', 'M', 'W')

        Returns:
            DataFrame with filled missing dates
        """
        # Create complete date range
        start_date = df["date"].min()
        end_date = df["date"].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

        # Create complete DataFrame
        df_complete = pd.DataFrame({"date": date_range})

        # Merge with original data
        df_filled = df_complete.merge(df, on="date", how="left")

        # Fill missing values with forward fill then backward fill
        df_filled["target"] = df_filled["target"].ffill().bfill()

        # If still missing values, fill with 0
        df_filled["target"] = df_filled["target"].fillna(0)

        missing_filled = len(df_filled) - len(df)
        if missing_filled > 0:
            logger.info(f"Filled {missing_filled} missing dates in time series")

        return df_filled

    def _apply_darts_preprocessing(self, series: TimeSeries) -> TimeSeries:
        """
        Apply Darts-specific preprocessing steps.
        Note: Outlier removal is disabled for sales forecasting as outliers
        often represent legitimate high-value sales.

        Args:
            series: Input TimeSeries

        Returns:
            Preprocessed TimeSeries
        """
        processed_series = series

        # Apply smoothing if configured
        ts_config = self.config.get("time_series", {})
        if ts_config.get("apply_smoothing", False):
            window_size = ts_config.get("smoothing_window", 3)
            processed_series = self._apply_moving_average_smoothing(processed_series, window_size)

        return processed_series

    def _apply_moving_average_smoothing(self, series: TimeSeries, window: int) -> TimeSeries:
        """
        Apply moving average smoothing to time series.

        Args:
            series: Input TimeSeries
            window: Window size for moving average

        Returns:
            Smoothed TimeSeries
        """
        # Convert to pandas
        df = series.to_dataframe()

        # Apply moving average
        df_smoothed = df.rolling(window=window, center=True, min_periods=1).mean()

        logger.info(f"Applied moving average smoothing with window size {window}")

        # Convert back to TimeSeries
        return TimeSeries.from_dataframe(df_smoothed)

    def check_stationarity(self, series: TimeSeries) -> Dict[str, Any]:
        """
        Perform stationarity tests on time series.

        Args:
            series: TimeSeries to test

        Returns:
            Dictionary with test results
        """
        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            # Get values as numpy array
            values = series.values().flatten()

            # ADF test
            adf_result = adfuller(values, autolag="AIC")
            adf_stationary = adf_result[1] < 0.05

            # KPSS test
            kpss_result = kpss(values, regression="c", nlags="auto")
            kpss_stationary = kpss_result[1] > 0.05

            results = {
                "adf_statistic": adf_result[0],
                "adf_pvalue": adf_result[1],
                "adf_stationary": adf_stationary,
                "kpss_statistic": kpss_result[0],
                "kpss_pvalue": kpss_result[1],
                "kpss_stationary": kpss_stationary,
                "both_tests_stationary": adf_stationary and kpss_stationary,
            }

            logger.info(f"Stationarity tests - ADF: {adf_stationary}, KPSS: {kpss_stationary}")
            return results

        except ImportError:
            logger.warning("statsmodels not available for stationarity tests")
            return {"error": "statsmodels not available"}

    def prepare_for_differencing(self, series: TimeSeries, order: int = 1) -> TimeSeries:
        """
        Apply differencing to make time series stationary.

        Args:
            series: Input TimeSeries
            order: Order of differencing

        Returns:
            Differenced TimeSeries
        """
        differenced = series.diff(n=order)
        logger.info(f"Applied differencing of order {order}")
        return differenced


class TabularFormatter:
    """
    Formatter for tabular ML models (XGBoost).
    Prepares data in tabular format with additional features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize formatter with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def format_for_xgboost(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Format data specifically for XGBoost model.

        Args:
            df: Processed DataFrame from MLDataPreprocessor

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Formatting data for XGBoost model")

        df_formatted = df.copy()

        # Separate features and target
        target = df_formatted["target"]
        features = df_formatted.drop("target", axis=1)

        # Additional feature engineering for XGBoost
        features = self._create_interaction_features(features)
        features = self._create_statistical_features(features)

        logger.info(f"XGBoost format: {len(features.columns)} features, {len(target)} samples")

        return features, target

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features for better ML performance.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with interaction features
        """
        df_interactions = df.copy()

        # Month-Year interaction
        if "month" in df_interactions.columns and "year" in df_interactions.columns:
            df_interactions["month_year"] = df_interactions["month"] * 100 + (
                df_interactions["year"] % 100
            )

        # Quarter-Year interaction
        if "quarter" in df_interactions.columns and "year" in df_interactions.columns:
            df_interactions["quarter_year"] = df_interactions["quarter"] * 100 + (
                df_interactions["year"] % 100
            )

        # Cyclical feature interactions
        if "month_sin" in df_interactions.columns and "month_cos" in df_interactions.columns:
            df_interactions["month_cyclical_interaction"] = (
                df_interactions["month_sin"] * df_interactions["month_cos"]
            )

        logger.info("Created interaction features for XGBoost")

        return df_interactions

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features from existing features.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with statistical features
        """
        df_stats = df.copy()

        # Find numerical columns for statistical features
        numerical_columns = df_stats.select_dtypes(include=[np.number]).columns.tolist()

        # Remove temporal and already aggregated columns
        exclude_patterns = ["year", "month", "day", "customer_", "target_lag", "target_ma"]
        base_numerical = [
            col
            for col in numerical_columns
            if not any(pattern in col for pattern in exclude_patterns)
        ]

        if len(base_numerical) >= 2:
            # Create sum of numerical features
            df_stats["numerical_sum"] = df_stats[base_numerical].sum(axis=1)

            # Create mean of numerical features
            df_stats["numerical_mean"] = df_stats[base_numerical].mean(axis=1)

            logger.info("Created statistical features for XGBoost")

        return df_stats

    def create_train_test_split(
        self, features: pd.DataFrame, target: pd.Series, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create time-aware train/test split for XGBoost.

        Args:
            features: Features DataFrame
            target: Target Series
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Creating time-aware train/test split for XGBoost")

        # For time series, use temporal split (not random)
        split_point = int(len(features) * (1 - test_size))

        X_train = features.iloc[:split_point].copy()
        X_test = features.iloc[split_point:].copy()
        y_train = target.iloc[:split_point].copy()
        y_test = target.iloc[split_point:].copy()

        logger.info(f"XGBoost split: {len(X_train)} train, {len(X_test)} test samples")

        return X_train, X_test, y_train, y_test

    def save_tabular_data(
        self, features: pd.DataFrame, target: pd.Series, output_dir: Path
    ) -> None:
        """
        Save tabular data for XGBoost.

        Args:
            features: Features DataFrame
            target: Target Series
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Combine features and target
        df_combined = features.copy()
        df_combined["target"] = target

        # Save combined data
        save_processed_data(df_combined, output_dir / "xgboost_data.parquet")

        # Save feature names for reference
        feature_info = {
            "feature_columns": features.columns.tolist(),
            "num_features": len(features.columns),
            "num_samples": len(features),
        }

        import json

        with open(output_dir / "feature_info.json", "w") as f:
            json.dump(feature_info, f, indent=2)

        logger.info(f"Saved tabular data to {output_dir}")


class ModelDataManager:
    """
    Main class to manage data formatting for all model types.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data manager with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ts_formatter = TimeSeriesFormatter(config)
        self.tabular_formatter = TabularFormatter(config)

    def prepare_all_model_data(
        self, processed_df: pd.DataFrame, output_base_dir: Path
    ) -> Dict[str, Any]:
        """
        Prepare data for all model types.

        Args:
            processed_df: Fully processed DataFrame from MLDataPreprocessor
            output_base_dir: Base output directory

        Returns:
            Dictionary with prepared data for each model type
        """
        logger.info("Preparing data for all model types")

        results = {}

        # Prepare time series data
        series, df_aggregated = self.ts_formatter.format_for_time_series_models(processed_df)

        # Save time series data
        ts_output_dir = output_base_dir / "time_series"
        self.ts_formatter.save_time_series_data(series, df_aggregated, ts_output_dir)

        results["time_series"] = {
            "series": series,
            "aggregated_df": df_aggregated,
            "output_dir": ts_output_dir,
        }

        # Prepare XGBoost data
        features, target = self.tabular_formatter.format_for_xgboost(processed_df)

        # Save XGBoost data
        xgb_output_dir = output_base_dir / "xgboost"
        self.tabular_formatter.save_tabular_data(features, target, xgb_output_dir)

        results["xgboost"] = {"features": features, "target": target, "output_dir": xgb_output_dir}

        logger.info("Data preparation completed for all model types")

        return results


if __name__ == "__main__":
    print("Model formatters loaded successfully!")
