"""
XGBoost model hyperparameter optimizer using Optuna.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from config import SEARCH_SPACES
from preprocessing import MLDataPreprocessor, aggregate_to_time_series

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class XGBoostOptimizer(BaseOptimizer):
    """
    Optimizer for XGBoost forecasting model using Optuna.

    Performs hyperparameter optimization with time-series cross-validation
    to find the best XGBoost configuration for sales forecasting.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_path: Path,
        search_space: Optional[Dict[str, Any]] = None,
        optuna_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize XGBoost optimizer.

        Args:
            config: Full configuration dictionary
            data_path: Path to input data file
            search_space: Hyperparameter search space (default: from config)
            optuna_config: Optuna configuration (default: from config)
        """
        search_space = search_space or SEARCH_SPACES["xgboost"]

        super().__init__(
            model_name="xgboost",
            config=config,
            search_space=search_space,
            optuna_config=optuna_config,
        )

        self.data_path = data_path
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        logger.info("XGBoost optimizer initialized")

    def prepare_data(self):
        """
        Load and prepare data for optimization.

        This follows the same preprocessing pipeline as the XGBoost model.
        """
        logger.info(f"Loading and preprocessing data from {self.data_path}")

        # Initialize preprocessor
        preprocessor = MLDataPreprocessor(self.config)

        # Load and preprocess data
        processed_df = preprocessor.process_data(Path(self.data_path))

        # Reconstruct date information
        if all(col in processed_df.columns for col in ["year", "month", "day"]):
            processed_df["date"] = pd.to_datetime(processed_df[["year", "month", "day"]])
        else:
            raise ValueError("Missing temporal features in processed data")

        # Aggregate to monthly time series
        ts_config = self.config["time_series"]
        df_aggregated = aggregate_to_time_series(
            processed_df,
            date_column="date",
            value_column="target",
            frequency=ts_config["frequency"],
            aggregation_method=ts_config["aggregation_method"],
        )

        # Create features from aggregated time series
        features, target = self._create_xgboost_features(df_aggregated)

        # Split into train and validation sets
        # Use same logic as XGBoostForecaster for consistency
        test_ratio = self.config["xgboost"]["test_ratio"]
        total_length = len(features)
        train_length = int(total_length * (1 - test_ratio))

        # Ensure minimum training data (same as model)
        min_train_length = max(24, int(total_length * 0.7))
        train_length = max(train_length, min_train_length)

        self.X_train = features.iloc[:train_length]
        self.y_train = target.iloc[:train_length]
        self.X_val = features.iloc[train_length:]
        self.y_val = target.iloc[train_length:]

        logger.info(
            f"Data prepared: {len(self.X_train)} train, {len(self.X_val)} val samples "
            f"(train_length={train_length}, split_ratio={train_length / total_length:.2f})"
        )

    def _create_xgboost_features(self, df_aggregated):
        """
        Create XGBoost features from aggregated time series.

        Args:
            df_aggregated: DataFrame with aggregated time series data

        Returns:
            Tuple of (features DataFrame, target Series)
        """

        df_features = df_aggregated.reset_index()

        # Extract temporal features
        df_features["year"] = df_features["date"].dt.year
        df_features["month"] = df_features["date"].dt.month
        df_features["quarter"] = df_features["date"].dt.quarter
        df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
        df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)
        df_features["quarter_sin"] = np.sin(2 * np.pi * df_features["quarter"] / 4)
        df_features["quarter_cos"] = np.cos(2 * np.pi * df_features["quarter"] / 4)

        # Create lag features
        lag_periods = self.config["feature_engineering"]["lag_periods"]
        for lag in lag_periods:
            df_features[f"target_lag_{lag}"] = df_features["target"].shift(lag)

        # Create moving averages
        ma_windows = self.config["feature_engineering"]["ma_windows"]
        for window in ma_windows:
            df_features[f"target_ma_{window}"] = (
                df_features["target"].rolling(window=window, min_periods=1).mean()
            )

        # Create rolling statistics
        for window in [3, 6, 12]:
            df_features[f"target_std_{window}"] = (
                df_features["target"].rolling(window=window, min_periods=1).std()
            )

        # Create trend features
        df_features["target_diff_1"] = df_features["target"].diff(1)
        df_features["target_pct_change_1"] = df_features["target"].pct_change(1)

        # Drop rows with NaN
        df_features = df_features.dropna()

        # Separate features and target
        feature_columns = [col for col in df_features.columns if col not in ["date", "target"]]
        features = df_features[feature_columns]
        target = df_features["target"]

        # Set date as index
        features.index = df_features["date"]
        target.index = df_features["date"]

        return features, target

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for XGBoost optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Mean Absolute Error on validation set
        """
        # Ensure data is prepared
        if self.X_train is None:
            self.prepare_data()

        # Suggest hyperparameters
        params = self.suggest_params(trial)

        # Build XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            gamma=params.get("gamma", 0.0),
            min_child_weight=params.get("min_child_weight", 1),
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )

        # Train model
        try:
            model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )

            # Predict on validation set
            y_pred = model.predict(self.X_val)

            # Calculate MAE
            mae = np.mean(np.abs(self.y_val.values - y_pred))

            # Report intermediate value for pruning
            trial.report(mae, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return mae

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float("inf")

    def optimize_with_cv(
        self,
        n_splits: int = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization with time-series cross-validation.

        Automatically adjusts n_splits based on dataset size:
        - <80 points: 2 splits
        - 80-150 points: 3 splits
        - >150 points: 5 splits

        Args:
            n_splits: Number of CV splits (auto if None)
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Dictionary with optimization results
        """
        # Prepare data if not already done
        if self.X_train is None:
            self.prepare_data()

        # Combine train and val for CV
        X_full = pd.concat([self.X_train, self.X_val])
        y_full = pd.concat([self.y_train, self.y_val])

        # Auto-adjust n_splits based on dataset size
        if n_splits is None:
            total_points = len(X_full)
            if total_points < 80:
                n_splits = 2
            elif total_points < 150:
                n_splits = 3
            else:
                n_splits = 5
            logger.info(f"Auto-selected {n_splits} CV splits for {total_points} data points")

        logger.info(f"Running optimization with {n_splits}-fold time-series CV")

        def objective_cv(trial):
            params = self.suggest_params(trial)

            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_full):
                X_tr, X_v = X_full.iloc[train_idx], X_full.iloc[val_idx]
                y_tr, y_v = y_full.iloc[train_idx], y_full.iloc[val_idx]

                model = xgb.XGBRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    learning_rate=params["learning_rate"],
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    reg_alpha=params.get("reg_alpha", 0.1),
                    reg_lambda=params.get("reg_lambda", 1.0),
                    gamma=params.get("gamma", 0.0),
                    min_child_weight=params.get("min_child_weight", 1),
                    objective="reg:squarederror",
                    random_state=42,
                    verbosity=0,
                )

                try:
                    model.fit(X_tr, y_tr, verbose=False)
                    y_pred = model.predict(X_v)
                    mae = np.mean(np.abs(y_v.values - y_pred))
                    cv_scores.append(mae)
                except Exception as e:
                    logger.warning(f"CV fold failed: {e}")
                    return float("inf")

            mean_mae = np.mean(cv_scores)
            trial.report(mean_mae, step=0)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_mae

        # Replace objective temporarily
        original_objective = self.objective
        self.objective = objective_cv

        # Run optimization
        results = self.optimize(n_trials=n_trials, timeout=timeout)

        # Restore original objective
        self.objective = original_objective

        return results
