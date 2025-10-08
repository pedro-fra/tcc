"""
XGBoost sales forecasting model implementation for tabular time series data.
Implements XGBoost with hyperparameter tuning and comprehensive forecasting pipeline.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visualization module
try:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from visualization.xgboost_plots import XGBoostVisualizer

    PLOTTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization module not available: {e}")
    PLOTTING_AVAILABLE = False


class XGBoostForecaster:
    """
    XGBoost forecasting model for sales prediction using tabular features.
    Implements complete pipeline from data loading to prediction evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost forecaster with configuration.

        Args:
            config: Configuration dictionary containing XGBoost parameters
        """
        self.config = config
        self.xgboost_config = config.get("xgboost", {})
        self.model = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.metrics = {}
        self.feature_importance = None
        self.model_fitted = False

        # Initialize visualizer if available
        if PLOTTING_AVAILABLE:
            try:
                self.visualizer = XGBoostVisualizer()
                logger.info("XGBoost visualizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize XGBoost visualizer: {e}")
                self.visualizer = None
        else:
            self.visualizer = None

        logger.info("XGBoostForecaster initialized with configuration")

    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess sales data for XGBoost using existing preprocessing pipeline.

        Args:
            file_path: Path to the CSV data file

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info(f"Loading and preprocessing data from {file_path}")

        try:
            # Import preprocessing modules
            from preprocessing.data_preprocessor import MLDataPreprocessor

            # Initialize preprocessor
            preprocessor = MLDataPreprocessor(self.config)

            # Load and preprocess data
            processed_df = preprocessor.process_data(Path(file_path))

            # First aggregate to time series format like other models
            from preprocessing import aggregate_to_time_series

            # Reconstruct date information from temporal features
            if all(col in processed_df.columns for col in ["year", "month", "day"]):
                processed_df["date"] = pd.to_datetime(processed_df[["year", "month", "day"]])
            else:
                raise ValueError("Missing temporal features (year, month, day) in processed data")

            # Aggregate to monthly time series like other models
            ts_config = self.config["time_series"]
            df_aggregated = aggregate_to_time_series(
                processed_df,
                date_column="date",
                value_column="target",
                frequency=ts_config["frequency"],
                aggregation_method=ts_config["aggregation_method"],
            )

            # Create time-based features from aggregated data
            features, target = self._create_xgboost_features_from_timeseries(df_aggregated)

            logger.info("Data loaded and preprocessed successfully:")
            logger.info(f"  - Features shape: {features.shape}")
            logger.info(f"  - Target shape: {target.shape}")
            logger.info(f"  - Feature columns: {len(features.columns)}")
            logger.info(f"  - Date range: {features.index.min()} to {features.index.max()}")

            return features, target

        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {e}")
            raise

    def prepare_data(
        self, features: pd.DataFrame, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets maintaining temporal order.

        Args:
            features: Features DataFrame
            target: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing train/test split")

        # Data already has proper date index from aggregation step
        logger.info(f"Using date index: {features.index.min()} to {features.index.max()}")

        # Calculate split point based on test ratio
        test_ratio = self.xgboost_config.get("test_ratio", 0.2)
        total_length = len(features)
        train_length = int(total_length * (1 - test_ratio))

        # Ensure minimum training data
        min_train_length = max(24, int(total_length * 0.7))  # At least 24 months or 70%
        train_length = max(train_length, min_train_length)

        # Split maintaining temporal order (no shuffling for time series)
        X_train = features.iloc[:train_length]
        X_test = features.iloc[train_length:]
        y_train = target.iloc[:train_length]
        y_test = target.iloc[train_length:]

        logger.info(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
        logger.info(f"Test period: {X_test.index.min()} to {X_test.index.max()}")
        logger.info(f"Split: {len(X_train)} train, {len(X_test)} test samples")

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def _create_xgboost_features_from_timeseries(
        self, df_aggregated: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create XGBoost features from aggregated time series data.

        Args:
            df_aggregated: DataFrame with date index and aggregated target values

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Creating XGBoost features from aggregated time series data")

        # Reset index to work with date column
        df_features = df_aggregated.reset_index()

        # Extract temporal features
        df_features["year"] = df_features["date"].dt.year
        df_features["month"] = df_features["date"].dt.month
        df_features["quarter"] = df_features["date"].dt.quarter
        df_features["day_of_week"] = df_features["date"].dt.dayofweek
        df_features["day_of_month"] = df_features["date"].dt.day
        df_features["week_of_year"] = df_features["date"].dt.isocalendar().week

        # Cyclical encoding for temporal features
        df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
        df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)
        df_features["quarter_sin"] = np.sin(2 * np.pi * df_features["quarter"] / 4)
        df_features["quarter_cos"] = np.cos(2 * np.pi * df_features["quarter"] / 4)
        df_features["day_of_week_sin"] = np.sin(2 * np.pi * df_features["day_of_week"] / 7)
        df_features["day_of_week_cos"] = np.cos(2 * np.pi * df_features["day_of_week"] / 7)

        # Calendar-specific features
        df_features["is_month_start"] = df_features["date"].dt.is_month_start.astype(int)
        df_features["is_month_end"] = df_features["date"].dt.is_month_end.astype(int)
        df_features["is_quarter_start"] = df_features["date"].dt.is_quarter_start.astype(int)
        df_features["is_quarter_end"] = df_features["date"].dt.is_quarter_end.astype(int)
        df_features["is_year_start"] = df_features["date"].dt.is_year_start.astype(int)
        df_features["is_year_end"] = df_features["date"].dt.is_year_end.astype(int)

        # Semester feature
        df_features["semester"] = ((df_features["month"] - 1) // 6) + 1
        df_features["semester_sin"] = np.sin(2 * np.pi * df_features["semester"] / 2)
        df_features["semester_cos"] = np.cos(2 * np.pi * df_features["semester"] / 2)

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
            df_features[f"target_min_{window}"] = (
                df_features["target"].rolling(window=window, min_periods=1).min()
            )
            df_features[f"target_max_{window}"] = (
                df_features["target"].rolling(window=window, min_periods=1).max()
            )

        # Create trend features
        df_features["target_diff_1"] = df_features["target"].diff(1)
        df_features["target_diff_12"] = df_features["target"].diff(12)  # Year-over-year
        df_features["target_pct_change_1"] = df_features["target"].pct_change(1)
        df_features["target_pct_change_12"] = df_features["target"].pct_change(12)

        # Create interaction features
        df_features["month_year_interaction"] = df_features["month"] * df_features["year"]
        df_features["quarter_year_interaction"] = df_features["quarter"] * df_features["year"]

        # Drop rows with NaN values (due to lag features)
        df_features = df_features.dropna()

        # Separate features and target
        feature_columns = [col for col in df_features.columns if col not in ["date", "target"]]
        features = df_features[feature_columns]
        target = df_features["target"]

        # Set date as index for visualization
        features.index = df_features["date"]
        target.index = df_features["date"]

        logger.info(f"Created {len(feature_columns)} features from time series data")
        logger.info(f"Final dataset: {len(features)} time points after handling NaN values")

        return features, target

    def initialize_model(self) -> xgb.XGBRegressor:
        """
        Initialize XGBoost model with configuration parameters.

        Returns:
            Configured XGBoost model instance
        """
        logger.info("Initializing XGBoost model")

        # Get XGBoost configuration parameters
        model_params = {
            "n_estimators": self.xgboost_config.get("n_estimators", 1000),
            "max_depth": self.xgboost_config.get("max_depth", 6),
            "learning_rate": self.xgboost_config.get("learning_rate", 0.1),
            "subsample": self.xgboost_config.get("subsample", 0.8),
            "colsample_bytree": self.xgboost_config.get("colsample_bytree", 0.8),
            "reg_alpha": self.xgboost_config.get("reg_alpha", 0.1),
            "reg_lambda": self.xgboost_config.get("reg_lambda", 1.0),
            "random_state": self.xgboost_config.get("random_state", 42),
            "objective": self.xgboost_config.get("objective", "reg:squarederror"),
            "eval_metric": self.xgboost_config.get("eval_metric", "rmse"),
            "verbosity": 0,  # Suppress XGBoost output
        }

        # Initialize model
        self.model = xgb.XGBRegressor(**model_params)

        logger.info(f"XGBoost initialized with parameters: {model_params}")

        return self.model

    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit XGBoost model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
        """
        logger.info("Fitting XGBoost model")

        if self.model is None:
            self.initialize_model()

        # Log training data statistics
        logger.info("Training data statistics:")
        logger.info(f"  - Features shape: {X_train.shape}")
        logger.info(f"  - Target shape: {y_train.shape}")
        logger.info(f"  - Target mean: {y_train.mean():.2f}")
        logger.info(f"  - Target std: {y_train.std():.2f}")

        try:
            # Check if hyperparameter tuning is enabled
            if self.xgboost_config.get("hyperparameter_tuning", False):
                logger.info("Starting hyperparameter tuning...")
                self._perform_hyperparameter_tuning(X_train, y_train)
            else:
                # Fit model directly (XGBoost Regressor doesn't support early_stopping_rounds in fit)
                self.model.fit(X_train, y_train)
                self.best_model = self.model

            self.model_fitted = True

            # Get feature importance
            self.feature_importance = self.best_model.feature_importances_

            logger.info("XGBoost model fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {e}")
            raise

    def _perform_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Perform hyperparameter tuning using GridSearchCV with TimeSeriesSplit.

        Args:
            X_train: Training features
            y_train: Training target
        """
        param_grid = self.xgboost_config.get("param_grid", {})
        cv_folds = self.xgboost_config.get("cv_folds", 5)

        # Use TimeSeriesSplit for cross-validation to respect temporal order
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.best_model = grid_search.best_estimator_

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.2f}")

    def generate_predictions(self, X_test: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate predictions using the fitted XGBoost model.

        Args:
            X_test: Test features (uses instance X_test if not provided)

        Returns:
            Array with predictions
        """
        if not self.model_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        if X_test is None:
            X_test = self.X_test

        if X_test is None:
            raise ValueError("No test data available for predictions")

        logger.info(f"Generating predictions for {len(X_test)} samples")

        try:
            # Generate predictions
            predictions = self.best_model.predict(X_test)

            logger.info("Predictions generated successfully")
            logger.info(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")

            self.predictions = predictions
            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise

    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance using standard forecasting metrics.

        Args:
            y_true: Actual observed values
            y_pred: Model predictions

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance")

        try:
            # Calculate metrics
            mae_score = mean_absolute_error(y_true, y_pred)
            rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))

            # Calculate MAPE (handling division by zero)
            mape_score = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

            metrics = {
                "mae": float(mae_score),
                "rmse": float(rmse_score),
                "mape": float(mape_score),
            }

            # Store metrics
            self.metrics = metrics

            logger.info("Model evaluation completed:")
            logger.info(f"  - MAE: {mae_score:.4f}")
            logger.info(f"  - RMSE: {rmse_score:.4f}")
            logger.info(f"  - MAPE: {mape_score:.4f}%")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def save_model(self, output_dir: Path) -> None:
        """
        Save the fitted XGBoost model to disk.

        Args:
            output_dir: Directory to save the model
        """
        if not self.model_fitted:
            logger.warning("Model not fitted - cannot save")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "xgboost_model.pkl"

        try:
            with open(model_path, "wb") as f:
                pickle.dump(self.best_model, f)
            logger.info(f"XGBoost model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: Path) -> None:
        """
        Load a previously saved XGBoost model.

        Args:
            model_path: Path to the saved model file
        """
        try:
            with open(model_path, "rb") as f:
                self.best_model = pickle.load(f)
                self.model = self.best_model
            self.model_fitted = True
            logger.info(f"XGBoost model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_results(self, output_dir: Path) -> None:
        """
        Save model results and metadata to JSON files.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create model summary
        model_summary = {
            "model_type": "XGBoost",
            "model_parameters": self.xgboost_config,
            "training_period": {
                "start": str(self.X_train.index.min()) if self.X_train is not None else None,
                "end": str(self.X_train.index.max()) if self.X_train is not None else None,
                "length": len(self.X_train) if self.X_train is not None else None,
            },
            "test_period": {
                "start": str(self.X_test.index.min()) if self.X_test is not None else None,
                "end": str(self.X_test.index.max()) if self.X_test is not None else None,
                "length": len(self.X_test) if self.X_test is not None else None,
            },
            "evaluation_metrics": self.metrics,
            "feature_importance": self.feature_importance.tolist()
            if self.feature_importance is not None
            else None,
            "feature_names": self.X_train.columns.tolist() if self.X_train is not None else None,
            "model_fitted": self.model_fitted,
        }

        with open(output_dir / "xgboost_model_summary.json", "w") as f:
            json.dump(model_summary, f, indent=2)

        logger.info(f"XGBoost results saved to {output_dir}")

    def generate_plots(self, output_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive plots for XGBoost analysis and results.

        Args:
            output_dir: Directory to save plots

        Returns:
            Dictionary with plot information
        """
        if not PLOTTING_AVAILABLE or self.visualizer is None:
            logger.warning("Plotting not available - visualization module not imported")
            return {"plots_generated": False, "reason": "Visualization module not available"}

        logger.info("Generating XGBoost visualization plots")

        # Save plots in main data/plots directory structure
        project_root = Path(__file__).parent.parent.parent
        plots_dir = project_root / "data" / "plots" / "xgboost"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_info = {"plots_generated": True, "plot_files": {}}

        # 1. Feature analysis plots
        if self.X_train is not None and self.feature_importance is not None:
            logger.info("Creating feature analysis plots")
            feature_plots = self.visualizer.create_feature_analysis(
                features=self.X_train,
                feature_importance=self.feature_importance,
                output_dir=plots_dir,
            )
            plot_info["plot_files"]["features"] = feature_plots

        # 2. Prediction comparison plots
        if self.X_test is not None and self.y_test is not None and self.predictions is not None:
            logger.info("Creating prediction comparison plots")
            prediction_plots = self.visualizer.create_prediction_comparison_plots(
                X_test=self.X_test,
                y_test=self.y_test,
                predictions=self.predictions,
                X_train=self.X_train,
                y_train=self.y_train,
                output_dir=plots_dir,
            )
            plot_info["plot_files"]["predictions"] = prediction_plots

        # 3. Model performance plots
        if self.metrics and self.y_test is not None and self.predictions is not None:
            logger.info("Creating model performance plots")
            performance_plots = self.visualizer.create_performance_plots(
                y_test=self.y_test,
                predictions=self.predictions,
                metrics=self.metrics,
                output_dir=plots_dir,
            )
            plot_info["plot_files"]["performance"] = performance_plots

        logger.info(f"Plots saved to {plots_dir}")
        return plot_info

    def run_complete_pipeline(self, data_file: str) -> Dict[str, Any]:
        """
        Execute the complete XGBoost forecasting pipeline.

        Args:
            data_file: Path to the input data file

        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("Starting complete XGBoost forecasting pipeline")

        try:
            # 1. Load and preprocess data
            features, target = self.load_data(data_file)

            # 2. Prepare train/test split
            X_train, X_test, y_train, y_test = self.prepare_data(features, target)

            # 3. Initialize and fit model
            self.initialize_model()
            self.fit_model(X_train, y_train)

            # 4. Generate predictions
            predictions = self.generate_predictions(X_test)

            # 5. Evaluate model
            metrics = self.evaluate_model(y_test, predictions)

            # 6. Save results
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "processed_data" / "xgboost"

            self.save_results(output_dir)
            self.save_model(output_dir)

            # 7. Generate plots
            plot_info = self.generate_plots(output_dir)

            # Compile complete results
            results = {
                "model_type": "XGBoost",
                "data_info": {
                    "total_samples": len(features),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features_count": features.shape[1],
                    "time_range": f"{features.index.min()} to {features.index.max()}",
                },
                "metrics": metrics,
                "model_fitted": self.model_fitted,
                "plots_generated": plot_info.get("plots_generated", False),
                "output_directory": str(output_dir),
            }

            logger.info("XGBoost forecasting pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    print("XGBoost forecasting model loaded successfully!")
