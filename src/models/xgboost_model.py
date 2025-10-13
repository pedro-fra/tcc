"""
XGBoost sales forecasting model implementation using Darts library.
Implements XGBModel with automated lag feature creation and comprehensive forecasting pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import XGBModel

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
    XGBoost forecasting model using Darts XGBModel for sales prediction.
    Implements complete pipeline from data loading to prediction evaluation using Darts library.
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
        self.train_series = None
        self.test_series = None
        self.predictions = None
        self.metrics = {}
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

        logger.info("XGBoostForecaster initialized with Darts XGBModel")

    def load_time_series_data(self, data_path: Path) -> TimeSeries:
        """
        Load preprocessed time series data for XGBoost modeling using Darts TimeSeries format.

        Args:
            data_path: Path to time series data directory

        Returns:
            TimeSeries object ready for XGBoost modeling
        """
        logger.info("Loading time series data for XGBoost model")

        # Load Darts TimeSeries from CSV
        ts_file = data_path / "time_series_darts.csv"
        if not ts_file.exists():
            raise FileNotFoundError(f"Time series file not found: {ts_file}")

        # Read CSV and convert to TimeSeries
        df = pd.read_csv(ts_file, index_col=0, parse_dates=True)
        series = TimeSeries.from_dataframe(df)

        logger.info(f"Loaded time series with {len(series)} time points")
        logger.info(f"Date range: {series.start_time()} to {series.end_time()}")
        logger.info(
            f"Series statistics - Mean: {series.values().mean():.2f}, Std: {series.values().std():.2f}"
        )

        return series

    def prepare_data(self, series: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
        """
        Split time series into training and testing sets maintaining temporal order.

        Args:
            series: Input TimeSeries

        Returns:
            Tuple of (train_series, test_series)
        """
        logger.info("Preparing train/test split for time series")

        # Calculate split point based on test ratio
        test_ratio = self.xgboost_config.get("test_ratio", 0.2)
        total_length = len(series)
        train_length = int(total_length * (1 - test_ratio))

        # Ensure minimum training data
        min_train_length = max(24, int(total_length * 0.7))  # At least 24 months or 70%
        train_length = max(train_length, min_train_length)

        # Split using Darts slicing - maintains temporal order
        train_series = series[:train_length]
        test_series = series[train_length:]

        logger.info(f"Train period: {train_series.start_time()} to {train_series.end_time()}")
        logger.info(f"Test period: {test_series.start_time()} to {test_series.end_time()}")
        logger.info(f"Split: {len(train_series)} train, {len(test_series)} test samples")

        # Store for later use
        self.train_series = train_series
        self.test_series = test_series

        return train_series, test_series

    def initialize_model(self) -> XGBModel:
        """
        Initialize Darts XGBModel with configuration parameters.

        Returns:
            Configured Darts XGBModel instance
        """
        logger.info("Initializing Darts XGBModel")

        # Get lag configuration
        lags = self.xgboost_config.get("lags", [-1, -2, -3, -6, -12])

        # XGBoost model parameters for the underlying XGBRegressor
        model_params = {
            "n_estimators": self.xgboost_config.get("n_estimators", 100),
            "max_depth": self.xgboost_config.get("max_depth", 6),
            "learning_rate": self.xgboost_config.get("learning_rate", 0.1),
            "subsample": self.xgboost_config.get("subsample", 0.8),
            "colsample_bytree": self.xgboost_config.get("colsample_bytree", 0.8),
            "reg_alpha": self.xgboost_config.get("reg_alpha", 0.1),
            "reg_lambda": self.xgboost_config.get("reg_lambda", 1.0),
            "random_state": self.xgboost_config.get("random_state", 42),
        }

        # Initialize Darts XGBModel
        self.model = XGBModel(lags=lags, **model_params)

        logger.info(f"Darts XGBModel initialized with lags: {lags}")
        logger.info(f"XGBoost parameters: {model_params}")

        return self.model

    def fit_model(self, train_series: TimeSeries) -> None:
        """
        Fit XGBModel using Darts API.

        Args:
            train_series: Training TimeSeries
        """
        logger.info("Fitting XGBModel using Darts API")

        if self.model is None:
            self.initialize_model()

        # Log training data statistics
        logger.info("Training data statistics:")
        logger.info(f"  - Series length: {len(train_series)}")
        logger.info(f"  - Date range: {train_series.start_time()} to {train_series.end_time()}")
        logger.info(
            f"  - Value range: {float(train_series.min().values()[:, 0].min()):.2f} to {float(train_series.max().values()[:, 0].max()):.2f}"
        )
        logger.info(f"  - Mean value: {train_series.values().mean():.2f}")

        try:
            # Fit model using Darts API
            self.model.fit(train_series)
            self.model_fitted = True

            logger.info("XGBModel fitted successfully using Darts API")

        except Exception as e:
            logger.error(f"Error fitting XGBModel: {e}")
            raise

    def generate_predictions(
        self, n: int, series: Optional[TimeSeries] = None, num_samples: int = 1
    ) -> TimeSeries:
        """
        Generate predictions using the fitted Darts XGBModel.

        Args:
            n: Number of time steps to predict
            series: Series to predict from (uses train_series if not provided)
            num_samples: Number of samples for probabilistic forecasting

        Returns:
            TimeSeries with predictions
        """
        if not self.model_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        if series is None:
            series = self.train_series

        if series is None:
            raise ValueError("No series available for predictions")

        logger.info(f"Generating predictions for {n} time steps")
        if num_samples > 1:
            logger.info(f"Probabilistic forecasting with {num_samples} samples")

        try:
            # Generate predictions using Darts API
            predictions = self.model.predict(n=n, series=series, num_samples=num_samples)

            logger.info("Predictions generated successfully using Darts API")
            logger.info(
                f"Prediction period: {predictions.start_time()} to {predictions.end_time()}"
            )
            logger.info(
                f"Prediction range: {float(predictions.min().values()[:, 0].min()):.2f} to {float(predictions.max().values()[:, 0].max()):.2f}"
            )

            self.predictions = predictions
            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise

    def evaluate_model(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        """
        Evaluate model performance using Darts metrics.

        Args:
            actual: Actual observed TimeSeries
            predicted: Model predictions TimeSeries

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance using Darts metrics")

        try:
            # Calculate metrics using Darts functions
            mae_score = mae(actual, predicted)
            rmse_score = rmse(actual, predicted)
            mape_score = mape(actual, predicted)

            metrics = {
                "mae": float(mae_score),
                "rmse": float(rmse_score),
                "mape": float(mape_score),
            }

            # Store metrics
            self.metrics = metrics

            logger.info("Model evaluation completed using Darts metrics:")
            logger.info(f"  - MAE: {mae_score:.4f}")
            logger.info(f"  - RMSE: {rmse_score:.4f}")
            logger.info(f"  - MAPE: {mape_score:.4f}%")

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def save_model(self, output_dir: Path) -> None:
        """
        Save the fitted Darts XGBModel to disk.

        Args:
            output_dir: Directory to save the model
        """
        if not self.model_fitted:
            logger.warning("Model not fitted - cannot save")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "xgboost_darts_model.pkl"

        try:
            # Save using Darts model save method
            self.model.save(model_path)
            logger.info(f"Darts XGBModel saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: Path) -> None:
        """
        Load a previously saved Darts XGBModel.

        Args:
            model_path: Path to the saved model file
        """
        try:
            # Load using Darts model load method
            self.model = XGBModel.load(model_path)
            self.model_fitted = True
            logger.info(f"Darts XGBModel loaded from {model_path}")
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
            "model_type": "XGBoost_Darts",
            "model_parameters": self.xgboost_config,
            "training_period": {
                "start": str(self.train_series.start_time())
                if self.train_series is not None
                else None,
                "end": str(self.train_series.end_time()) if self.train_series is not None else None,
                "length": len(self.train_series) if self.train_series is not None else None,
            },
            "test_period": {
                "start": str(self.test_series.start_time())
                if self.test_series is not None
                else None,
                "end": str(self.test_series.end_time()) if self.test_series is not None else None,
                "length": len(self.test_series) if self.test_series is not None else None,
            },
            "evaluation_metrics": self.metrics,
            "model_fitted": self.model_fitted,
            "lags_used": self.xgboost_config.get("lags", [-1, -2, -3, -6, -12]),
        }

        with open(output_dir / "xgboost_darts_model_summary.json", "w") as f:
            json.dump(model_summary, f, indent=2)

        logger.info(f"XGBoost Darts results saved to {output_dir}")

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

        try:
            # Generate time series comparison plots
            if (
                self.train_series is not None
                and self.test_series is not None
                and self.predictions is not None
            ):
                logger.info("Creating prediction comparison plots")
                prediction_plots = self.visualizer.create_prediction_comparison_plots(
                    train_series=self.train_series,
                    test_series=self.test_series,
                    predictions=self.predictions,
                    output_dir=plots_dir,
                )
                plot_info["plot_files"]["predictions"] = prediction_plots

            # Generate performance plots
            if self.metrics and self.test_series is not None and self.predictions is not None:
                logger.info("Creating model performance plots")
                performance_plots = self.visualizer.create_performance_plots(
                    actual_series=self.test_series,
                    predicted_series=self.predictions,
                    metrics=self.metrics,
                    output_dir=plots_dir,
                )
                plot_info["plot_files"]["performance"] = performance_plots

        except Exception as e:
            logger.warning(f"Error generating plots: {e}")
            plot_info["plots_generated"] = False
            plot_info["error"] = str(e)

        logger.info(f"Plots saved to {plots_dir}")
        return plot_info

    def run_complete_pipeline(self, data_path: Path) -> Dict[str, Any]:
        """
        Execute the complete XGBoost forecasting pipeline using Darts.

        Args:
            data_path: Path to the input data directory

        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("Starting complete XGBoost forecasting pipeline with Darts")

        try:
            # 1. Load time series data
            series = self.load_time_series_data(data_path)

            # 2. Prepare train/test split
            train_series, test_series = self.prepare_data(series)

            # 3. Initialize and fit model
            self.initialize_model()
            self.fit_model(train_series)

            # 4. Generate predictions
            n_predict = len(test_series)
            predictions = self.generate_predictions(n=n_predict, series=train_series)

            # 5. Evaluate model
            metrics = self.evaluate_model(test_series, predictions)

            # 6. Save results
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "processed_data" / "xgboost"

            self.save_results(output_dir)
            self.save_model(output_dir)

            # 7. Generate plots
            plot_info = self.generate_plots(output_dir)

            # Compile complete results
            results = {
                "model_type": "XGBoost_Darts",
                "data_info": {
                    "total_samples": len(series),
                    "train_samples": len(train_series),
                    "test_samples": len(test_series),
                    "time_range": f"{series.start_time()} to {series.end_time()}",
                    "lags_used": self.xgboost_config.get("lags", [-1, -2, -3, -6, -12]),
                },
                "metrics": metrics,
                "model_fitted": self.model_fitted,
                "plots_generated": plot_info.get("plots_generated", False),
                "output_directory": str(output_dir),
            }

            logger.info("XGBoost Darts forecasting pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    print("XGBoost Darts forecasting model loaded successfully!")
