"""
Exponential Smoothing sales forecasting model implementation using Darts library.
Implements ExponentialSmoothing for automated parameter selection and comprehensive forecasting pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import ExponentialSmoothing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visualization module
try:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from visualization.exponential_plots import ExponentialVisualizer

    PLOTTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization module not available: {e}")
    PLOTTING_AVAILABLE = False


class ExponentialSmoothingForecaster:
    """
    Exponential Smoothing forecasting model using Darts for sales prediction.
    Implements complete pipeline from data loading to prediction evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Exponential Smoothing forecaster with configuration.

        Args:
            config: Configuration dictionary containing Exponential Smoothing parameters
        """
        self.config = config
        self.exp_config = config.get("exponential_smoothing", {})
        self.model = None
        self.train_series = None
        self.test_series = None
        self.predictions = None
        self.metrics = {}
        self.model_fitted = False
        self.trend_analysis = None
        self.seasonality_analysis = None

        # Initialize visualizer if available
        self.visualizer = ExponentialVisualizer(config) if PLOTTING_AVAILABLE else None

    def load_time_series_data(self, data_path: Path) -> TimeSeries:
        """
        Load preprocessed time series data for Exponential Smoothing modeling.

        Args:
            data_path: Path to time series data directory

        Returns:
            TimeSeries object ready for Exponential Smoothing modeling
        """
        logger.info("Loading time series data for Exponential Smoothing model")

        # Load Darts TimeSeries from CSV
        ts_file = data_path / "time_series_darts.csv"
        if not ts_file.exists():
            raise FileNotFoundError(f"Time series file not found: {ts_file}")

        # Read CSV and convert to TimeSeries
        df = pd.read_csv(ts_file, index_col=0, parse_dates=True)

        # Use original data without transformation for Exponential Smoothing
        series = TimeSeries.from_dataframe(df)

        # Store original series for evaluation purposes
        self.original_series = series.copy()

        logger.info(f"Loaded time series with {len(series)} time points")
        logger.info(f"Date range: {series.start_time()} to {series.end_time()}")
        logger.info("Using original data for Exponential Smoothing")

        return series

    def analyze_trend_seasonality(self, series: TimeSeries) -> Dict[str, Any]:
        """
        Analyze trend and seasonality patterns in the time series.

        Args:
            series: TimeSeries for analysis

        Returns:
            Dictionary with trend and seasonality analysis results
        """
        logger.info("Performing trend and seasonality analysis")

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Convert to pandas series for decomposition
            df = series.to_dataframe()
            ts_data = df.iloc[:, 0]

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                ts_data,
                model="additive",
                period=12,  # Monthly seasonality
            )

            # Calculate trend strength
            trend_strength = np.var(decomposition.trend.dropna()) / np.var(ts_data)

            # Calculate seasonality strength
            seasonal_strength = np.var(decomposition.seasonal) / np.var(ts_data)

            # Determine trend type
            trend_slope = np.polyfit(
                range(len(decomposition.trend.dropna())), decomposition.trend.dropna(), 1
            )[0]

            trend_type = (
                "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "none"
            )

            # Determine seasonality presence
            has_seasonality = seasonal_strength > 0.01  # Threshold for seasonality

            analysis_results = {
                "trend_strength": float(trend_strength),
                "seasonal_strength": float(seasonal_strength),
                "trend_type": trend_type,
                "trend_slope": float(trend_slope),
                "has_seasonality": bool(has_seasonality),
                "seasonal_period": 12,  # Monthly data with annual seasonality
                "decomposition_available": True,
                "recommended_model": self._recommend_model_type(
                    trend_strength, seasonal_strength, has_seasonality
                ),
            }

            # Store results for plotting
            self.trend_analysis = analysis_results

            logger.info(f"Trend strength: {trend_strength:.4f}")
            logger.info(f"Seasonal strength: {seasonal_strength:.4f}")
            logger.info(f"Trend type: {trend_type}")
            logger.info(f"Has seasonality: {has_seasonality}")

            return analysis_results

        except ImportError as e:
            logger.error("statsmodels not available for trend/seasonality analysis")
            raise ImportError("statsmodels required for Exponential Smoothing analysis") from e

    def _recommend_model_type(
        self, trend_strength: float, seasonal_strength: float, has_seasonality: bool
    ) -> str:
        """
        Recommend Exponential Smoothing model type based on analysis.

        Args:
            trend_strength: Strength of trend component
            seasonal_strength: Strength of seasonal component
            has_seasonality: Whether seasonality is present

        Returns:
            Recommended model type string
        """
        if trend_strength < 0.01 and not has_seasonality:
            return "Simple Exponential Smoothing (no trend, no seasonality)"
        elif trend_strength >= 0.01 and not has_seasonality:
            return "Holt's Linear Trend (trend, no seasonality)"
        elif trend_strength < 0.01 and has_seasonality:
            return "Seasonal Exponential Smoothing (no trend, seasonality)"
        else:
            return "Holt-Winters (trend and seasonality)"

    def create_train_test_split(
        self, series: TimeSeries, test_ratio: float = 0.2
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Create time-aware train/test split for Exponential Smoothing model.

        Args:
            series: Input TimeSeries
            test_ratio: Proportion of data for testing

        Returns:
            Tuple of (train_series, test_series)
        """
        logger.info("Creating time-aware train/test split")

        # Calculate split point
        split_point = int(len(series) * (1 - test_ratio))

        # Split maintaining temporal order
        train_series = series[:split_point]
        test_series = series[split_point:]

        logger.info(f"Train period: {train_series.start_time()} to {train_series.end_time()}")
        logger.info(f"Test period: {test_series.start_time()} to {test_series.end_time()}")
        logger.info(f"Split: {len(train_series)} train, {len(test_series)} test points")

        self.train_series = train_series
        self.test_series = test_series

        return train_series, test_series

    def initialize_model(self) -> ExponentialSmoothing:
        """
        Initialize ExponentialSmoothing model with configuration parameters.

        Returns:
            Configured ExponentialSmoothing model instance
        """
        logger.info("Initializing Exponential Smoothing model")

        # Get Exponential Smoothing configuration parameters
        # Use basic configuration but check if trend/seasonal can be detected automatically
        model_params = {}

        # Let Darts auto-detect the best configuration
        logger.info("Using auto-detection for trend and seasonal components")

        # Initialize model
        self.model = ExponentialSmoothing(**model_params)

        logger.info(f"Exponential Smoothing initialized with parameters: {model_params}")

        return self.model

    def fit_model(self, train_series: TimeSeries) -> None:
        """
        Fit Exponential Smoothing model to training data.

        Args:
            train_series: Training TimeSeries data
        """
        logger.info("Fitting Exponential Smoothing model to training data")

        if self.model is None:
            self.initialize_model()

        # Log training data statistics for debugging
        train_values = train_series.values().flatten()
        logger.info(
            f"Training data stats: min={train_values.min():.0f}, max={train_values.max():.0f}, mean={train_values.mean():.0f}, std={train_values.std():.0f}"
        )

        # Fit model
        self.model.fit(train_series)
        self.model_fitted = True

        logger.info("Exponential Smoothing model fitted successfully")

    def generate_predictions(self, forecast_horizon: int) -> TimeSeries:
        """
        Generate forecasts using fitted Exponential Smoothing model.

        Args:
            forecast_horizon: Number of periods to forecast

        Returns:
            TimeSeries with predictions
        """
        if not self.model_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        logger.info(f"Generating {forecast_horizon} period forecast")

        # Generate predictions
        predictions = self.model.predict(n=forecast_horizon)

        # Store predictions
        self.predictions = predictions

        logger.info(
            f"Generated predictions from {predictions.start_time()} to {predictions.end_time()}"
        )

        return predictions

    def evaluate_model(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        """
        Evaluate Exponential Smoothing model performance using multiple metrics.

        Args:
            actual: Actual values TimeSeries
            predicted: Predicted values TimeSeries

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating Exponential Smoothing model performance")

        # Calculate metrics
        mae_score = mae(actual, predicted)
        rmse_score = rmse(actual, predicted)
        mape_score = mape(actual, predicted)

        # Store metrics
        self.metrics = {
            "mae": float(mae_score),
            "rmse": float(rmse_score),
            "mape": float(mape_score),
            "forecast_horizon": len(predicted),
            "test_samples": len(actual),
        }

        # Log results
        logger.info("=== Exponential Smoothing Model Evaluation ===")
        logger.info(f"MAE: {mae_score:.4f}")
        logger.info(f"RMSE: {rmse_score:.4f}")
        logger.info(f"MAPE: {mape_score:.4f}%")

        return self.metrics

    def save_model_results(self, output_dir: Path) -> None:
        """
        Save Exponential Smoothing model results and artifacts.

        Args:
            output_dir: Directory to save results
        """
        logger.info("Saving Exponential Smoothing model results")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        if self.predictions is not None:
            predictions_df = self.predictions.to_dataframe()
            predictions_df.to_csv(output_dir / "exponential_predictions.csv")

        # Save metrics
        if self.metrics:
            with open(output_dir / "exponential_metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)

        # Save model summary
        model_summary = {
            "model_type": "ExponentialSmoothing",
            "fitted": self.model_fitted,
            "train_samples": len(self.train_series) if self.train_series else 0,
            "test_samples": len(self.test_series) if self.test_series else 0,
            "forecast_periods": len(self.predictions) if self.predictions else 0,
        }

        # Add trend/seasonality analysis results
        if self.trend_analysis:
            # Convert numpy types to JSON-serializable format
            trend_results = {}
            for key, value in self.trend_analysis.items():
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    trend_results[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    trend_results[key] = float(value)
                elif isinstance(value, np.bool_):
                    trend_results[key] = bool(value)
                else:
                    trend_results[key] = value
            model_summary["trend_seasonality_analysis"] = trend_results

        # Add fitted parameters placeholder
        if self.model_fitted:
            model_summary["model_fitted"] = True

        with open(output_dir / "exponential_model_summary.json", "w") as f:
            json.dump(model_summary, f, indent=2)

        logger.info(f"Exponential Smoothing results saved to {output_dir}")

    def generate_plots(
        self, output_dir: Path, series: Optional[TimeSeries] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive plots for Exponential Smoothing analysis and results.

        Args:
            output_dir: Directory to save plots
            series: Original time series for EDA (optional)

        Returns:
            Dictionary with plot information
        """
        if not PLOTTING_AVAILABLE or self.visualizer is None:
            logger.warning("Plotting not available - visualization module not imported")
            return {"plots_generated": False, "reason": "Visualization module not available"}

        logger.info("Generating Exponential Smoothing visualization plots")

        # Save plots in main data/plots directory structure
        project_root = Path(__file__).parent.parent.parent
        plots_dir = project_root / "data" / "plots" / "exponential_smoothing"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_info = {"plots_generated": True, "plot_files": {}}

        # 1. Exploratory Data Analysis (if series provided)
        if series is not None:
            logger.info("Creating exploratory data analysis plots")
            eda_plots = self.visualizer.create_exploratory_data_analysis(series, plots_dir)
            plot_info["plot_files"]["eda"] = eda_plots

        # 2. Prediction comparison plots (if predictions available)
        if (
            self.test_series is not None
            and self.predictions is not None
            and self.train_series is not None
        ):
            logger.info("Creating prediction comparison plots")
            prediction_plots = self.visualizer.create_prediction_comparison_plots(
                actual=self.test_series,
                predicted=self.predictions,
                train_series=self.train_series,
                output_dir=plots_dir,
                trend_analysis=self.trend_analysis,
            )
            plot_info["plot_files"]["predictions"] = prediction_plots

        # 3. Trend/Seasonality analysis plots (if available)
        if self.trend_analysis and series is not None:
            logger.info("Creating trend and seasonality analysis plots")
            trend_plot = self.visualizer.create_trend_seasonality_plots(
                original_series=series,
                trend_analysis=self.trend_analysis,
                output_dir=plots_dir,
            )
            plot_info["plot_files"]["trend_seasonality"] = trend_plot

        # 4. Model summary plot (if metrics available)
        if self.metrics:
            logger.info("Creating model summary plot")
            summary_plot = self.visualizer.create_model_summary_plot(
                metrics=self.metrics, output_dir=plots_dir
            )
            plot_info["plot_files"]["summary"] = summary_plot

        logger.info(f"All plots saved to {plots_dir}")
        return plot_info

    def run_complete_pipeline(self, data_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Execute complete Exponential Smoothing forecasting pipeline.

        Args:
            data_path: Path to preprocessed time series data
            output_dir: Directory to save results

        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting complete Exponential Smoothing forecasting pipeline")

        try:
            # Load data
            series = self.load_time_series_data(data_path)

            # Analyze trend and seasonality
            trend_analysis = self.analyze_trend_seasonality(series)

            # Create train/test split
            train_series, test_series = self.create_train_test_split(series)

            # Initialize and fit model
            self.initialize_model()
            self.fit_model(train_series)

            # Generate predictions
            predictions = self.generate_predictions(len(test_series))

            # Evaluate model
            metrics = self.evaluate_model(test_series, predictions)

            # Save results
            self.save_model_results(output_dir)

            # Generate comprehensive plots
            plot_info = self.generate_plots(output_dir, series)

            # Compile pipeline results
            pipeline_results = {
                "success": True,
                "trend_seasonality": trend_analysis,
                "metrics": metrics,
                "model_summary": {
                    "train_samples": len(train_series),
                    "test_samples": len(test_series),
                    "forecast_accuracy": metrics,
                },
                "plots": plot_info,
            }

            logger.info("Exponential Smoothing pipeline completed successfully")

            return pipeline_results

        except Exception as e:
            logger.error(f"Exponential Smoothing pipeline failed: {e}")
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("Exponential Smoothing forecasting model loaded successfully!")
