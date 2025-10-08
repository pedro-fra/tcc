"""
Theta method sales forecasting model implementation using Darts library.
Implements AutoTheta for automated parameter selection and comprehensive forecasting pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import AutoTheta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visualization module
try:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from visualization.theta_plots import ThetaVisualizer

    PLOTTING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization module not available: {e}")
    PLOTTING_AVAILABLE = False


class ThetaForecaster:
    """
    Theta method forecasting model using Darts AutoTheta for sales prediction.
    Implements complete pipeline from data loading to prediction evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Theta forecaster with configuration.

        Args:
            config: Configuration dictionary containing Theta parameters
        """
        self.config = config
        self.theta_config = config.get("theta", {})
        self.model = None
        self.train_series = None
        self.test_series = None
        self.predictions = None
        self.metrics = {}
        self.model_fitted = False

        # Initialize visualizer if available
        if PLOTTING_AVAILABLE:
            try:
                self.visualizer = ThetaVisualizer()
                logger.info("Theta visualizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Theta visualizer: {e}")
                self.visualizer = None
        else:
            self.visualizer = None

        logger.info("ThetaForecaster initialized with configuration")

    def load_data(self, file_path: str) -> TimeSeries:
        """
        Load sales data and convert to Darts TimeSeries format.

        Args:
            file_path: Path to the CSV data file

        Returns:
            TimeSeries object with sales data
        """
        logger.info(f"Loading data from {file_path}")

        # Load data using project configuration
        data_config = self.config.get("data", {})

        try:
            # Read CSV with project-specific configuration
            df = pd.read_csv(
                file_path,
                sep=data_config.get("csv_separator", ";"),
                encoding=data_config.get("encoding", "utf-8-sig"),
            )

            # Filter for sales operations only
            target_operation = data_config.get("target_operation", "VENDA")
            df = df[df[data_config.get("operation_column", "OPERACAO")] == target_operation]

            # Convert date column
            date_col = data_config.get("date_column", "DATA_EMISSAO_PEDIDO")
            value_col = data_config.get("value_column", "VALOR_LIQ")
            date_format = data_config.get("date_format", "%d/%m/%Y")

            df[date_col] = pd.to_datetime(df[date_col], format=date_format)

            # Aggregate by month (sum sales values)
            df_monthly = (
                df.groupby(pd.Grouper(key=date_col, freq="M"))[value_col].sum().reset_index()
            )

            # Remove zero values
            df_monthly = df_monthly[df_monthly[value_col] > 0]

            # Convert to TimeSeries
            series = TimeSeries.from_dataframe(
                df_monthly, time_col=date_col, value_cols=[value_col]
            )

            logger.info("Data loaded successfully:")
            logger.info(f"  - Time range: {series.start_time()} to {series.end_time()}")
            logger.info(f"  - Total periods: {len(series)}")
            logger.info(f"  - Mean value: {series.values().mean():.2f}")
            logger.info(f"  - Std deviation: {series.values().std():.2f}")

            return series

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def prepare_data(self, series: TimeSeries) -> Tuple[TimeSeries, TimeSeries]:
        """
        Split time series into training and testing sets maintaining temporal order.

        Args:
            series: Complete time series data

        Returns:
            Tuple of (train_series, test_series)
        """
        logger.info("Preparing train/test split")

        # Calculate split point based on test ratio
        test_ratio = self.theta_config.get("test_ratio", 0.2)
        total_length = len(series)
        train_length = int(total_length * (1 - test_ratio))

        # Ensure minimum training data
        min_train_length = max(24, int(total_length * 0.7))  # At least 24 months or 70%
        train_length = max(train_length, min_train_length)

        split_point = train_length

        # Split maintaining temporal order
        train_series = series[:split_point]
        test_series = series[split_point:]

        logger.info(f"Train period: {train_series.start_time()} to {train_series.end_time()}")
        logger.info(f"Test period: {test_series.start_time()} to {test_series.end_time()}")
        logger.info(f"Split: {len(train_series)} train, {len(test_series)} test points")

        self.train_series = train_series
        self.test_series = test_series

        return train_series, test_series

    def initialize_model(self) -> AutoTheta:
        """
        Initialize AutoTheta model with configuration parameters.

        Returns:
            Configured AutoTheta model instance
        """
        logger.info("Initializing AutoTheta model")

        # Get Theta configuration parameters
        # Note: AutoTheta has fewer configurable parameters than expected
        model_params = {
            "season_length": self.theta_config.get("season_length", 12),
        }

        # Log configuration used
        logger.info(f"Using season_length: {model_params['season_length']}")

        # Note: theta parameter and use_mle are handled automatically by AutoTheta
        # random_state may not be available in this model

        # Initialize model
        self.model = AutoTheta(**model_params)

        logger.info(f"AutoTheta initialized with parameters: {model_params}")

        return self.model

    def fit_model(self, train_series: TimeSeries) -> None:
        """
        Fit Theta model to training data with automated parameter selection.

        Args:
            train_series: Training TimeSeries data
        """
        logger.info("Fitting AutoTheta model to training data")

        if self.model is None:
            self.initialize_model()

        # Log training data statistics for debugging
        train_values = train_series.values().flatten()
        logger.info("Training data statistics:")
        logger.info(f"  - Length: {len(train_values)}")
        logger.info(f"  - Mean: {train_values.mean():.2f}")
        logger.info(f"  - Std: {train_values.std():.2f}")
        logger.info(f"  - Min: {train_values.min():.2f}")
        logger.info(f"  - Max: {train_values.max():.2f}")

        try:
            # Fit the model
            self.model.fit(train_series)
            self.model_fitted = True

            logger.info("AutoTheta model fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting Theta model: {e}")
            raise

    def generate_predictions(
        self, forecast_periods: Optional[int] = None, num_samples: int = 1
    ) -> TimeSeries:
        """
        Generate predictions using the fitted Theta model.

        Args:
            forecast_periods: Number of periods to forecast (defaults to test set length)
            num_samples: Number of Monte Carlo samples for probabilistic forecasting

        Returns:
            TimeSeries with predictions
        """
        if not self.model_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        if forecast_periods is None:
            forecast_periods = len(self.test_series) if self.test_series is not None else 12

        logger.info(f"Generating predictions for {forecast_periods} periods")

        try:
            # Generate predictions
            predictions = self.model.predict(n=forecast_periods, num_samples=num_samples)

            logger.info("Predictions generated successfully")
            logger.info(f"Prediction range: {predictions.start_time()} to {predictions.end_time()}")

            if num_samples > 1:
                logger.info(f"Probabilistic forecast with {num_samples} samples generated")

            self.predictions = predictions
            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise

    def evaluate_model(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        """
        Evaluate model performance using standard forecasting metrics.

        Args:
            actual: Actual observed values
            predicted: Model predictions

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance")

        try:
            # Calculate metrics
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
        Save the fitted Theta model to disk.

        Args:
            output_dir: Directory to save the model
        """
        if not self.model_fitted:
            logger.warning("Model not fitted - cannot save")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "theta_model.pkl"

        try:
            self.model.save(str(model_path))
            logger.info(f"Theta model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: Path) -> None:
        """
        Load a previously saved Theta model.

        Args:
            model_path: Path to the saved model file
        """
        try:
            self.model = AutoTheta.load(str(model_path))
            self.model_fitted = True
            logger.info(f"Theta model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert complex objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, bool):
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)

    def save_results(self, output_dir: Path) -> None:
        """
        Save model results and metadata to JSON files.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create model summary
        model_summary = {
            "model_type": "AutoTheta",
            "model_parameters": self.theta_config,
            "training_period": {
                "start": str(self.train_series.start_time()) if self.train_series else None,
                "end": str(self.train_series.end_time()) if self.train_series else None,
                "length": len(self.train_series) if self.train_series else None,
            },
            "test_period": {
                "start": str(self.test_series.start_time()) if self.test_series else None,
                "end": str(self.test_series.end_time()) if self.test_series else None,
                "length": len(self.test_series) if self.test_series else None,
            },
            "evaluation_metrics": self.metrics,
            "model_fitted": self.model_fitted,
        }

        with open(output_dir / "theta_model_summary.json", "w") as f:
            json.dump(model_summary, f, indent=2)

        logger.info(f"Theta results saved to {output_dir}")

    def generate_plots(
        self, output_dir: Path, series: Optional[TimeSeries] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive plots for Theta analysis and results.

        Args:
            output_dir: Directory to save plots
            series: Original time series for EDA (optional)

        Returns:
            Dictionary with plot information
        """
        if not PLOTTING_AVAILABLE or self.visualizer is None:
            logger.warning("Plotting not available - visualization module not imported")
            return {"plots_generated": False, "reason": "Visualization module not available"}

        logger.info("Generating Theta visualization plots")

        # Save plots in main data/plots directory structure
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        plots_dir = project_root / "data" / "plots" / "theta"
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
            )
            plot_info["plot_files"]["predictions"] = prediction_plots

        # 3. Model performance plots (if metrics available)
        if self.metrics and self.test_series is not None and self.predictions is not None:
            logger.info("Creating model performance plots")
            performance_plots = self.visualizer.create_performance_plots(
                actual=self.test_series,
                predicted=self.predictions,
                metrics=self.metrics,
                output_dir=plots_dir,
            )
            plot_info["plot_files"]["performance"] = performance_plots

        logger.info(f"Plots saved to {plots_dir}")
        return plot_info

    def run_complete_pipeline(self, data_file: str) -> Dict[str, Any]:
        """
        Execute the complete Theta forecasting pipeline.

        Args:
            data_file: Path to the input data file

        Returns:
            Dictionary containing all results and metrics
        """
        logger.info("Starting complete Theta forecasting pipeline")

        try:
            # 1. Load data
            series = self.load_data(data_file)

            # 2. Prepare train/test split
            train_series, test_series = self.prepare_data(series)

            # 3. Initialize and fit model
            self.initialize_model()
            self.fit_model(train_series)

            # 4. Generate predictions
            predictions = self.generate_predictions()

            # 5. Evaluate model
            metrics = self.evaluate_model(test_series, predictions)

            # 6. Save results
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "processed_data" / "theta"

            self.save_results(output_dir)
            self.save_model(output_dir)

            # 7. Generate plots
            plot_info = self.generate_plots(output_dir, series)

            # Compile complete results
            results = {
                "model_type": "AutoTheta",
                "data_info": {
                    "total_periods": len(series),
                    "train_periods": len(train_series),
                    "test_periods": len(test_series),
                    "time_range": f"{series.start_time()} to {series.end_time()}",
                },
                "metrics": metrics,
                "model_fitted": self.model_fitted,
                "plots_generated": plot_info.get("plots_generated", False),
                "output_directory": str(output_dir),
            }

            logger.info("Theta forecasting pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    print("Theta forecasting model loaded successfully!")
