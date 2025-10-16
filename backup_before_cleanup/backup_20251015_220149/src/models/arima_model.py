"""
ARIMA sales forecasting model implementation using Darts library.
Implements AutoARIMA for automated parameter selection and comprehensive forecasting pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import AutoARIMA

# Import visualization module
try:
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from visualization.arima_plots import ArimaVisualizer

    PLOTTING_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Visualization module not available: {e}")
    PLOTTING_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArimaForecaster:
    """
    ARIMA forecasting model using Darts AutoARIMA for sales prediction.
    Implements complete pipeline from data loading to prediction evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ARIMA forecaster with configuration.

        Args:
            config: Configuration dictionary containing ARIMA parameters
        """
        self.config = config
        self.arima_config = config.get("arima", {})
        self.model = None
        self.train_series = None
        self.test_series = None
        self.predictions = None
        self.metrics = {}
        self.model_fitted = False
        self.stationarity_results = None
        self.differencing_results = None
        self.acf_pacf_results = None

        # Initialize visualizer if available
        self.visualizer = ArimaVisualizer(config) if PLOTTING_AVAILABLE else None

    def load_time_series_data(self, data_path: Path) -> TimeSeries:
        """
        Load preprocessed time series data for ARIMA modeling with optional log transformation.

        Args:
            data_path: Path to time series data directory

        Returns:
            TimeSeries object ready for ARIMA modeling
        """
        logger.info("Loading time series data for ARIMA model")

        # Load Darts TimeSeries from CSV
        ts_file = data_path / "time_series_darts.csv"
        if not ts_file.exists():
            raise FileNotFoundError(f"Time series file not found: {ts_file}")

        # Read CSV and convert to TimeSeries
        df = pd.read_csv(ts_file, index_col=0, parse_dates=True)
        original_series = TimeSeries.from_dataframe(df)

        # Store original series for evaluation purposes
        self.original_series = original_series.copy()

        # Check if log transformation should be applied
        if self.arima_config.get("use_log_transform", False):
            series = self._apply_log_transformation(original_series)
            self.log_transformed = True
            logger.info("Applied log transformation for variance stabilization")
        else:
            series = original_series
            self.log_transformed = False
            logger.info("Using original data without transformation")

        logger.info(f"Loaded time series with {len(series)} time points")
        logger.info(f"Date range: {series.start_time()} to {series.end_time()}")

        return series

    def _apply_log_transformation(self, series: TimeSeries) -> TimeSeries:
        """
        Apply log transformation to the time series for variance stabilization.

        Args:
            series: Original TimeSeries

        Returns:
            Log-transformed TimeSeries
        """
        logger.info("Applying log transformation for ARIMA preprocessing")

        values = series.values().flatten()
        log_offset = self.arima_config.get("log_offset", 1)

        # Check for negative values if configured
        if self.arima_config.get("check_for_negative", True):
            if np.any(values <= 0):
                min_val = np.min(values)
                logger.warning(
                    f"Found non-positive values (min: {min_val}). Using offset: {log_offset}"
                )

        # Apply log transformation with offset
        try:
            log_values = np.log(values + log_offset)
            log_series = TimeSeries.from_times_and_values(
                times=series.time_index, values=log_values, columns=series.columns
            )

            # Store transformation parameters for inverse transformation
            self.log_offset = log_offset

            logger.info(
                f"Log transformation completed. Original range: [{values.min():.0f}, {values.max():.0f}]"
            )
            logger.info(f"Log-transformed range: [{log_values.min():.3f}, {log_values.max():.3f}]")

            return log_series

        except Exception as e:
            logger.error(f"Log transformation failed: {e}")
            logger.warning("Falling back to original data")
            self.log_transformed = False
            return series

    def _inverse_log_transformation(self, log_series: TimeSeries) -> TimeSeries:
        """
        Apply inverse log transformation to convert back to original scale.

        Args:
            log_series: Log-transformed TimeSeries

        Returns:
            TimeSeries in original scale
        """
        if not self.log_transformed:
            return log_series

        try:
            log_values = log_series.values().flatten()
            original_values = np.exp(log_values) - self.log_offset

            original_series = TimeSeries.from_times_and_values(
                times=log_series.time_index, values=original_values, columns=log_series.columns
            )

            return original_series

        except Exception as e:
            logger.error(f"Inverse log transformation failed: {e}")
            return log_series

    def check_stationarity(self, series: TimeSeries) -> Dict[str, Any]:
        """
        Perform comprehensive stationarity tests on time series.

        Args:
            series: TimeSeries to test for stationarity

        Returns:
            Dictionary with detailed stationarity test results
        """
        logger.info("Performing stationarity tests")

        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            # Get values as numpy array
            values = series.values().flatten()

            # Augmented Dickey-Fuller test
            adf_result = adfuller(values, autolag="AIC")
            adf_stationary = adf_result[1] < 0.05

            # KPSS test
            kpss_result = kpss(values, regression="c", nlags="auto")
            kpss_stationary = kpss_result[1] > 0.05

            # Compile results
            stationarity_results = {
                "adf_test": {
                    "statistic": float(adf_result[0]),
                    "pvalue": float(adf_result[1]),
                    "critical_values": {k: float(v) for k, v in adf_result[4].items()},
                    "is_stationary": bool(adf_stationary),
                },
                "kpss_test": {
                    "statistic": float(kpss_result[0]),
                    "pvalue": float(kpss_result[1]),
                    "critical_values": {k: float(v) for k, v in kpss_result[3].items()},
                    "is_stationary": bool(kpss_stationary),
                },
                "both_tests_agree": bool(adf_stationary and kpss_stationary),
                "recommendation": self._get_stationarity_recommendation(
                    adf_stationary, kpss_stationary
                ),
            }

            # Store results for plotting
            self.stationarity_results = stationarity_results

            logger.info(f"ADF test - Stationary: {adf_stationary} (p-value: {adf_result[1]:.4f})")
            logger.info(
                f"KPSS test - Stationary: {kpss_stationary} (p-value: {kpss_result[1]:.4f})"
            )

            return stationarity_results

        except ImportError as e:
            logger.error("statsmodels not available for stationarity tests")
            raise ImportError("statsmodels required for ARIMA stationarity testing") from e

    def apply_differencing(self, series: TimeSeries, max_diff: int = 2) -> Tuple[TimeSeries, int]:
        """
        Apply differencing to make series stationary based on stationarity tests.

        Args:
            series: Input TimeSeries
            max_diff: Maximum number of differencing operations

        Returns:
            Tuple of (differenced_series, number_of_differences_applied)
        """
        logger.info("Applying differencing analysis to achieve stationarity")

        current_series = series.copy()
        diff_order = 0

        for d in range(max_diff + 1):
            # Test current series for stationarity
            stationarity_results = self.check_stationarity(current_series)

            adf_stationary = stationarity_results["adf_test"]["is_stationary"]
            kpss_stationary = stationarity_results["kpss_test"]["is_stationary"]

            logger.info(f"Differencing order {d}: ADF={adf_stationary}, KPSS={kpss_stationary}")

            # If both tests agree on stationarity, stop differencing
            if adf_stationary and kpss_stationary:
                logger.info(f"Series achieved stationarity with {d} differences")
                break

            # Apply one more difference if not at maximum
            if d < max_diff:
                current_series = current_series.diff()
                diff_order = d + 1
                logger.info(f"Applied difference {diff_order}")
            else:
                logger.warning(
                    f"Maximum differencing ({max_diff}) reached without achieving stationarity"
                )

        return current_series, diff_order

    def analyze_acf_pacf(self, series: TimeSeries, lags: int = 40) -> Dict[str, Any]:
        """
        Perform ACF and PACF analysis for ARIMA parameter selection.

        Args:
            series: TimeSeries for analysis
            lags: Number of lags to analyze

        Returns:
            Dictionary with ACF/PACF analysis results
        """
        logger.info(f"Performing ACF/PACF analysis with {lags} lags")

        try:
            from statsmodels.tsa.stattools import acf, pacf

            # Get values as numpy array
            values = series.values().flatten()

            # Calculate ACF and PACF
            acf_values, acf_confint = acf(values, nlags=lags, alpha=0.05)
            pacf_values, pacf_confint = pacf(values, nlags=lags, alpha=0.05)

            # Suggest ARIMA parameters based on ACF/PACF patterns
            suggested_params = self._suggest_arima_params(acf_values, pacf_values)

            analysis_results = {
                "acf_values": acf_values.tolist(),
                "pacf_values": pacf_values.tolist(),
                "acf_confint": acf_confint.tolist(),
                "pacf_confint": pacf_confint.tolist(),
                "lags": list(range(len(acf_values))),
                "suggested_params": suggested_params,
                "analysis_summary": self._interpret_acf_pacf(acf_values, pacf_values),
            }

            logger.info(f"ACF/PACF analysis completed. Suggested parameters: {suggested_params}")

            return analysis_results

        except ImportError as e:
            logger.error("statsmodels not available for ACF/PACF analysis")
            raise ImportError("statsmodels required for ACF/PACF analysis") from e

    def _suggest_arima_params(
        self, acf_values: np.ndarray, pacf_values: np.ndarray
    ) -> Dict[str, int]:
        """
        Suggest ARIMA parameters based on ACF/PACF patterns.

        Args:
            acf_values: Autocorrelation function values
            pacf_values: Partial autocorrelation function values

        Returns:
            Dictionary with suggested p, d, q parameters
        """
        # Find significant lags (outside confidence intervals)
        # Simple heuristic: count significant lags in first 10 lags

        # For PACF (AR order p): count significant lags before cutoff
        p_suggested = 0
        for i in range(1, min(11, len(pacf_values))):
            if abs(pacf_values[i]) > 0.2:  # Simple threshold
                p_suggested = i
            else:
                break

        # For ACF (MA order q): count significant lags before cutoff
        q_suggested = 0
        for i in range(1, min(11, len(acf_values))):
            if abs(acf_values[i]) > 0.2:  # Simple threshold
                q_suggested = i
            else:
                break

        # Limit suggestions to reasonable values
        p_suggested = min(p_suggested, 5)
        q_suggested = min(q_suggested, 5)

        return {
            "suggested_p": p_suggested,
            "suggested_d": 1,  # Default, should be determined by stationarity tests
            "suggested_q": q_suggested,
        }

    def _interpret_acf_pacf(self, acf_values: np.ndarray, pacf_values: np.ndarray) -> str:
        """
        Provide interpretation of ACF/PACF patterns.

        Args:
            acf_values: Autocorrelation function values
            pacf_values: Partial autocorrelation function values

        Returns:
            String with pattern interpretation
        """
        # Simple pattern recognition
        acf_decay = "slow" if len([x for x in acf_values[1:6] if abs(x) > 0.1]) > 3 else "fast"
        pacf_cutoff = len([x for x in pacf_values[1:6] if abs(x) > 0.1])

        if acf_decay == "slow" and pacf_cutoff <= 2:
            return "Pattern suggests AR model (PACF cuts off, ACF decays slowly)"
        elif acf_decay == "fast" and pacf_cutoff > 2:
            return "Pattern suggests MA model (ACF cuts off, PACF decays slowly)"
        else:
            return "Pattern suggests ARMA model (both ACF and PACF decay gradually)"

    def create_train_test_split(
        self, series: TimeSeries, test_ratio: float = 0.2
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Create time-aware train/test split for ARIMA model.

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

    def initialize_model(self) -> AutoARIMA:
        """
        Initialize AutoARIMA model with configuration parameters.

        Returns:
            Configured AutoARIMA model instance
        """
        logger.info("Initializing AutoARIMA model")

        # Get ARIMA configuration parameters with enhanced settings
        model_params = {
            "start_p": self.arima_config.get("start_p", 1),
            "start_q": self.arima_config.get("start_q", 1),
            "max_p": self.arima_config.get("max_p", 5),
            "max_q": self.arima_config.get("max_q", 5),
            "max_d": self.arima_config.get("max_d", 2),
            "seasonal": self.arima_config.get("seasonal", True),
            "season_length": self.arima_config.get(
                "season_length", 12
            ),  # Key parameter for seasonal patterns
            "max_P": self.arima_config.get("max_P", 2),
            "max_Q": self.arima_config.get("max_Q", 2),
            "max_D": self.arima_config.get("max_D", 1),
            "stepwise": self.arima_config.get(
                "stepwise", True
            ),  # Enable stepwise for better performance
            "approximation": self.arima_config.get("approximation", False),
            "seasonal_test": self.arima_config.get("seasonal_test", "seas"),
            "test": self.arima_config.get("test", "kpss"),
        }

        # Initialize model
        self.model = AutoARIMA(**model_params)

        logger.info(f"AutoARIMA initialized with parameters: {model_params}")

        return self.model

    def fit_model(self, train_series: TimeSeries) -> None:
        """
        Fit ARIMA model to training data with automated parameter selection.

        Args:
            train_series: Training TimeSeries data
        """
        logger.info("Fitting AutoARIMA model to training data")

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

        logger.info("ARIMA model fitted successfully")

    def generate_predictions(self, forecast_horizon: int) -> TimeSeries:
        """
        Generate forecasts using fitted ARIMA model.

        Args:
            forecast_horizon: Number of periods to forecast

        Returns:
            TimeSeries with predictions (transformed back to original scale)
        """
        if not self.model_fitted:
            raise ValueError("Model must be fitted before generating predictions")

        logger.info(f"Generating {forecast_horizon} period forecast")

        # Generate predictions on transformed scale
        predictions_transformed = self.model.predict(n=forecast_horizon)

        # Transform predictions back to original scale if log transformation was applied
        if self.log_transformed:
            logger.info("Applying inverse log transformation to predictions")
            predictions = self._inverse_log_transformation(predictions_transformed)
        else:
            predictions = predictions_transformed

        # Store predictions
        self.predictions = predictions

        logger.info(
            f"Generated predictions from {predictions.start_time()} to {predictions.end_time()}"
        )

        return predictions

    def evaluate_model(self, actual: TimeSeries, predicted: TimeSeries) -> Dict[str, float]:
        """
        Evaluate ARIMA model performance using multiple metrics.

        Args:
            actual: Actual values TimeSeries
            predicted: Predicted values TimeSeries

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating ARIMA model performance")

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
        logger.info("=== ARIMA Model Evaluation ===")
        logger.info(f"MAE: {mae_score:.4f}")
        logger.info(f"RMSE: {rmse_score:.4f}")
        logger.info(f"MAPE: {mape_score:.4f}%")

        return self.metrics

    def get_prediction_intervals(self, confidence_level: float = 0.95) -> Optional[TimeSeries]:
        """
        Get prediction intervals for forecasts.

        Args:
            confidence_level: Confidence level for intervals

        Returns:
            TimeSeries with prediction intervals or None if not supported
        """
        logger.info(f"Generating {confidence_level * 100}% prediction intervals")

        try:
            # Generate predictions with intervals
            forecast_horizon = len(self.test_series) if self.test_series is not None else 12
            predictions_with_intervals = self.model.predict(
                n=forecast_horizon,
                num_samples=1000,  # Monte Carlo samples for intervals
            )

            return predictions_with_intervals

        except Exception as e:
            logger.warning(f"Could not generate prediction intervals: {e}")
            return None

    def save_model_results(self, output_dir: Path) -> None:
        """
        Save ARIMA model results and artifacts.

        Args:
            output_dir: Directory to save results
        """
        logger.info("Saving ARIMA model results")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        if self.predictions is not None:
            predictions_df = self.predictions.to_dataframe()
            predictions_df.to_csv(output_dir / "arima_predictions.csv")

        # Save metrics
        if self.metrics:
            import json

            with open(output_dir / "arima_metrics.json", "w") as f:
                json.dump(self.metrics, f, indent=2)

        # Save model summary
        model_summary = {
            "model_type": "AutoARIMA",
            "fitted": self.model_fitted,
            "train_samples": len(self.train_series) if self.train_series else 0,
            "test_samples": len(self.test_series) if self.test_series else 0,
            "forecast_periods": len(self.predictions) if self.predictions else 0,
            "log_transformed": self.log_transformed,
            "preprocessing": {
                "log_transformation": self.log_transformed,
                "log_offset": getattr(self, "log_offset", None) if self.log_transformed else None,
            },
        }

        # Add differencing analysis results
        if self.differencing_results:
            # Convert boolean values to JSON-serializable format
            diff_results = {}
            for key, value in self.differencing_results.items():
                if isinstance(value, bool):
                    diff_results[key] = value
                elif isinstance(value, dict):
                    # Handle nested dictionaries with potential boolean values
                    diff_results[key] = self._convert_to_json_serializable(value)
                else:
                    diff_results[key] = value
            model_summary["differencing_analysis"] = diff_results

        # Add ACF/PACF analysis summary
        if self.acf_pacf_results:
            model_summary["acf_pacf_analysis"] = {
                "suggested_params": self.acf_pacf_results.get("suggested_params"),
                "pattern_interpretation": self.acf_pacf_results.get("analysis_summary"),
            }

        # Add fitted parameters placeholder (specific model info not easily accessible)
        if self.model_fitted:
            model_summary["model_fitted"] = True

        with open(output_dir / "arima_model_summary.json", "w") as f:
            json.dump(model_summary, f, indent=2)

        logger.info(f"ARIMA results saved to {output_dir}")

    def generate_plots(
        self, output_dir: Path, series: Optional[TimeSeries] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive plots for ARIMA analysis and results.

        Args:
            output_dir: Directory to save plots
            series: Original time series for EDA (optional)

        Returns:
            Dictionary with plot information
        """
        if not PLOTTING_AVAILABLE or self.visualizer is None:
            logger.warning("Plotting not available - visualization module not imported")
            return {"plots_generated": False, "reason": "Visualization module not available"}

        logger.info("Generating ARIMA visualization plots")

        # Save plots in main data/plots directory structure
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent
        plots_dir = project_root / "data" / "plots" / "arima"
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
                stationarity_results=self.stationarity_results,
            )
            plot_info["plot_files"]["predictions"] = prediction_plots

        # 3. ACF/PACF analysis plots (if available)
        if self.acf_pacf_results:
            logger.info("Creating ACF/PACF analysis plots")
            acf_pacf_plot = self.visualizer.create_acf_pacf_plots(
                acf_pacf_results=self.acf_pacf_results, output_dir=plots_dir
            )
            plot_info["plot_files"]["acf_pacf"] = acf_pacf_plot

        # 4. Differencing analysis plots (if differencing was applied)
        if (
            self.differencing_results
            and self.differencing_results.get("differences_applied", 0) > 0
        ):
            logger.info("Creating differencing analysis plots")
            differencing_plot = self.visualizer.create_differencing_analysis_plots(
                original_series=series,
                differencing_results=self.differencing_results,
                output_dir=plots_dir,
            )
            plot_info["plot_files"]["differencing"] = differencing_plot

        # 5. Model summary plot (if metrics available)
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
        Execute complete ARIMA forecasting pipeline.

        Args:
            data_path: Path to preprocessed time series data
            output_dir: Directory to save results

        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting complete ARIMA forecasting pipeline")

        try:
            # Load data
            series = self.load_time_series_data(data_path)

            # Check stationarity
            stationarity_results = self.check_stationarity(series)

            # Perform ACF/PACF analysis on original series
            acf_pacf_results = self.analyze_acf_pacf(series)

            # Perform differencing analysis for methodology compliance
            # but use original series for AutoARIMA (let it handle differencing internally)
            differencing_results = {"differences_applied": 0, "final_series_stationary": False}

            # Check if differencing is recommended
            if not (
                stationarity_results["adf_test"]["is_stationary"]
                and stationarity_results["kpss_test"]["is_stationary"]
            ):
                logger.info("Series is non-stationary, analyzing differencing requirements")
                differenced_series, diff_order = self.apply_differencing(series)

                # Re-check stationarity after differencing
                final_stationarity = self.check_stationarity(differenced_series)
                differencing_results = {
                    "differences_applied": int(diff_order),
                    "final_series_stationary": bool(final_stationarity["both_tests_agree"]),
                    "final_stationarity_tests": final_stationarity,
                }

                # Update ACF/PACF analysis with differenced series
                if diff_order > 0:
                    acf_pacf_results["differenced_series_analysis"] = self.analyze_acf_pacf(
                        differenced_series
                    )

            # Store analysis results
            self.differencing_results = differencing_results
            self.acf_pacf_results = acf_pacf_results

            # Use original series for training - AutoARIMA will handle differencing internally
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
                "stationarity": stationarity_results,
                "metrics": metrics,
                "model_summary": {
                    "train_samples": len(train_series),
                    "test_samples": len(test_series),
                    "forecast_accuracy": metrics,
                },
                "plots": plot_info,
            }

            logger.info("ARIMA pipeline completed successfully")

            return pipeline_results

        except Exception as e:
            logger.error(f"ARIMA pipeline failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_stationarity_recommendation(self, adf_stationary: bool, kpss_stationary: bool) -> str:
        """
        Get recommendation based on stationarity test results.

        Args:
            adf_stationary: ADF test result
            kpss_stationary: KPSS test result

        Returns:
            Recommendation string
        """
        if adf_stationary and kpss_stationary:
            return "Series is stationary - proceed with ARIMA modeling"
        elif not adf_stationary and not kpss_stationary:
            return "Series is non-stationary - consider differencing"
        elif adf_stationary and not kpss_stationary:
            return "Mixed results - trend-stationary, consider detrending"
        else:
            return "Mixed results - difference-stationary, apply differencing"

    def _convert_to_json_serializable(self, obj):
        """
        Convert objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        else:
            return obj


if __name__ == "__main__":
    print("ARIMA forecasting model loaded successfully!")
