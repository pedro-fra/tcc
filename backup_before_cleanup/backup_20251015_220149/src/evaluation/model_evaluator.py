"""
Comprehensive model evaluation and reporting for sales forecasting models.
Provides standardized evaluation metrics and comparison framework.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, mse, rmse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison framework.
    Provides standardized metrics and reporting for all forecasting models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model evaluator with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evaluation_results = {}

    def evaluate_forecast_accuracy(
        self, actual: TimeSeries, predicted: TimeSeries, model_name: str
    ) -> Dict[str, float]:
        """
        Calculate comprehensive forecast accuracy metrics.

        Args:
            actual: Actual values TimeSeries
            predicted: Predicted values TimeSeries
            model_name: Name of the model being evaluated

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating forecast accuracy for {model_name}")

        # Ensure same length for comparison
        min_length = min(len(actual), len(predicted))
        actual_aligned = actual[-min_length:]
        predicted_aligned = predicted[:min_length]

        # Calculate standard metrics
        mae_score = mae(actual_aligned, predicted_aligned)
        mse_score = mse(actual_aligned, predicted_aligned)
        rmse_score = rmse(actual_aligned, predicted_aligned)
        mape_score = mape(actual_aligned, predicted_aligned)

        # Calculate additional metrics
        metrics = {
            "mae": float(mae_score),
            "mse": float(mse_score),
            "rmse": float(rmse_score),
            "mape": float(mape_score),
            "samples_evaluated": min_length,
            "actual_mean": float(actual_aligned.values().mean()),
            "predicted_mean": float(predicted_aligned.values().mean()),
            "actual_std": float(actual_aligned.values().std()),
            "predicted_std": float(predicted_aligned.values().std()),
        }

        # Calculate relative metrics
        metrics.update(self._calculate_relative_metrics(actual_aligned, predicted_aligned))

        # Calculate directional accuracy
        metrics.update(self._calculate_directional_accuracy(actual_aligned, predicted_aligned))

        logger.info(f"{model_name} evaluation completed")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")

        return metrics

    def _calculate_relative_metrics(
        self, actual: TimeSeries, predicted: TimeSeries
    ) -> Dict[str, float]:
        """
        Calculate relative performance metrics.

        Args:
            actual: Actual values TimeSeries
            predicted: Predicted values TimeSeries

        Returns:
            Dictionary with relative metrics
        """
        actual_values = actual.values().flatten()
        predicted_values = predicted.values().flatten()

        # Mean Absolute Percentage Error components
        actual_mean = np.mean(actual_values)

        # Normalized metrics (relative to mean of actual values)
        normalized_mae = float(mae(actual, predicted) / actual_mean)
        normalized_rmse = float(rmse(actual, predicted) / actual_mean)

        # Coefficient of determination (R²)
        ss_res = np.sum((actual_values - predicted_values) ** 2)
        ss_tot = np.sum((actual_values - actual_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "normalized_mae": normalized_mae,
            "normalized_rmse": normalized_rmse,
            "r_squared": float(r_squared),
        }

    def _calculate_directional_accuracy(
        self, actual: TimeSeries, predicted: TimeSeries
    ) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics.

        Args:
            actual: Actual values TimeSeries
            predicted: Predicted values TimeSeries

        Returns:
            Dictionary with directional accuracy metrics
        """
        if len(actual) < 2 or len(predicted) < 2:
            return {"directional_accuracy": 0.0}

        actual_values = actual.values().flatten()
        predicted_values = predicted.values().flatten()

        # Calculate direction changes
        actual_directions = np.diff(actual_values) > 0
        predicted_directions = np.diff(predicted_values) > 0

        # Directional accuracy
        correct_directions = np.sum(actual_directions == predicted_directions)
        total_directions = len(actual_directions)
        directional_accuracy = correct_directions / total_directions if total_directions > 0 else 0

        return {
            "directional_accuracy": float(directional_accuracy),
            "direction_changes_actual": int(np.sum(actual_directions)),
            "direction_changes_predicted": int(np.sum(predicted_directions)),
        }

    def evaluate_model_from_files(
        self, model_name: str, results_dir: Path, actual_series: Optional[TimeSeries] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model from saved prediction files.

        Args:
            model_name: Name of the model
            results_dir: Directory containing model results
            actual_series: Actual time series for comparison

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model_name} from files in {results_dir}")

        # Load predictions
        predictions_file = results_dir / f"{model_name.lower()}_predictions.csv"
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

        predicted_df = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
        predicted_series = TimeSeries.from_dataframe(predicted_df)

        # Load existing metrics if available
        metrics_file = results_dir / f"{model_name.lower()}_metrics.json"
        existing_metrics = {}
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                existing_metrics = json.load(f)

        evaluation_result = {
            "model_name": model_name,
            "predictions_file": str(predictions_file),
            "prediction_period": {
                "start": str(predicted_series.start_time()),
                "end": str(predicted_series.end_time()),
                "periods": len(predicted_series),
            },
            "existing_metrics": existing_metrics,
        }

        # If actual series provided, calculate additional metrics
        if actual_series is not None:
            additional_metrics = self.evaluate_forecast_accuracy(
                actual_series, predicted_series, model_name
            )
            evaluation_result["additional_metrics"] = additional_metrics

        return evaluation_result

    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and rank them by performance.

        Args:
            model_results: List of model evaluation results

        Returns:
            Dictionary with model comparison results
        """
        logger.info("Comparing model performance")

        if not model_results:
            return {"error": "No model results provided for comparison"}

        comparison = {
            "models_compared": len(model_results),
            "comparison_metrics": ["mae", "rmse", "mape", "r_squared", "directional_accuracy"],
            "model_rankings": {},
            "best_models": {},
            "summary_statistics": {},
        }

        # Extract metrics for comparison
        metrics_data = {}
        for result in model_results:
            model_name = result["model_name"]

            # Use existing metrics or additional metrics
            if "additional_metrics" in result:
                metrics = result["additional_metrics"]
            elif "existing_metrics" in result:
                metrics = result["existing_metrics"]
            else:
                continue

            metrics_data[model_name] = metrics

        # Rank models by each metric
        for metric in comparison["comparison_metrics"]:
            if all(metric in metrics for metrics in metrics_data.values()):
                # Sort models by metric (lower is better for mae, rmse, mape; higher for r_squared, directional_accuracy)
                reverse_sort = metric in ["r_squared", "directional_accuracy"]

                sorted_models = sorted(
                    metrics_data.items(), key=lambda x: x[1][metric], reverse=reverse_sort
                )

                comparison["model_rankings"][metric] = [
                    {"model": model, "value": metrics[metric]} for model, metrics in sorted_models
                ]

                # Best model for this metric
                comparison["best_models"][metric] = sorted_models[0][0]

        # Calculate summary statistics
        for metric in comparison["comparison_metrics"]:
            if metric in comparison["model_rankings"]:
                values = [item["value"] for item in comparison["model_rankings"][metric]]
                comparison["summary_statistics"][metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        logger.info("Model comparison completed")
        return comparison

    def generate_evaluation_report(
        self, model_results: List[Dict[str, Any]], output_file: Path
    ) -> None:
        """
        Generate comprehensive evaluation report.

        Args:
            model_results: List of model evaluation results
            output_file: Path to save the report
        """
        logger.info("Generating evaluation report")

        # Compare models
        comparison = self.compare_models(model_results)

        # Create comprehensive report
        report = {
            "evaluation_summary": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "models_evaluated": len(model_results),
                "evaluation_framework": "TCC Sales Forecasting Evaluation",
            },
            "model_results": model_results,
            "model_comparison": comparison,
            "recommendations": self._generate_recommendations(comparison),
        }

        # Save report
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output_file}")

    def _generate_recommendations(self, comparison: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate model recommendations based on comparison results.

        Args:
            comparison: Model comparison results

        Returns:
            Dictionary with recommendations
        """
        if "best_models" not in comparison:
            return {"error": "Insufficient data for recommendations"}

        recommendations = {}
        best_models = comparison["best_models"]

        # Overall recommendation based on key metrics
        key_metrics = ["mae", "rmse", "mape"]
        model_scores = {}

        for model_name in set(best_models.values()):
            score = sum(1 for metric in key_metrics if best_models.get(metric) == model_name)
            model_scores[model_name] = score

        if model_scores:
            best_overall = max(model_scores.items(), key=lambda x: x[1])
            recommendations["best_overall"] = (
                f"{best_overall[0]} (won {best_overall[1]}/{len(key_metrics)} key metrics)"
            )

        # Specific use case recommendations
        if "mae" in best_models:
            recommendations["best_for_accuracy"] = f"{best_models['mae']} (lowest MAE)"

        if "mape" in best_models:
            recommendations["best_for_percentage_accuracy"] = f"{best_models['mape']} (lowest MAPE)"

        if "directional_accuracy" in best_models:
            recommendations["best_for_trend_prediction"] = (
                f"{best_models['directional_accuracy']} (highest directional accuracy)"
            )

        return recommendations


def load_time_series_for_evaluation(data_path: Path) -> TimeSeries:
    """
    Load time series data for model evaluation.

    Args:
        data_path: Path to time series data

    Returns:
        TimeSeries object for evaluation
    """
    ts_file = data_path / "time_series_darts.csv"
    if not ts_file.exists():
        raise FileNotFoundError(f"Time series file not found: {ts_file}")

    df = pd.read_csv(ts_file, index_col=0, parse_dates=True)
    return TimeSeries.from_dataframe(df)


if __name__ == "__main__":
    print("Model evaluation framework loaded successfully!")
