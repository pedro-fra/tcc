"""
Comparison framework between Machine Learning models and Power BI forecasts.
Provides standardized evaluation and visualization for TCC analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerBIComparison:
    """
    Compare Machine Learning forecasts with Power BI baseline predictions.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize comparison framework.

        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = Path(data_dir)
        self.powerbi_data = None
        self.ml_forecasts = {}
        self.comparison_results = {}

    def load_powerbi_forecast(self, powerbi_file: Path) -> pd.DataFrame:
        """
        Load Power BI forecast data from CSV export.

        Args:
            powerbi_file: Path to Power BI exported CSV

        Returns:
            DataFrame with Power BI forecasts
        """
        logger.info(f"Loading Power BI forecast from {powerbi_file}")

        try:
            df = pd.read_csv(powerbi_file, parse_dates=["Data"])
            df.set_index("Data", inplace=True)

            required_columns = ["Valor_Real", "Valor_Projetado"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError(
                    f"CSV must contain columns: {required_columns}. Found: {df.columns.tolist()}"
                )

            logger.info(f"Loaded {len(df)} records from Power BI")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

            self.powerbi_data = df
            return df

        except Exception as e:
            logger.error(f"Error loading Power BI data: {e}")
            raise

    def load_ml_forecast(self, model_name: str, forecast_file: Optional[Path] = None) -> TimeSeries:
        """
        Load ML model forecast from saved predictions.

        Args:
            model_name: Name of the ML model (xgboost, arima, theta, exponential)
            forecast_file: Optional custom path to forecast file

        Returns:
            TimeSeries with ML predictions
        """
        logger.info(f"Loading {model_name} forecast")

        if forecast_file is None:
            model_dir = self.data_dir / "processed_data" / model_name.lower()
            forecast_file = model_dir / f"{model_name.lower()}_predictions.csv"

        if not forecast_file.exists():
            raise FileNotFoundError(f"Forecast file not found: {forecast_file}")

        df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)
        ts = TimeSeries.from_dataframe(df)

        logger.info(
            f"Loaded {model_name} forecast: {len(ts)} periods from {ts.start_time()} to {ts.end_time()}"
        )

        self.ml_forecasts[model_name] = ts
        return ts

    def align_forecasts(
        self, powerbi_df: pd.DataFrame, ml_series: TimeSeries
    ) -> tuple[pd.Series, pd.Series]:
        """
        Align Power BI and ML forecasts for comparison.

        Args:
            powerbi_df: Power BI forecast DataFrame
            ml_series: ML model TimeSeries

        Returns:
            Tuple of aligned (powerbi_values, ml_values)
        """
        ml_df = ml_series.pd_dataframe()

        common_dates = powerbi_df.index.intersection(ml_df.index)

        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between Power BI and ML forecasts")

        logger.info(f"Aligned {len(common_dates)} common dates")

        powerbi_aligned = powerbi_df.loc[common_dates, "Valor_Projetado"]
        ml_aligned = ml_df.loc[common_dates].iloc[:, 0]

        return powerbi_aligned, ml_aligned

    def calculate_comparison_metrics(
        self, powerbi_values: pd.Series, ml_values: pd.Series, model_name: str
    ) -> Dict[str, Any]:
        """
        Calculate comparison metrics between Power BI and ML forecasts.

        Args:
            powerbi_values: Power BI forecast values
            ml_values: ML model forecast values
            model_name: Name of ML model

        Returns:
            Dictionary with comparison metrics
        """
        logger.info(f"Calculating comparison metrics for {model_name}")

        powerbi_array = powerbi_values.values
        ml_array = ml_values.values

        diff = ml_array - powerbi_array
        abs_diff = np.abs(diff)
        pct_diff = (diff / powerbi_array) * 100

        metrics = {
            "model_name": model_name,
            "periods_compared": len(powerbi_values),
            "powerbi_total": float(powerbi_array.sum()),
            "ml_total": float(ml_array.sum()),
            "total_difference": float(diff.sum()),
            "total_difference_pct": float((diff.sum() / powerbi_array.sum()) * 100),
            "mean_absolute_difference": float(abs_diff.mean()),
            "mean_percentage_difference": float(pct_diff.mean()),
            "median_percentage_difference": float(np.median(pct_diff)),
            "std_percentage_difference": float(pct_diff.std()),
            "max_overestimate": float(diff.max()),
            "max_underestimate": float(diff.min()),
            "periods_overestimated": int((diff > 0).sum()),
            "periods_underestimated": int((diff < 0).sum()),
            "correlation": float(np.corrcoef(powerbi_array, ml_array)[0, 1]),
        }

        if abs_diff.mean() > 0:
            metrics["powerbi_mae_improvement_pct"] = float(
                ((abs_diff.mean() - abs_diff.mean()) / abs_diff.mean()) * 100
            )

        logger.info(f"{model_name} vs Power BI:")
        logger.info(f"  Total difference: {metrics['total_difference_pct']:.2f}%")
        logger.info(f"  Mean % difference: {metrics['mean_percentage_difference']:.2f}%")
        logger.info(f"  Correlation: {metrics['correlation']:.4f}")

        return metrics

    def compare_with_actual(
        self, actual_values: pd.Series, powerbi_values: pd.Series, ml_values: pd.Series
    ) -> Dict[str, Any]:
        """
        Compare both Power BI and ML forecasts against actual values.

        Args:
            actual_values: Actual observed values
            powerbi_values: Power BI forecasts
            ml_values: ML model forecasts

        Returns:
            Dictionary with accuracy metrics for both models
        """
        logger.info("Comparing both models against actual values")

        common_dates = actual_values.index.intersection(powerbi_values.index).intersection(
            ml_values.index
        )

        if len(common_dates) == 0:
            logger.warning("No common dates with actual values for comparison")
            return {}

        actual_aligned = actual_values.loc[common_dates].values
        powerbi_aligned = powerbi_values.loc[common_dates].values
        ml_aligned = ml_values.loc[common_dates].values

        powerbi_errors = powerbi_aligned - actual_aligned
        ml_errors = ml_aligned - actual_aligned

        metrics = {
            "periods_evaluated": len(common_dates),
            "powerbi_metrics": {
                "mae": float(np.mean(np.abs(powerbi_errors))),
                "rmse": float(np.sqrt(np.mean(powerbi_errors**2))),
                "mape": float(np.mean(np.abs(powerbi_errors / actual_aligned)) * 100),
                "bias": float(np.mean(powerbi_errors)),
            },
            "ml_metrics": {
                "mae": float(np.mean(np.abs(ml_errors))),
                "rmse": float(np.sqrt(np.mean(ml_errors**2))),
                "mape": float(np.mean(np.abs(ml_errors / actual_aligned)) * 100),
                "bias": float(np.mean(ml_errors)),
            },
        }

        mae_improvement = (
            (metrics["powerbi_metrics"]["mae"] - metrics["ml_metrics"]["mae"])
            / metrics["powerbi_metrics"]["mae"]
        ) * 100
        rmse_improvement = (
            (metrics["powerbi_metrics"]["rmse"] - metrics["ml_metrics"]["rmse"])
            / metrics["powerbi_metrics"]["rmse"]
        ) * 100
        mape_improvement = (
            (metrics["powerbi_metrics"]["mape"] - metrics["ml_metrics"]["mape"])
            / metrics["powerbi_metrics"]["mape"]
        ) * 100

        metrics["improvement"] = {
            "mae_improvement_pct": float(mae_improvement),
            "rmse_improvement_pct": float(rmse_improvement),
            "mape_improvement_pct": float(mape_improvement),
            "winner": "ML" if mae_improvement > 0 else "Power BI",
        }

        logger.info(
            f"  Power BI MAE: {metrics['powerbi_metrics']['mae']:.2f}, MAPE: {metrics['powerbi_metrics']['mape']:.2f}%"
        )
        logger.info(
            f"  ML MAE: {metrics['ml_metrics']['mae']:.2f}, MAPE: {metrics['ml_metrics']['mape']:.2f}%"
        )
        logger.info(f"  Winner: {metrics['improvement']['winner']}")

        return metrics

    def run_full_comparison(
        self, model_name: str, powerbi_file: Path, include_actual: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete comparison between ML model and Power BI.

        Args:
            model_name: Name of ML model to compare
            powerbi_file: Path to Power BI exported CSV
            include_actual: Whether to compare against actual values

        Returns:
            Complete comparison results
        """
        logger.info(f"Running full comparison: {model_name} vs Power BI")

        powerbi_df = self.load_powerbi_forecast(powerbi_file)
        ml_series = self.load_ml_forecast(model_name)

        powerbi_aligned, ml_aligned = self.align_forecasts(powerbi_df, ml_series)

        comparison_metrics = self.calculate_comparison_metrics(
            powerbi_aligned, ml_aligned, model_name
        )

        results = {
            "comparison_type": "ML vs Power BI",
            "model_name": model_name,
            "comparison_period": {
                "start": str(powerbi_aligned.index.min()),
                "end": str(powerbi_aligned.index.max()),
                "periods": len(powerbi_aligned),
            },
            "forecast_comparison": comparison_metrics,
        }

        if include_actual and "Valor_Real" in powerbi_df.columns:
            actual_values = powerbi_df["Valor_Real"].dropna()
            if len(actual_values) > 0:
                accuracy_metrics = self.compare_with_actual(
                    actual_values, powerbi_aligned, ml_aligned
                )
                results["accuracy_evaluation"] = accuracy_metrics

        self.comparison_results[model_name] = results

        logger.info(f"Comparison completed for {model_name}")
        return results

    def save_comparison_results(self, output_file: Path) -> None:
        """
        Save comparison results to JSON file.

        Args:
            output_file: Path to save results
        """
        logger.info(f"Saving comparison results to {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "comparison_framework": "TCC Sales Forecasting - ML vs Power BI",
            "generated_at": pd.Timestamp.now().isoformat(),
            "models_compared": list(self.comparison_results.keys()),
            "results": self.comparison_results,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Comparison results saved successfully")

    def generate_comparison_dataframe(self) -> pd.DataFrame:
        """
        Generate comparison summary DataFrame for easy analysis.

        Returns:
            DataFrame with comparison summary
        """
        if not self.comparison_results:
            logger.warning("No comparison results available")
            return pd.DataFrame()

        rows = []
        for model_name, results in self.comparison_results.items():
            row = {"model": model_name}

            if "forecast_comparison" in results:
                fc = results["forecast_comparison"]
                row.update(
                    {
                        "periods": fc["periods_compared"],
                        "total_diff_pct": fc["total_difference_pct"],
                        "mean_pct_diff": fc["mean_percentage_difference"],
                        "correlation": fc["correlation"],
                    }
                )

            if "accuracy_evaluation" in results:
                ae = results["accuracy_evaluation"]
                row.update(
                    {
                        "powerbi_mae": ae["powerbi_metrics"]["mae"],
                        "ml_mae": ae["ml_metrics"]["mae"],
                        "mae_improvement_pct": ae["improvement"]["mae_improvement_pct"],
                        "winner": ae["improvement"]["winner"],
                    }
                )

            rows.append(row)

        return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Power BI comparison framework loaded successfully!")
