"""
Main script for comparing ML forecasts with Power BI baseline.
Generates comprehensive comparison analysis for TCC.
"""

import logging
from pathlib import Path

from src.evaluation.powerbi_comparison import PowerBIComparison
from src.visualization.powerbi_comparison_plots import PowerBIComparisonPlots

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Execute comparison between XGBoost (best ML model) and Power BI.
    """
    logger.info("Starting XGBoost vs Power BI comparison analysis")

    data_dir = Path("data")
    output_dir = Path("data/plots/powerbi_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    powerbi_file = data_dir / "powerbi_forecast.csv"

    if not powerbi_file.exists():
        logger.error(f"Power BI forecast file not found: {powerbi_file}")
        logger.error(
            "Please export Power BI data following instructions in powerbi_export_template.txt"
        )
        return

    comparator = PowerBIComparison(data_dir)
    plotter = PowerBIComparisonPlots(output_dir)

    model_name = "xgboost"

    logger.info(f"Comparing best ML model: {model_name.upper()}")

    powerbi_df = comparator.load_powerbi_forecast(powerbi_file)

    try:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Comparing {model_name.upper()} vs Power BI")
        logger.info(f"{'=' * 60}")

        results = comparator.run_full_comparison(model_name, powerbi_file, include_actual=True)

        ml_series = comparator.ml_forecasts[model_name]

        powerbi_aligned, ml_aligned = comparator.align_forecasts(powerbi_df, ml_series)

        actual_values = None
        if "Valor_Real" in powerbi_df.columns:
            actual_values = powerbi_df.loc[powerbi_aligned.index, "Valor_Real"]
            actual_values = actual_values.dropna()
            if len(actual_values) == 0:
                actual_values = None

        plotter.plot_forecast_comparison(
            dates=powerbi_aligned.index,
            actual=actual_values,
            powerbi=powerbi_aligned,
            ml_forecast=ml_aligned,
            model_name=model_name.upper(),
        )

        plotter.plot_difference_analysis(
            dates=powerbi_aligned.index,
            powerbi=powerbi_aligned,
            ml_forecast=ml_aligned,
            model_name=model_name.upper(),
        )

        logger.info(f"{model_name.upper()} comparison completed successfully")

        logger.info("\n" + "=" * 60)
        logger.info("SAVING COMPARISON RESULTS")
        logger.info("=" * 60)

        comparator.save_comparison_results(output_dir / "xgboost_vs_powerbi_results.json")

        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON ANALYSIS COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info("\nGenerated files:")
        logger.info("  - xgboost_vs_powerbi_forecast_comparison.png")
        logger.info("  - xgboost_vs_powerbi_difference.png")
        logger.info("  - xgboost_vs_powerbi_results.json")

        if "accuracy_evaluation" in results:
            acc = results["accuracy_evaluation"]
            logger.info("\nAccuracy Metrics:")
            logger.info(f"  Power BI MAE: {acc['powerbi_metrics']['mae']:.2f}")
            logger.info(f"  XGBoost MAE: {acc['ml_metrics']['mae']:.2f}")
            logger.info(f"  Improvement: {acc['improvement']['mae_improvement_pct']:.2f}%")
            logger.info(f"  Winner: {acc['improvement']['winner']}")

    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main()
