"""
Execute Exponential Smoothing sales forecasting model on preprocessed data.
Complete pipeline for Exponential Smoothing model training, prediction, and evaluation.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import DATA_DIR, PROCESSED_DATA_DIR, create_directories, load_config
from src.models.exponential_smoothing_model import ExponentialSmoothingForecaster

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Execute complete Exponential Smoothing forecasting pipeline.
    """
    logger.info("Starting Exponential Smoothing sales forecasting pipeline")

    # Create necessary directories
    create_directories()

    # Load configuration
    config = load_config()

    # Define data paths
    model_data_dir = PROCESSED_DATA_DIR / "model_data"
    time_series_data_dir = model_data_dir / "time_series"
    exponential_output_dir = PROCESSED_DATA_DIR / "exponential_smoothing"

    # Check if preprocessed data exists
    if not time_series_data_dir.exists():
        logger.error(f"Time series data directory not found: {time_series_data_dir}")
        logger.info("Please run 'uv run run_preprocessing.py' first to generate time series data")
        return False

    # Check for required time series file
    ts_file = time_series_data_dir / "time_series_darts.csv"
    if not ts_file.exists():
        logger.error(f"Time series file not found: {ts_file}")
        logger.info("Please ensure preprocessing generated the required time series data")
        return False

    try:
        # Initialize Exponential Smoothing forecaster
        logger.info("Initializing Exponential Smoothing forecaster")
        forecaster = ExponentialSmoothingForecaster(config)

        # Run complete forecasting pipeline
        logger.info("Running complete Exponential Smoothing forecasting pipeline")
        results = forecaster.run_complete_pipeline(
            data_path=time_series_data_dir, output_dir=exponential_output_dir
        )

        if results["success"]:
            # Print comprehensive results
            logger.info("=== EXPONENTIAL SMOOTHING FORECASTING RESULTS ===")

            # Trend/Seasonality analysis results
            trend_seasonality = results["trend_seasonality"]
            logger.info("Trend and Seasonality Analysis:")
            logger.info(f"  Trend strength: {trend_seasonality['trend_strength']:.4f}")
            logger.info(f"  Seasonal strength: {trend_seasonality['seasonal_strength']:.4f}")
            logger.info(f"  Trend type: {trend_seasonality['trend_type']}")
            logger.info(f"  Has seasonality: {trend_seasonality['has_seasonality']}")
            logger.info(f"  Recommended model: {trend_seasonality['recommended_model']}")

            # Model performance
            metrics = results["metrics"]
            logger.info("Model Performance:")
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")

            # Model summary
            summary = results["model_summary"]
            logger.info("Model Summary:")
            logger.info(f"  Training samples: {summary['train_samples']}")
            logger.info(f"  Test samples: {summary['test_samples']}")

            # Output locations
            logger.info("Output Files:")
            logger.info(f"  Results directory: {exponential_output_dir}")
            logger.info(f"  Predictions: {exponential_output_dir / 'exponential_predictions.csv'}")
            logger.info(f"  Metrics: {exponential_output_dir / 'exponential_metrics.json'}")
            logger.info(
                f"  Model summary: {exponential_output_dir / 'exponential_model_summary.json'}"
            )

            # Plot information
            if "plots" in results and results["plots"]["plots_generated"]:
                plots_dir = DATA_DIR / "plots" / "exponential_smoothing"
                logger.info("Generated Plots:")
                logger.info("  EDA plots: Time series overview, decomposition, seasonality")
                logger.info("  Prediction plots: Comparison, residuals, performance metrics")
                logger.info("  Analysis plots: Trend/seasonality analysis, model summary")
                logger.info(f"  Plots directory: {plots_dir}")

            logger.info("Exponential Smoothing forecasting completed successfully!")

            return True

        else:
            logger.error(f"Exponential Smoothing forecasting failed: {results['error']}")
            return False

    except Exception as e:
        logger.error(f"Exponential Smoothing forecasting failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_exponential_summary():
    """
    Print summary of available Exponential Smoothing results.
    """
    exponential_output_dir = PROCESSED_DATA_DIR / "exponential_smoothing"

    if not exponential_output_dir.exists():
        print(
            "No Exponential Smoothing results found. Run 'uv run run_exponential_forecasting.py' first."
        )
        return

    print("=== EXPONENTIAL SMOOTHING FORECASTING SUMMARY ===")

    # Check for results files
    predictions_file = exponential_output_dir / "exponential_predictions.csv"
    metrics_file = exponential_output_dir / "exponential_metrics.json"
    summary_file = exponential_output_dir / "exponential_model_summary.json"

    if predictions_file.exists():
        print(f"Predictions: {predictions_file}")

        # Load and show prediction preview
        import pandas as pd

        predictions = pd.read_csv(predictions_file, index_col=0, parse_dates=True)
        print(f"   Forecast period: {predictions.index.min()} to {predictions.index.max()}")
        print(f"   Forecast points: {len(predictions)}")

    if metrics_file.exists():
        print(f"Metrics: {metrics_file}")

        # Load and show metrics
        import json

        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")

    if summary_file.exists():
        print(f"Model Summary: {summary_file}")

        # Load and show model info
        import json

        with open(summary_file, "r") as f:
            summary = json.load(f)
        print(f"   Model type: {summary['model_type']}")
        print(f"   Training samples: {summary['train_samples']}")
        print(f"   Test samples: {summary['test_samples']}")

        # Show trend/seasonality analysis if available
        if "trend_seasonality_analysis" in summary:
            trend_info = summary["trend_seasonality_analysis"]
            print(f"   Trend type: {trend_info.get('trend_type', 'Unknown')}")
            print(f"   Has seasonality: {trend_info.get('has_seasonality', 'Unknown')}")
            print(f"   Recommended model: {trend_info.get('recommended_model', 'Unknown')}")

    print(f"\nResults directory: {exponential_output_dir}")


if __name__ == "__main__":
    # Check if user wants summary instead of running
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_exponential_summary()
    else:
        success = main()
        if success:
            print("\nExponential Smoothing forecasting completed successfully!")
            print("Results saved and ready for analysis.")
            print("\nTo view summary: uv run run_exponential_forecasting.py --summary")
        else:
            print("\nExponential Smoothing forecasting failed. Please check the logs above.")
            sys.exit(1)
