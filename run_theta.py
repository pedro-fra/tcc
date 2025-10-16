"""
Execute Theta sales forecasting model on preprocessed data.
Complete pipeline for Theta model training, prediction, and evaluation.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import DATA_DIR, PROCESSED_DATA_DIR, create_directories, load_config
from src.models.theta import ThetaForecaster

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Execute complete Theta forecasting pipeline.
    """
    logger.info("Starting Theta sales forecasting pipeline")

    # Create necessary directories
    create_directories()

    # Load configuration
    config = load_config()

    # Define data paths
    model_data_dir = PROCESSED_DATA_DIR / "model_data"
    time_series_data_dir = model_data_dir / "time_series"
    theta_output_dir = PROCESSED_DATA_DIR / "theta"

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
        # Initialize Theta forecaster
        logger.info("Initializing Theta forecaster")
        forecaster = ThetaForecaster(config)

        # Load data directly
        logger.info("Loading time series data")
        raw_data_file = DATA_DIR / "fat_factory.csv"

        if not raw_data_file.exists():
            logger.error(f"Raw data file not found: {raw_data_file}")
            return False

        # Run complete forecasting pipeline
        logger.info("Running complete Theta forecasting pipeline")
        results = forecaster.run_complete_pipeline(str(raw_data_file))

        if results:
            # Print comprehensive results
            logger.info("=== THETA FORECASTING RESULTS ===")

            # Data information
            data_info = results["data_info"]
            logger.info("Data Information:")
            logger.info(f"  Total periods: {data_info['total_periods']}")
            logger.info(f"  Training periods: {data_info['train_periods']}")
            logger.info(f"  Test periods: {data_info['test_periods']}")
            logger.info(f"  Time range: {data_info['time_range']}")

            # Model performance
            metrics = results["metrics"]
            logger.info("Model Performance:")
            logger.info(f"  MAE: {metrics['mae']:.4f}")
            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")

            # Output locations
            logger.info("Output Files:")
            logger.info(f"  Results directory: {theta_output_dir}")
            logger.info(f"  Model: {theta_output_dir / 'theta_model.pkl'}")
            logger.info(f"  Model summary: {theta_output_dir / 'theta_model_summary.json'}")

            # Plot information
            if results.get("plots_generated", False):
                plots_dir = DATA_DIR / "plots" / "theta"
                logger.info("Generated Plots:")
                logger.info("  EDA plots: Time series overview and analysis")
                logger.info("  Prediction plots: Comparison with actual values")
                logger.info("  Performance plots: Metrics and residual analysis")
                logger.info(f"  Plots directory: {plots_dir}")

            logger.info("Theta forecasting completed successfully!")

            return True

        else:
            logger.error("Theta forecasting failed")
            return False

    except Exception as e:
        logger.error(f"Theta forecasting failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def print_theta_summary():
    """
    Print summary of available Theta results.
    """
    theta_output_dir = PROCESSED_DATA_DIR / "theta"

    if not theta_output_dir.exists():
        print("No Theta results found. Run 'uv run run_theta.py' first.")
        return

    print("=== THETA FORECASTING SUMMARY ===")

    # Check for results files
    model_file = theta_output_dir / "theta_model.pkl"
    summary_file = theta_output_dir / "theta_model_summary.json"

    if model_file.exists():
        print(f"Model: {model_file}")

    if summary_file.exists():
        print(f"Model Summary: {summary_file}")

        # Load and show model info
        import json

        with open(summary_file, "r") as f:
            summary = json.load(f)

        print(f"   Model type: {summary['model_type']}")
        print(f"   Training samples: {summary['training_period']['length']}")
        print(f"   Test samples: {summary['test_period']['length']}")

        if "evaluation_metrics" in summary:
            metrics = summary["evaluation_metrics"]
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAPE: {metrics['mape']:.2f}%")

    print(f"\nResults directory: {theta_output_dir}")


if __name__ == "__main__":
    # Check if user wants summary instead of running
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_theta_summary()
    else:
        success = main()
        if success:
            print("\nTheta forecasting completed successfully!")
            print("Results saved and ready for analysis.")
            print("\nTo view summary: uv run run_theta.py --summary")
        else:
            print("\nTheta forecasting failed. Please check the logs above.")
            sys.exit(1)
