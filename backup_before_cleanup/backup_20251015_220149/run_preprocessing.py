"""
Run the complete preprocessing pipeline on real data.
This script demonstrates the full preprocessing workflow for the TCC project.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_FILE, create_directories, load_config
from src.preprocessing.data_preprocessor import MLDataPreprocessor
from src.preprocessing.model_formatters import ModelDataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_validation_tests(model_data, data_manager):
    """
    Run validation tests on processed data to ensure quality.

    Args:
        model_data: Dictionary with prepared model data
        data_manager: ModelDataManager instance

    Returns:
        bool: True if all validations pass
    """
    try:
        validation_results = []

        # Test time series functionality
        if "time_series" in model_data:
            logger.info("Validating time series data")
            ts_formatter = data_manager.ts_formatter
            series = model_data["time_series"]["series"]

            # Check series validity
            if len(series) > 0:
                logger.info(f"Time series validation: {len(series)} points")
                validation_results.append(True)

                # Test stationarity check
                try:
                    stationarity_results = ts_formatter.check_stationarity(series)
                    logger.info(
                        f"Stationarity test completed: ADF stationary={stationarity_results.get('adf_stationary', False)}"
                    )
                    validation_results.append(True)
                except Exception as e:
                    logger.warning(f"Stationarity test failed: {e}")
                    validation_results.append(False)

                # Test train/test splits
                try:
                    # Use the ARIMA split method as reference for validation
                    split_point = int(len(series) * 0.8)
                    train_ts = series[:split_point]
                    test_ts = series[split_point:]
                    logger.info(
                        f"Train/test split validation: {len(train_ts)} train, {len(test_ts)} test"
                    )
                    validation_results.append(len(train_ts) > 0 and len(test_ts) > 0)
                except Exception as e:
                    logger.warning(f"Train/test split validation failed: {e}")
                    validation_results.append(False)
            else:
                logger.error("Time series is empty")
                validation_results.append(False)

        # Test XGBoost functionality
        if "xgboost" in model_data:
            logger.info("Validating XGBoost data")
            tabular_formatter = data_manager.tabular_formatter
            features = model_data["xgboost"]["features"]
            target = model_data["xgboost"]["target"]

            # Check data validity
            if len(features) > 0 and len(target) > 0:
                logger.info(
                    f"XGBoost validation: {len(features)} samples, {len(features.columns)} features"
                )
                validation_results.append(True)

                # Test train/test split
                try:
                    X_train, X_test, y_train, y_test = tabular_formatter.create_train_test_split(
                        features, target
                    )
                    logger.info(
                        f"XGBoost split validation: {len(X_train)} train, {len(X_test)} test"
                    )
                    validation_results.append(len(X_train) > 0 and len(X_test) > 0)
                except Exception as e:
                    logger.warning(f"XGBoost split validation failed: {e}")
                    validation_results.append(False)
            else:
                logger.error("XGBoost data is empty")
                validation_results.append(False)

        # Check if Darts integration is working
        try:
            import darts  # noqa: F401

            logger.info("Darts library integration confirmed")
            validation_results.append(True)
        except ImportError as e:
            logger.error(f"Darts library not available: {e}")
            validation_results.append(False)

        # Return overall validation result
        all_passed = all(validation_results)
        passed_count = sum(validation_results)
        total_count = len(validation_results)

        logger.info(f"Validation summary: {passed_count}/{total_count} tests passed")

        return all_passed

    except Exception as e:
        logger.error(f"Validation testing failed: {e}")
        return False


def main():
    """
    Execute the complete preprocessing pipeline.
    """
    logger.info("Starting TCC preprocessing pipeline")

    # Create necessary directories
    create_directories()

    # Load configuration
    config = load_config()

    # Check if raw data file exists
    if not RAW_DATA_FILE.exists():
        logger.error(f"Raw data file not found: {RAW_DATA_FILE}")
        logger.info("Please ensure the data/fat_factory.csv file exists")
        return False

    try:
        # Initialize preprocessor
        logger.info("Initializing data preprocessor")
        preprocessor = MLDataPreprocessor(config)

        # Process data
        logger.info("Running 9-step preprocessing pipeline")
        processed_df = preprocessor.process_data(RAW_DATA_FILE)

        logger.info(f"Preprocessing completed. Final shape: {processed_df.shape}")
        logger.info(f"Features: {list(processed_df.columns)}")

        # Initialize model data manager
        logger.info("Preparing data for all model types")
        data_manager = ModelDataManager(config)

        # Prepare data for all models
        output_dir = PROCESSED_DATA_DIR / "model_data"
        model_data = data_manager.prepare_all_model_data(processed_df, output_dir)

        # Save quality report
        quality_report_file = PROCESSED_DATA_DIR / "data_quality_report.json"
        preprocessor.save_quality_report(quality_report_file)

        # Print summary
        logger.info("=== PREPROCESSING SUMMARY ===")
        logger.info(f"Total records processed: {len(processed_df)}")
        logger.info(f"Total features created: {len(processed_df.columns) - 1}")  # Exclude target

        if "time_series" in model_data:
            ts_data = model_data["time_series"]
            logger.info(f"Time series length: {len(ts_data['series'])} time points")
            logger.info(
                f"Time series range: {ts_data['series'].start_time()} to {ts_data['series'].end_time()}"
            )

        if "xgboost" in model_data:
            xgb_data = model_data["xgboost"]
            logger.info(f"XGBoost features: {len(xgb_data['features'].columns)}")
            logger.info(f"XGBoost samples: {len(xgb_data['features'])}")

        logger.info(f"Data saved to: {output_dir}")
        logger.info(f"Quality report saved to: {quality_report_file}")

        # Run validation tests
        logger.info("Running data validation tests")
        validation_success = run_validation_tests(model_data, data_manager)

        if validation_success:
            logger.info("All validation tests passed!")
        else:
            logger.warning("Some validation tests failed, but preprocessing completed")

        logger.info("Preprocessing pipeline completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Preprocessing failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\nAll preprocessing steps completed successfully!")
        print("Data is ready for model training and evaluation.")
    else:
        print("\nPreprocessing failed. Please check the logs above.")
        sys.exit(1)
