"""
Main script for running hyperparameter optimization for all models.
"""

import argparse
import logging
from pathlib import Path

from config import OPTUNA_CONFIG, PROCESSED_DATA_DIR, load_config
from optimization import (
    ARIMAOptimizer,
    ExponentialSmoothingOptimizer,
    ThetaOptimizer,
    XGBoostOptimizer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def optimize_model(
    model_name: str,
    data_path: Path,
    n_trials: int = None,
    timeout: int = None,
    use_cv: bool = False,
):
    """
    Run optimization for a specific model.

    Args:
        model_name: Name of model to optimize (arima, theta, exponential_smoothing, xgboost)
        data_path: Path to input data
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        use_cv: Use cross-validation for optimization

    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Starting optimization for {model_name.upper()}")

    # Load configuration
    config = load_config()

    # Initialize appropriate optimizer
    if model_name == "arima":
        optimizer = ARIMAOptimizer(config, data_path)
    elif model_name == "theta":
        optimizer = ThetaOptimizer(config, data_path)
    elif model_name == "exponential_smoothing":
        optimizer = ExponentialSmoothingOptimizer(config, data_path)
    elif model_name == "xgboost":
        optimizer = XGBoostOptimizer(config, data_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Create study
    optimizer.create_study()

    # Run optimization
    if use_cv and hasattr(optimizer, "optimize_with_cv"):
        logger.info("Running optimization with cross-validation")
        results = optimizer.optimize_with_cv(n_trials=n_trials, timeout=timeout)
    elif use_cv and hasattr(optimizer, "optimize_with_walk_forward"):
        logger.info("Running optimization with walk-forward validation")
        results = optimizer.optimize_with_walk_forward(n_trials=n_trials, timeout=timeout)
    else:
        logger.info("Running standard optimization")
        results = optimizer.optimize(n_trials=n_trials, timeout=timeout)

    # Save best parameters
    output_dir = PROCESSED_DATA_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_best_params.json"
    optimizer.save_best_params(output_path)

    # Get optimization history
    history = optimizer.get_optimization_history()
    logger.info(f"Optimization history: {history}")

    # Get parameter importance
    importance = optimizer.get_param_importance()
    if importance:
        logger.info("Parameter importance:")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {param}: {score:.4f}")

    logger.info(f"Optimization completed for {model_name.upper()}")
    logger.info(f"Best value: {results['best_value']:.4f}")
    logger.info(f"Best parameters: {results['best_params']}")

    return results


def optimize_all_models(
    data_path: Path,
    models: list = None,
    n_trials: int = None,
    timeout: int = None,
    use_cv: bool = False,
):
    """
    Run optimization for all models sequentially.

    Args:
        data_path: Path to input data
        models: List of model names to optimize (default: all models)
        n_trials: Number of optimization trials per model
        timeout: Timeout in seconds per model
        use_cv: Use cross-validation for optimization

    Returns:
        Dictionary with results for all models
    """
    if models is None:
        models = ["xgboost", "arima", "theta", "exponential_smoothing"]

    all_results = {}

    for model_name in models:
        try:
            results = optimize_model(
                model_name=model_name,
                data_path=data_path,
                n_trials=n_trials,
                timeout=timeout,
                use_cv=use_cv,
            )
            all_results[model_name] = results
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
            all_results[model_name] = {"error": str(e)}

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 80)

    for model_name, results in all_results.items():
        if "error" in results:
            logger.info(f"{model_name.upper()}: FAILED - {results['error']}")
        else:
            logger.info(
                f"{model_name.upper()}: "
                f"Best MAE = {results['best_value']:.4f} "
                f"({results['n_trials']} trials)"
            )

    logger.info("=" * 80)

    return all_results


def main():
    """
    Main function for running optimization from command line.
    """
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for forecasting models"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["arima", "theta", "exponential_smoothing", "xgboost", "all"],
        default="all",
        help="Model to optimize (default: all)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/fat_factory.csv",
        help="Path to input data file",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=f"Number of optimization trials (default: {OPTUNA_CONFIG['n_trials']})",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=f"Timeout in seconds (default: {OPTUNA_CONFIG['timeout']})",
    )

    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use cross-validation for optimization",
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Optuna Dashboard after optimization",
    )

    args = parser.parse_args()

    # Convert data path to Path object
    data_path = Path(args.data)

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Run optimization
    if args.model == "all":
        optimize_all_models(
            data_path=data_path,
            n_trials=args.n_trials,
            timeout=args.timeout,
            use_cv=args.cv,
        )
    else:
        optimize_model(
            model_name=args.model,
            data_path=data_path,
            n_trials=args.n_trials,
            timeout=args.timeout,
            use_cv=args.cv,
        )

    # Launch dashboard if requested
    if args.dashboard:
        logger.info("Launching Optuna Dashboard...")
        logger.info(f"Storage: {OPTUNA_CONFIG['storage_url']}")
        logger.info("Run: optuna-dashboard sqlite:///optuna_studies.db")


if __name__ == "__main__":
    main()
