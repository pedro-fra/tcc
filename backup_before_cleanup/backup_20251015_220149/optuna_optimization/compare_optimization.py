"""
Comparison script to evaluate models before and after Optuna optimization.
Tests baseline models with default parameters vs optimized models.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))

from config import PROCESSED_DATA_DIR, load_config
from models.arima_model import ArimaForecaster
from models.exponential_smoothing_model import ExponentialSmoothingForecaster
from models.theta import ThetaForecaster
from models.xgboost_model import XGBoostForecaster
from optimization import (
    ARIMAOptimizer,
    ExponentialSmoothingOptimizer,
    ThetaOptimizer,
    XGBoostOptimizer,
)


def test_baseline_model(model_name: str, data_path: Path, config: dict) -> dict:
    """
    Test model with default/baseline parameters.

    Args:
        model_name: Name of the model
        data_path: Path to data file
        config: Configuration dictionary

    Returns:
        Dictionary with baseline results
    """
    print(f"\n{'=' * 80}")
    print(f"Testing BASELINE {model_name.upper()} model")
    print(f"{'=' * 80}")

    start_time = time.time()

    if model_name == "xgboost":
        model = XGBoostForecaster(config)
        results = model.run_complete_pipeline(str(data_path))
    elif model_name == "arima":
        model = ArimaForecaster(config)
        results = model.run_complete_pipeline(str(data_path))
    elif model_name == "theta":
        model = ThetaForecaster(config)
        results = model.run_complete_pipeline(str(data_path))
    elif model_name == "exponential_smoothing":
        model = ExponentialSmoothingForecaster(config)
        results = model.run_complete_pipeline(str(data_path))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    elapsed_time = time.time() - start_time

    metrics = results.get("metrics", {})

    baseline_results = {
        "model_name": model_name,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
        "training_time": elapsed_time,
        "parameters": "default",
    }

    print("\nBaseline Results:")
    print(f"  MAE:  {baseline_results['mae']:.2f}")
    print(f"  RMSE: {baseline_results['rmse']:.2f}")
    print(f"  MAPE: {baseline_results['mape']:.2f}%")
    print(f"  Time: {baseline_results['training_time']:.2f}s")

    return baseline_results


def optimize_and_test_model(
    model_name: str, data_path: Path, config: dict, n_trials: int = 20
) -> dict:
    """
    Optimize model with Optuna and test with best parameters.

    Args:
        model_name: Name of the model
        data_path: Path to data file
        config: Configuration dictionary
        n_trials: Number of optimization trials

    Returns:
        Dictionary with optimized results
    """
    print(f"\n{'=' * 80}")
    print(f"Optimizing {model_name.upper()} model with Optuna")
    print(f"{'=' * 80}")

    # Initialize optimizer
    if model_name == "xgboost":
        optimizer = XGBoostOptimizer(config, data_path)
    elif model_name == "arima":
        optimizer = ARIMAOptimizer(config, data_path)
    elif model_name == "theta":
        optimizer = ThetaOptimizer(config, data_path)
    elif model_name == "exponential_smoothing":
        optimizer = ExponentialSmoothingOptimizer(config, data_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Create study
    optimizer.create_study(study_name=f"{model_name}_comparison_test")

    # Run optimization
    print(f"\nRunning optimization with {n_trials} trials...")
    start_time = time.time()
    opt_results = optimizer.optimize(n_trials=n_trials, show_progress=True)
    optimization_time = time.time() - start_time

    print(f"\nOptimization completed in {optimization_time:.2f}s")
    print(f"Best MAE: {opt_results['best_value']:.2f}")
    print(f"Best parameters: {opt_results['best_params']}")

    # Save best parameters
    output_dir = PROCESSED_DATA_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_optimized_params.json"
    optimizer.save_best_params(output_path)

    # Get parameter importance
    importance = optimizer.get_param_importance()
    if importance:
        print("\nParameter Importance:")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {param}: {score:.4f}")

    # Test with optimized parameters
    print(f"\n{'=' * 80}")
    print(f"Testing OPTIMIZED {model_name.upper()} model")
    print(f"{'=' * 80}")

    # Update config with best parameters
    optimized_config = config.copy()
    optimized_config[model_name].update(opt_results["best_params"])

    start_time = time.time()

    if model_name == "xgboost":
        model = XGBoostForecaster(optimized_config)
        results = model.run_complete_pipeline(str(data_path))
    elif model_name == "arima":
        model = ArimaForecaster(optimized_config)
        results = model.run_complete_pipeline(str(data_path))
    elif model_name == "theta":
        model = ThetaForecaster(optimized_config)
        results = model.run_complete_pipeline(str(data_path))
    elif model_name == "exponential_smoothing":
        model = ExponentialSmoothingForecaster(optimized_config)
        results = model.run_complete_pipeline(str(data_path))

    training_time = time.time() - start_time

    metrics = results.get("metrics", {})

    optimized_results = {
        "model_name": model_name,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
        "training_time": training_time,
        "optimization_time": optimization_time,
        "n_trials": n_trials,
        "parameters": opt_results["best_params"],
    }

    print("\nOptimized Results:")
    print(f"  MAE:  {optimized_results['mae']:.2f}")
    print(f"  RMSE: {optimized_results['rmse']:.2f}")
    print(f"  MAPE: {optimized_results['mape']:.2f}%")
    print(f"  Time: {optimized_results['training_time']:.2f}s")

    return optimized_results


def compare_results(baseline: dict, optimized: dict) -> dict:
    """
    Compare baseline vs optimized results.

    Args:
        baseline: Baseline results dictionary
        optimized: Optimized results dictionary

    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "model_name": baseline["model_name"],
        "mae_improvement": ((baseline["mae"] - optimized["mae"]) / baseline["mae"]) * 100,
        "rmse_improvement": ((baseline["rmse"] - optimized["rmse"]) / baseline["rmse"]) * 100,
        "mape_improvement": ((baseline["mape"] - optimized["mape"]) / baseline["mape"]) * 100,
        "baseline_mae": baseline["mae"],
        "optimized_mae": optimized["mae"],
        "baseline_rmse": baseline["rmse"],
        "optimized_rmse": optimized["rmse"],
        "baseline_mape": baseline["mape"],
        "optimized_mape": optimized["mape"],
        "optimization_overhead": optimized.get("optimization_time", 0),
    }

    return comparison


def print_comparison_summary(comparisons: list):
    """
    Print summary table of all comparisons.

    Args:
        comparisons: List of comparison dictionaries
    """
    print("\n" + "=" * 100)
    print("OPTIMIZATION COMPARISON SUMMARY")
    print("=" * 100)

    # Create DataFrame for better formatting
    df = pd.DataFrame(comparisons)

    print("\n--- Baseline vs Optimized Metrics ---")
    for _, row in df.iterrows():
        print(f"\n{row['model_name'].upper()}:")
        print(
            f"  MAE:  {row['baseline_mae']:>10.2f} -> {row['optimized_mae']:>10.2f} "
            f"({row['mae_improvement']:>+6.2f}%)"
        )
        print(
            f"  RMSE: {row['baseline_rmse']:>10.2f} -> {row['optimized_rmse']:>10.2f} "
            f"({row['rmse_improvement']:>+6.2f}%)"
        )
        print(
            f"  MAPE: {row['baseline_mape']:>10.2f} -> {row['optimized_mape']:>10.2f} "
            f"({row['mape_improvement']:>+6.2f}%)"
        )
        print(f"  Optimization time: {row['optimization_overhead']:.2f}s")

    print("\n--- Improvement Summary ---")
    avg_mae_imp = df["mae_improvement"].mean()
    avg_rmse_imp = df["rmse_improvement"].mean()
    avg_mape_imp = df["mape_improvement"].mean()

    print(f"Average MAE improvement:  {avg_mae_imp:>+6.2f}%")
    print(f"Average RMSE improvement: {avg_rmse_imp:>+6.2f}%")
    print(f"Average MAPE improvement: {avg_mape_imp:>+6.2f}%")

    # Find best model
    best_model_idx = df["optimized_mae"].idxmin()
    best_model = df.loc[best_model_idx]

    print("\n--- Best Performing Model ---")
    print(f"{best_model['model_name'].upper()}")
    print(f"  Optimized MAE: {best_model['optimized_mae']:.2f}")
    print(f"  Improvement: {best_model['mae_improvement']:+.2f}%")

    print("\n" + "=" * 100)

    # Save to CSV
    output_path = Path("optimization_comparison_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    """
    Main function to run complete comparison.
    """
    print("=" * 100)
    print("MODEL OPTIMIZATION COMPARISON")
    print("Testing baseline models vs Optuna-optimized models")
    print("=" * 100)

    # Load configuration
    config = load_config()
    data_path = Path("data/fat_factory.csv")

    # Models to test (start with XGBoost as it's fastest)
    models = ["xgboost"]  # Can add: "arima", "theta", "exponential_smoothing"
    n_trials = 20  # Reduced for faster testing

    all_comparisons = []

    for model_name in models:
        try:
            # Test baseline
            baseline_results = test_baseline_model(model_name, data_path, config)

            # Optimize and test
            optimized_results = optimize_and_test_model(
                model_name, data_path, config, n_trials=n_trials
            )

            # Compare
            comparison = compare_results(baseline_results, optimized_results)
            all_comparisons.append(comparison)

            print(f"\n{model_name.upper()} Comparison:")
            print(f"  MAE Improvement: {comparison['mae_improvement']:+.2f}%")
            print(f"  RMSE Improvement: {comparison['rmse_improvement']:+.2f}%")
            print(f"  MAPE Improvement: {comparison['mape_improvement']:+.2f}%")

        except Exception as e:
            print(f"\nError testing {model_name}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    if all_comparisons:
        print_comparison_summary(all_comparisons)

        # Save detailed results
        results_file = Path("optimization_comparison_detailed.json")
        with open(results_file, "w") as f:
            json.dump(all_comparisons, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
