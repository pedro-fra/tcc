"""
Test improved optimization with adjusted parameters.
Tests XGBoost with:
- Consistent train/test split
- Reduced search space
- CV temporal
- More trials
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from config import load_config
from models.xgboost_model import XGBoostForecaster
from optimization import XGBoostOptimizer


def test_improved_optimization():
    """Test XGBoost with improved optimization strategy."""
    print("=" * 100)
    print("Testing IMPROVED XGBoost Optimization")
    print("=" * 100)

    # Load configuration
    config = load_config()
    data_path = Path("data/fat_factory.csv")

    # Test 1: Baseline with default parameters
    print("\n" + "=" * 100)
    print("STEP 1: Baseline Model (Default Parameters)")
    print("=" * 100)

    model_baseline = XGBoostForecaster(config)
    results_baseline = model_baseline.run_complete_pipeline(str(data_path))
    metrics_baseline = results_baseline["metrics"]

    print("\nBaseline Results:")
    print(f"  MAE:  {metrics_baseline['mae']:,.2f}")
    print(f"  RMSE: {metrics_baseline['rmse']:,.2f}")
    print(f"  MAPE: {metrics_baseline['mape']:.2f}%")

    # Test 2: Optimization with CV
    print("\n" + "=" * 100)
    print("STEP 2: Optuna Optimization with CV (50 trials)")
    print("=" * 100)

    optimizer = XGBoostOptimizer(config, data_path)
    optimizer.create_study(study_name="xgboost_improved_test")

    print("\nRunning optimization with Time-Series CV...")
    print("- Focused search space (important parameters)")
    print("- Consistent train/test split")
    print("- Automatic CV fold selection")

    # Use CV for more robust optimization
    opt_results = optimizer.optimize_with_cv(n_trials=50, n_splits=None)

    print("\n✓ Optimization completed!")
    print(f"  Best CV MAE: {opt_results['best_value']:,.2f}")
    print(f"  Number of trials: {opt_results['n_trials']}")
    print("\nBest Parameters Found:")
    for param, value in opt_results["best_params"].items():
        print(f"  {param}: {value}")

    # Get parameter importance
    importance = optimizer.get_param_importance()
    if importance:
        print("\nParameter Importance (Top 5):")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {param}: {score:.2%}")

    # Test 3: Test optimized model
    print("\n" + "=" * 100)
    print("STEP 3: Testing Optimized Model")
    print("=" * 100)

    optimized_config = config.copy()
    optimized_config["xgboost"].update(opt_results["best_params"])

    model_optimized = XGBoostForecaster(optimized_config)
    results_optimized = model_optimized.run_complete_pipeline(str(data_path))
    metrics_optimized = results_optimized["metrics"]

    print("\nOptimized Model Results:")
    print(f"  MAE:  {metrics_optimized['mae']:,.2f}")
    print(f"  RMSE: {metrics_optimized['rmse']:,.2f}")
    print(f"  MAPE: {metrics_optimized['mape']:.2f}%")

    # Calculate improvements
    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    mae_improvement = (
        (metrics_baseline["mae"] - metrics_optimized["mae"]) / metrics_baseline["mae"]
    ) * 100
    rmse_improvement = (
        (metrics_baseline["rmse"] - metrics_optimized["rmse"]) / metrics_baseline["rmse"]
    ) * 100
    mape_improvement = (
        (metrics_baseline["mape"] - metrics_optimized["mape"]) / metrics_baseline["mape"]
    ) * 100

    print("\nBaseline vs Optimized:")
    print(
        f"  MAE:  {metrics_baseline['mae']:>12,.2f} -> {metrics_optimized['mae']:>12,.2f} "
        f"({mae_improvement:>+6.2f}%)"
    )
    print(
        f"  RMSE: {metrics_baseline['rmse']:>12,.2f} -> {metrics_optimized['rmse']:>12,.2f} "
        f"({rmse_improvement:>+6.2f}%)"
    )
    print(
        f"  MAPE: {metrics_baseline['mape']:>12.2f} -> {metrics_optimized['mape']:>12.2f} "
        f"({mape_improvement:>+6.2f}%)"
    )

    if mae_improvement > 0:
        print(f"\n✓ SUCCESS: Model improved by {mae_improvement:.2f}% in MAE!")
    else:
        print(f"\n✗ WARNING: Model worsened by {abs(mae_improvement):.2f}% in MAE")
        print("  Consider:")
        print("  - More trials (100-200)")
        print("  - Different search space")
        print("  - Collect more historical data")

    print("\n" + "=" * 100)

    # Save results
    import json

    results_summary = {
        "baseline": {
            "mae": float(metrics_baseline["mae"]),
            "rmse": float(metrics_baseline["rmse"]),
            "mape": float(metrics_baseline["mape"]),
        },
        "optimized": {
            "mae": float(metrics_optimized["mae"]),
            "rmse": float(metrics_optimized["rmse"]),
            "mape": float(metrics_optimized["mape"]),
            "best_params": opt_results["best_params"],
        },
        "improvement": {
            "mae_pct": mae_improvement,
            "rmse_pct": rmse_improvement,
            "mape_pct": mape_improvement,
        },
    }

    output_file = Path("improved_optimization_results.json")
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results_summary


if __name__ == "__main__":
    test_improved_optimization()
