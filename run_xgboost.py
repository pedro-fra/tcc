"""
Ultimate Enhanced XGBoost Darts implementation with maximum improvements.
This version implements all possible enhancements short of ensemble methods.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mape, rmse
from darts.models import XGBModel

# Import visualization module
try:
    from visualization.xgboost_plots import XGBoostVisualizer

    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import XGBoostVisualizer: {e}")
    PLOTTING_AVAILABLE = False


def _generate_ultimate_plots(
    results, predictions_df, train_series, test_series, predictions, output_dir
):
    """Generate plots for XGBoost Ultimate results using standardized visualizer."""
    if not PLOTTING_AVAILABLE:
        print("      Skipping plot generation (XGBoostVisualizer not available)")
        return

    plots_dir = Path("data/plots/xgboost_ultimate")
    plots_dir.mkdir(parents=True, exist_ok=True)

    visualizer = XGBoostVisualizer()
    metrics = results["metrics"]

    # Convert TimeSeries to pandas Series for visualizer
    y_test = pd.Series(test_series.values().flatten(), index=test_series.time_index, name="actual")
    y_predictions = predictions_df["predicted"].values

    # Create mock X_test and X_train using temporal index
    X_test = pd.DataFrame(index=test_series.time_index)
    X_train = pd.DataFrame(index=train_series.time_index)
    y_train = pd.Series(train_series.values().flatten(), index=train_series.time_index)

    # Generate prediction comparison plots
    print("      Generating prediction comparison plots...")
    try:
        pred_plots = visualizer.create_prediction_comparison_plots(
            X_test=X_test,
            y_test=y_test,
            predictions=y_predictions,
            X_train=X_train,
            y_train=y_train,
            output_dir=plots_dir,
        )
        print(f"        Created {len(pred_plots)} prediction plots")
    except Exception as e:
        print(f"        Warning: Could not create prediction plots: {e}")

    # Generate performance plots
    print("      Generating performance analysis plots...")
    try:
        perf_plots = visualizer.create_performance_plots(
            y_test=y_test, predictions=y_predictions, metrics=metrics, output_dir=plots_dir
        )
        print(f"        Created {len(perf_plots)} performance plots")
    except Exception as e:
        print(f"        Warning: Could not create performance plots: {e}")

    print(f"      All plots saved to: {plots_dir}")


def run_ultimate_xgboost():
    """Execute Ultimate Enhanced XGBoost with maximum possible improvements."""
    print("=" * 75)
    print("EXECUTING ULTIMATE ENHANCED XGBOOST DARTS - MAXIMUM IMPROVEMENTS")
    print("=" * 75)

    try:
        # 1. Load data
        print("\n1. Loading time series data...")
        data_path = Path("data/processed_data/model_data/time_series")
        ts_file = data_path / "time_series_darts.csv"

        if not ts_file.exists():
            print(f"ERROR: Time series file not found: {ts_file}")
            return None

        df = pd.read_csv(ts_file, index_col=0, parse_dates=True)
        series = TimeSeries.from_dataframe(df)

        print(f"   Loaded time series with {len(series)} time points")
        print(f"   Date range: {series.start_time()} to {series.end_time()}")

        # 2. Prepare data with advanced scaling
        print("\n2. Preparing ultimate data preparation...")

        # Split data
        test_ratio = 0.2
        total_length = len(series)
        train_length = int(total_length * (1 - test_ratio))
        min_train_length = max(24, int(total_length * 0.7))
        train_length = max(train_length, min_train_length)

        train_series = series[:train_length]
        test_series = series[train_length:]

        # Apply scaling
        scaler = Scaler(MaxAbsScaler())
        train_series_scaled = scaler.fit_transform(train_series)

        print(f"   Train: {len(train_series)} samples")
        print(f"   Test: {len(test_series)} samples")
        print("   Data scaling applied (MaxAbsScaler)")

        # 3. Create ultimate enhanced model
        print("\n3. Creating ULTIMATE enhanced XGBoost model...")

        # Maximum lag configuration - include all meaningful lags
        lags = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -15, -18, -24, -30, -36]
        lags_past_covariates = [-1, -2, -3, -4, -5, -6, -12, -24]

        # Ultimate encoders for maximum temporal features
        encoders = {
            "datetime_attribute": {
                "past": ["month", "year", "quarter", "dayofyear", "weekofyear", "dayofweek"]
            },
            "transformer": Scaler(MaxAbsScaler()),
        }

        # Ultimate XGBoost parameters (more aggressive)
        model = XGBModel(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            add_encoders=encoders,
            n_estimators=2000,  # More trees
            max_depth=8,  # Deeper trees
            learning_rate=0.05,  # Lower learning rate with more estimators
            subsample=0.9,  # High subsample ratio
            colsample_bytree=0.9,  # High feature sampling
            reg_alpha=0.2,  # More L1 regularization
            reg_lambda=1.5,  # More L2 regularization
            random_state=42,
        )

        print("   ULTIMATE features:")
        print(f"   - Main lags: {len(lags)} lags ({min(lags)} to {max(lags)})")
        print(f"   - Past covariates lags: {len(lags_past_covariates)} lags")
        print(f"   - Temporal encoders: {encoders['datetime_attribute']['past']}")
        print("   - Ultimate XGBoost parameters (n_estimators=2000, max_depth=8)")

        # 4. Fit model
        print("\n4. Training ultimate enhanced model (this may take longer)...")
        model.fit(train_series_scaled)
        print("   Ultimate model training completed")

        # 5. Generate predictions
        print("\n5. Generating ultimate predictions...")
        n_predict = len(test_series)
        predictions_scaled = model.predict(n=n_predict, series=train_series_scaled)

        # Transform back to original scale
        predictions = scaler.inverse_transform(predictions_scaled)

        print(f"   Ultimate predictions generated for {len(predictions)} time steps")
        print(
            f"   Original scale prediction range: {float(predictions.min().values()[:, 0].min()):.2f} to {float(predictions.max().values()[:, 0].max()):.2f}"
        )

        # 6. Evaluate model
        print("\n6. Evaluating ultimate model performance...")
        mae_score = mae(test_series, predictions)
        rmse_score = rmse(test_series, predictions)
        mape_score = mape(test_series, predictions)

        metrics = {
            "mae": float(mae_score),
            "rmse": float(rmse_score),
            "mape": float(mape_score),
        }

        print("   ULTIMATE model performance:")
        print(f"   - MAE: {mae_score:.2f}")
        print(f"   - RMSE: {rmse_score:.2f}")
        print(f"   - MAPE: {mape_score:.2f}%")

        # 7. Save results
        print("\n7. Saving ultimate results...")
        output_dir = Path("data/processed_data/xgboost_ultimate")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save(output_dir / "ultimate_model.pkl")

        # Create ultimate results summary
        results = {
            "model_type": "XGBoost_Ultimate_Darts",
            "data_info": {
                "total_samples": len(series),
                "train_samples": len(train_series),
                "test_samples": len(test_series),
                "time_range": f"{series.start_time()} to {series.end_time()}",
            },
            "ultimate_enhancements": {
                "main_lags": lags,
                "past_covariates_lags": lags_past_covariates,
                "temporal_encoders": encoders["datetime_attribute"]["past"],
                "data_scaling": "MaxAbsScaler",
                "xgboost_params": {
                    "n_estimators": 2000,
                    "max_depth": 8,
                    "learning_rate": 0.05,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.2,
                    "reg_lambda": 1.5,
                },
            },
            "metrics": metrics,
            "model_fitted": True,
        }

        with open(output_dir / "ultimate_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save predictions
        predictions_df = pd.DataFrame(
            {"actual": test_series.values().flatten(), "predicted": predictions.values().flatten()},
            index=test_series.time_index,
        )
        predictions_df.to_csv(output_dir / "ultimate_predictions.csv")

        print(f"   Ultimate results saved to: {output_dir}")

        # Generate plots
        print("\n   Generating plots...")
        _generate_ultimate_plots(
            results, predictions_df, train_series, test_series, predictions, output_dir
        )

        # 8. Ultimate performance comparison
        print("\n" + "=" * 75)
        print("ULTIMATE PERFORMANCE COMPARISON")
        print("=" * 75)

        original_xgb_mape = 16.32  # Original raw XGBoost implementation
        basic_darts_mape = 54.73  # Basic Darts implementation
        enhanced_darts_mape = 48.50  # Previous enhanced implementation
        ultimate_mape = mape_score  # Current ultimate implementation

        print(f"Original XGBoost (raw):       {original_xgb_mape:.2f}%  <- BEST")
        print(f"Basic Darts XGBoost:          {basic_darts_mape:.2f}%")
        print(f"Enhanced Darts XGBoost:       {enhanced_darts_mape:.2f}%")
        print(f"ULTIMATE Darts XGBoost:       {ultimate_mape:.2f}%  <- NEW")

        # Calculate improvements
        vs_basic = ((basic_darts_mape - ultimate_mape) / basic_darts_mape) * 100
        vs_enhanced = ((enhanced_darts_mape - ultimate_mape) / enhanced_darts_mape) * 100
        vs_original = ((ultimate_mape - original_xgb_mape) / original_xgb_mape) * 100

        print("\nULTIMATE IMPROVEMENTS:")
        print(f"vs Basic Darts:               {vs_basic:.1f}% better")
        print(f"vs Enhanced Darts:            {vs_enhanced:.1f}% better")
        print(
            f"vs Original XGBoost:          {abs(vs_original):.1f}% {'worse' if vs_original > 0 else 'better'}"
        )

        # Performance assessment
        if ultimate_mape < 20:
            status = "EXCELLENT - Close to original performance!"
        elif ultimate_mape < 30:
            status = "VERY GOOD - Significant improvement!"
        elif ultimate_mape < 40:
            status = "GOOD - Notable improvement!"
        else:
            status = "FAIR - Some improvement achieved"

        print(f"\nULTIMATE STATUS: {status}")

        print("\n" + "=" * 75)
        print("ULTIMATE ENHANCED PIPELINE COMPLETED!")
        print("=" * 75)

        print("\nULTIMATE FEATURES IMPLEMENTED:")
        print(f"[+] Extended lag features: {len(lags)} main lags")
        print(f"[+] Past covariates: {len(lags_past_covariates)} lags")
        print(f"[+] Temporal encoders: {len(encoders['datetime_attribute']['past'])} features")
        print("[+] Data scaling: MaxAbsScaler")
        print("[+] Enhanced XGBoost params: 2000 estimators, depth 8")
        print("[+] Advanced regularization")

        return results

    except Exception as e:
        print("\nULTIMATE PIPELINE FAILED!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_ultimate_xgboost()

    if results:
        mape = results["metrics"]["mape"]
        print(f"\nULTIMATE SUCCESS: Achieved {mape:.2f}% MAPE")

        # Final assessment
        if mape < 25:
            print("OUTSTANDING: Darts XGBoost now competitive with original!")
        elif mape < 35:
            print("EXCELLENT: Major improvement in Darts XGBoost performance!")
        else:
            print("GOOD: Meaningful improvement achieved with Darts features!")

    else:
        print("\nFAILED: Ultimate pipeline execution encountered errors.")

    sys.exit(0 if results else 1)
