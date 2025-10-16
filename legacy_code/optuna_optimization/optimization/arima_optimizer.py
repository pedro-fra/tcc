"""
ARIMA model hyperparameter optimizer using Optuna.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
from darts.models import ARIMA

from config import SEARCH_SPACES
from preprocessing import MLDataPreprocessor, TimeSeriesFormatter

from .base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)


class ARIMAOptimizer(BaseOptimizer):
    """
    Optimizer for ARIMA forecasting model using Optuna.

    Performs hyperparameter optimization to find the best ARIMA configuration
    for sales forecasting using Darts library.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_path: Path,
        search_space: Optional[Dict[str, Any]] = None,
        optuna_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ARIMA optimizer.

        Args:
            config: Full configuration dictionary
            data_path: Path to input data file
            search_space: Hyperparameter search space (default: from config)
            optuna_config: Optuna configuration (default: from config)
        """
        search_space = search_space or SEARCH_SPACES["arima"]

        super().__init__(
            model_name="arima",
            config=config,
            search_space=search_space,
            optuna_config=optuna_config,
        )

        self.data_path = data_path
        self.train_series = None
        self.val_series = None

        logger.info("ARIMA optimizer initialized")

    def prepare_data(self):
        """
        Load and prepare data for optimization.

        This follows the same preprocessing pipeline as the ARIMA model.
        """
        logger.info(f"Loading and preprocessing data from {self.data_path}")

        # Initialize preprocessor
        preprocessor = MLDataPreprocessor(self.config)

        # Load and preprocess data
        processed_df = preprocessor.process_data(Path(self.data_path))

        # Format for time series models
        ts_formatter = TimeSeriesFormatter(self.config)
        series, df_aggregated = ts_formatter.format_for_time_series_models(processed_df)

        # Split into train and validation
        test_ratio = self.config["arima"]["test_ratio"]
        split_point = int(len(series) * (1 - test_ratio))

        self.train_series = series[:split_point]
        self.val_series = series[split_point:]

        logger.info(
            f"Data prepared: {len(self.train_series)} train, {len(self.val_series)} val time points"
        )

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for ARIMA optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Mean Absolute Error on validation set
        """
        # Ensure data is prepared
        if self.train_series is None:
            self.prepare_data()

        # Suggest hyperparameters
        params = self.suggest_params(trial)

        # Extract ARIMA orders
        p = params.get("p", 1)
        d = params.get("d", 1)
        q = params.get("q", 1)
        P = params.get("P", 1)
        D = params.get("D", 1)
        Q = params.get("Q", 1)
        seasonal_periods = params.get("seasonal_periods", 12)

        try:
            # Build ARIMA model
            model = ARIMA(
                p=p,
                d=d,
                q=q,
                seasonal_order=(P, D, Q, seasonal_periods),
                trend="c" if self.config["arima"].get("with_intercept", True) else None,
                random_state=42,
            )

            # Train model
            model.fit(self.train_series)

            # Predict on validation set
            predictions = model.predict(n=len(self.val_series))

            # Calculate MAE
            mae = np.mean(
                np.abs(self.val_series.values().flatten() - predictions.values().flatten())
            )

            # Report intermediate value for pruning
            trial.report(mae, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return mae

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            # Return large value for failed trials
            return float("inf")

    def optimize_with_walk_forward(
        self, n_splits: int = 3, n_trials: Optional[int] = None, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run optimization with walk-forward validation.

        Args:
            n_splits: Number of walk-forward splits
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Running optimization with {n_splits}-split walk-forward validation")

        # Prepare data
        if self.train_series is None:
            self.prepare_data()

        # Combine train and val for walk-forward
        full_series = self.train_series.append(self.val_series)

        def objective_wf(trial):
            params = self.suggest_params(trial)

            p = params.get("p", 1)
            d = params.get("d", 1)
            q = params.get("q", 1)
            P = params.get("P", 1)
            D = params.get("D", 1)
            Q = params.get("Q", 1)
            seasonal_periods = params.get("seasonal_periods", 12)

            # Calculate split points
            total_len = len(full_series)
            test_size = total_len // (n_splits + 1)
            wf_scores = []

            for i in range(n_splits):
                train_end = total_len - (n_splits - i) * test_size
                test_end = train_end + test_size

                train = full_series[:train_end]
                test = full_series[train_end:test_end]

                try:
                    model = ARIMA(
                        p=p,
                        d=d,
                        q=q,
                        seasonal_order=(P, D, Q, seasonal_periods),
                        trend="c" if self.config["arima"].get("with_intercept", True) else None,
                        random_state=42,
                    )

                    model.fit(train)
                    predictions = model.predict(n=len(test))

                    mae = np.mean(np.abs(test.values().flatten() - predictions.values().flatten()))
                    wf_scores.append(mae)

                except Exception as e:
                    logger.warning(f"Walk-forward fold {i} failed: {e}")
                    return float("inf")

            mean_mae = np.mean(wf_scores)
            trial.report(mean_mae, step=0)

            if trial.should_prune():
                raise optuna.TrialPruned()

            return mean_mae

        # Replace objective temporarily
        original_objective = self.objective
        self.objective = objective_wf

        # Run optimization
        results = self.optimize(n_trials=n_trials, timeout=timeout)

        # Restore original objective
        self.objective = original_objective

        return results
