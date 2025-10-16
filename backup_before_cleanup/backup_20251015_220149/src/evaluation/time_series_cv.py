"""
Time Series Cross-Validation Module

Implements time-series specific cross-validation strategies that respect temporal order.
Provides walk-forward validation and expanding window methods for robust model evaluation.
"""

import logging
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries

logger = logging.getLogger(__name__)


class TimeSeriesSplit:
    """
    Time Series cross-validator that respects temporal order.

    Provides train/test indices for time series cross-validation with:
    - Fixed or expanding training window
    - Fixed test window size
    - Configurable step size between folds
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = True,
        min_train_size: Optional[int] = None,
    ):
        """
        Initialize time series cross-validator.

        Parameters
        ----------
        n_splits : int, default=5
            Number of splits/folds
        test_size : int, optional
            Size of test set for each fold. If None, computed automatically
        gap : int, default=0
            Number of samples to exclude between train and test sets
        expanding_window : bool, default=True
            If True, training window expands. If False, uses sliding window
        min_train_size : int, optional
            Minimum number of samples for training. If None, uses n_splits
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size

        logger.info(
            f"TimeSeriesSplit initialized: n_splits={n_splits}, "
            f"expanding_window={expanding_window}, gap={gap}"
        )

    def split(self, X: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data

        Yields
        ------
        train : ndarray
            Training set indices for that split
        test : ndarray
            Testing set indices for that split
        """
        n_samples = len(X)

        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        if self.min_train_size is None:
            min_train_size = n_samples // (self.n_splits + 1)
        else:
            min_train_size = self.min_train_size

        indices = np.arange(n_samples)

        logger.info(
            f"Splitting {n_samples} samples into {self.n_splits} folds with test_size={test_size}"
        )

        for i in range(self.n_splits):
            if self.expanding_window:
                train_end = min_train_size + i * test_size
            else:
                train_start = i * test_size
                train_end = min_train_size + i * test_size

            test_start = train_end + self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                logger.warning(f"Fold {i + 1}: test set exceeds data, skipping")
                break

            if self.expanding_window:
                train_indices = indices[:train_end]
            else:
                train_indices = indices[train_start:train_end]

            test_indices = indices[test_start:test_end]

            logger.debug(f"Fold {i + 1}: train={len(train_indices)}, test={len(test_indices)}")

            yield train_indices, test_indices

    def get_n_splits(self, X: Optional[np.ndarray] = None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits


class TimeSeriesCrossValidator:
    """
    Performs time series cross-validation with custom evaluation function.

    Provides methods to:
    - Split data respecting temporal order
    - Evaluate models across multiple folds
    - Aggregate and report cross-validation results
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = True,
    ):
        """
        Initialize cross-validator.

        Parameters
        ----------
        n_splits : int, default=5
            Number of cross-validation folds
        test_size : int, optional
            Size of test set in each fold
        gap : int, default=0
            Gap between train and test sets
        expanding_window : bool, default=True
            Use expanding window (True) or sliding window (False)
        """
        self.splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            expanding_window=expanding_window,
        )

    def cross_val_score(
        self,
        model,
        series: TimeSeries,
        metric_func,
        verbose: bool = True,
    ) -> Tuple[List[float], float, float]:
        """
        Perform cross-validation and return scores for each fold.

        Parameters
        ----------
        model : object
            Model with fit() and predict() methods compatible with Darts
        series : TimeSeries
            Complete time series data
        metric_func : callable
            Function to compute metric: metric_func(actual, predicted) -> float
        verbose : bool, default=True
            Print progress information

        Returns
        -------
        scores : list of float
            Score for each fold
        mean_score : float
            Mean of scores across all folds
        std_score : float
            Standard deviation of scores
        """
        data = series.values().flatten()
        scores = []

        logger.info(f"Starting cross-validation with {self.splitter.n_splits} folds")

        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(data), start=1):
            train_series = series[train_idx[0] : train_idx[-1] + 1]
            test_series = series[test_idx[0] : test_idx[-1] + 1]

            if verbose:
                logger.info(f"Fold {fold_idx}: Train={len(train_series)}, Test={len(test_series)}")

            try:
                model.fit(train_series)
                predictions = model.predict(n=len(test_series))

                score = metric_func(test_series, predictions)
                scores.append(score)

                if verbose:
                    logger.info(f"Fold {fold_idx} score: {score:.4f}")

            except Exception as e:
                logger.error(f"Fold {fold_idx} failed: {e}")
                continue

        mean_score = np.mean(scores) if scores else np.nan
        std_score = np.std(scores) if scores else np.nan

        logger.info(f"Cross-validation complete: Mean={mean_score:.4f}, Std={std_score:.4f}")

        return scores, mean_score, std_score

    def cross_val_predict(
        self,
        model,
        series: TimeSeries,
        return_scores: bool = False,
        metric_func=None,
    ) -> pd.DataFrame:
        """
        Generate out-of-sample predictions for entire series using cross-validation.

        Parameters
        ----------
        model : object
            Model with fit() and predict() methods
        series : TimeSeries
            Complete time series
        return_scores : bool, default=False
            If True, return scores along with predictions
        metric_func : callable, optional
            Metric function if return_scores is True

        Returns
        -------
        predictions_df : pd.DataFrame
            DataFrame with actual values, predictions, and fold information
        """
        data = series.values().flatten()
        all_predictions = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(data), start=1):
            train_series = series[train_idx[0] : train_idx[-1] + 1]
            test_series = series[test_idx[0] : test_idx[-1] + 1]

            try:
                model.fit(train_series)
                predictions = model.predict(n=len(test_series))

                pred_df = pd.DataFrame(
                    {
                        "date": test_series.time_index,
                        "actual": test_series.values().flatten(),
                        "predicted": predictions.values().flatten(),
                        "fold": fold_idx,
                    }
                )

                all_predictions.append(pred_df)

            except Exception as e:
                logger.error(f"Fold {fold_idx} prediction failed: {e}")
                continue

        predictions_df = pd.concat(all_predictions, ignore_index=True)

        if return_scores and metric_func:
            scores = []
            for fold in predictions_df["fold"].unique():
                fold_data = predictions_df[predictions_df["fold"] == fold]
                score = metric_func(fold_data["actual"], fold_data["predicted"])
                scores.append({"fold": fold, "score": score})

            scores_df = pd.DataFrame(scores)
            logger.info(f"Mean CV score: {scores_df['score'].mean():.4f}")

        return predictions_df


def walk_forward_validation(
    model,
    series: TimeSeries,
    initial_train_size: int,
    horizon: int = 1,
    step: int = 1,
    metric_func=None,
) -> dict:
    """
    Perform walk-forward validation on time series.

    This method:
    1. Trains on initial window
    2. Predicts next horizon steps
    3. Slides window forward by step
    4. Repeats until end of series

    Parameters
    ----------
    model : object
        Model with fit() and predict() methods
    series : TimeSeries
        Complete time series
    initial_train_size : int
        Initial training window size
    horizon : int, default=1
        Number of steps to forecast ahead
    step : int, default=1
        Number of steps to move forward after each prediction
    metric_func : callable, optional
        Function to compute error metric

    Returns
    -------
    results : dict
        Dictionary containing predictions, actuals, and scores
    """
    predictions_list = []
    actuals_list = []
    scores_list = []

    total_length = len(series)
    current_train_end = initial_train_size

    logger.info(
        f"Starting walk-forward validation: initial_train={initial_train_size}, "
        f"horizon={horizon}, step={step}"
    )

    iteration = 0
    while current_train_end + horizon <= total_length:
        iteration += 1

        train_series = series[:current_train_end]
        test_start = current_train_end
        test_end = min(current_train_end + horizon, total_length)
        test_series = series[test_start:test_end]

        try:
            model.fit(train_series)
            pred = model.predict(n=horizon)

            actual_horizon = len(test_series)
            pred_values = pred.values().flatten()[:actual_horizon]
            actual_values = test_series.values().flatten()

            predictions_list.extend(pred_values)
            actuals_list.extend(actual_values)

            if metric_func:
                score = metric_func(actual_values, pred_values)
                scores_list.append(score)

            logger.debug(
                f"Iteration {iteration}: Train size={len(train_series)}, "
                f"Predicted {actual_horizon} steps"
            )

        except Exception as e:
            logger.error(f"Iteration {iteration} failed: {e}")

        current_train_end += step

    results = {
        "predictions": np.array(predictions_list),
        "actuals": np.array(actuals_list),
        "n_iterations": iteration,
    }

    if scores_list:
        results["scores"] = scores_list
        results["mean_score"] = np.mean(scores_list)
        results["std_score"] = np.std(scores_list)
        logger.info(f"Walk-forward validation complete: Mean score={results['mean_score']:.4f}")

    return results
