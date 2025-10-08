"""
Base optimizer class for hyperparameter tuning using Optuna.
Provides common functionality for all model optimizers.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from config import OPTUNA_CONFIG

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class for model hyperparameter optimization using Optuna.

    Provides common functionality for optimization studies including:
    - Study creation and management
    - Best parameters extraction and persistence
    - Metrics calculation
    - Logging and reporting
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        search_space: Dict[str, Any],
        optuna_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base optimizer.

        Args:
            model_name: Name of the model being optimized
            config: Model configuration dictionary
            search_space: Hyperparameter search space definition
            optuna_config: Optuna-specific configuration (optional)
        """
        self.model_name = model_name
        self.config = config
        self.search_space = search_space
        self.optuna_config = optuna_config or OPTUNA_CONFIG

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None

        logger.info(f"Initialized {model_name} optimizer with Optuna")

    def create_study(self, study_name: Optional[str] = None) -> optuna.Study:
        """
        Create or load an Optuna study for optimization.

        Args:
            study_name: Name for the study (default: model_name)

        Returns:
            Optuna Study object
        """
        if study_name is None:
            study_name = f"{self.model_name}_optimization"

        # Configure sampler
        sampler = TPESampler(
            seed=self.optuna_config["random_state"],
            n_startup_trials=self.optuna_config["n_startup_trials"],
            n_ei_candidates=self.optuna_config["n_ei_candidates"],
        )

        # Configure pruner
        pruner = MedianPruner(
            n_startup_trials=self.optuna_config["n_startup_trials"],
            n_warmup_steps=5,
        )

        # Create or load study
        storage = self.optuna_config["storage_url"]
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=self.optuna_config["direction"],
            load_if_exists=True,
        )

        logger.info(f"Created study: {study_name}")
        logger.info(f"Storage: {storage}")

        return self.study

    @abstractmethod
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function to minimize/maximize.

        Must be implemented by subclasses for each specific model.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (e.g., MAE, RMSE)
        """
        pass

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}

        for param_name, param_config in self.search_space.items():
            if "choices" in param_config:
                # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
            elif "log" in param_config and param_config["log"]:
                # Float parameter with log scale
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=True,
                )
            elif "low" in param_config and "high" in param_config:
                # Check if integer or float based on values
                if isinstance(param_config["low"], int) and isinstance(param_config["high"], int):
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"]
                    )

        return params

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run optimization study.

        Args:
            n_trials: Number of trials (default: from config)
            timeout: Timeout in seconds (default: from config)
            show_progress: Show progress bar

        Returns:
            Dictionary with optimization results
        """
        if self.study is None:
            self.create_study()

        n_trials = n_trials or self.optuna_config["n_trials"]
        timeout = timeout or self.optuna_config["timeout"]

        logger.info(f"Starting optimization for {self.model_name}")
        logger.info(f"n_trials={n_trials}, timeout={timeout}s")

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            callbacks=[self._log_callback],
        )

        # Extract best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"Optimization completed for {self.model_name}")
        logger.info(f"Best value: {self.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "study_name": self.study.study_name,
        }

    def _log_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """
        Callback function to log trial results.

        Args:
            study: Optuna study
            trial: Completed trial
        """
        if trial.value is not None:
            logger.info(
                f"Trial {trial.number} finished: value={trial.value:.4f}, params={trial.params}"
            )

    def save_best_params(self, output_path: Path):
        """
        Save best parameters to JSON file.

        Args:
            output_path: Path to save parameters
        """
        if self.best_params is None:
            logger.warning("No best parameters to save. Run optimization first.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "model_name": self.model_name,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials) if self.study else 0,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Best parameters saved to {output_path}")

    def load_best_params(self, input_path: Path) -> Dict[str, Any]:
        """
        Load best parameters from JSON file.

        Args:
            input_path: Path to load parameters from

        Returns:
            Dictionary with best parameters
        """
        with open(input_path) as f:
            results = json.load(f)

        self.best_params = results["best_params"]
        self.best_value = results["best_value"]

        logger.info(f"Loaded best parameters from {input_path}")

        return self.best_params

    def get_optimization_history(self) -> Dict[str, Any]:
        """
        Get optimization history and statistics.

        Returns:
            Dictionary with optimization statistics
        """
        if self.study is None:
            logger.warning("No study available. Run optimization first.")
            return {}

        trials = self.study.trials
        values = [t.value for t in trials if t.value is not None]

        return {
            "n_trials": len(trials),
            "best_value": self.best_value,
            "best_trial": self.study.best_trial.number,
            "mean_value": np.mean(values) if values else None,
            "std_value": np.std(values) if values else None,
            "min_value": np.min(values) if values else None,
            "max_value": np.max(values) if values else None,
        }

    def get_param_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance using Optuna's built-in method.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None or len(self.study.trials) < 2:
            logger.warning("Not enough trials to calculate importance.")
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            logger.info("Parameter importance calculated successfully")
            return importance
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}
