"""
Configuration file for sales forecasting preprocessing pipeline.
Contains all parameters and paths used in data processing.
"""

from pathlib import Path

# Base project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
SRC_DIR = PROJECT_ROOT / "src"

# Input data files
RAW_DATA_FILE = DATA_DIR / "fat_factory.csv"

# Output data files
PROCESSED_TIME_SERIES_FILE = PROCESSED_DATA_DIR / "time_series_data.parquet"
PROCESSED_TABULAR_FILE = PROCESSED_DATA_DIR / "tabular_data.parquet"
DATA_SUMMARY_FILE = PROCESSED_DATA_DIR / "data_summary.json"

# Data processing parameters
DATA_CONFIG = {
    "csv_separator": ";",
    "encoding": "utf-8-sig",  # Handle BOM in CSV
    "date_column": "DATA_EMISSAO_PEDIDO",
    "value_column": "VALOR_LIQ",
    "customer_column": "apelido",
    "operation_column": "OPERACAO",
    "date_format": "%d/%m/%Y",
    "target_operation": "VENDA",  # Filter only sales transactions
}

# Anonymization parameters
ANONYMIZATION_CONFIG = {
    "hash_length": 4,  # Number of digits for anonymized IDs
    "customer_prefix": "CLIENTE_",
    "use_hash": True,  # Whether to use hash-based anonymization
}

# Time series configuration
TIME_SERIES_CONFIG = {
    "frequency": "M",  # Monthly aggregation
    "aggregation_method": "sum",  # Sum sales values
    "fill_missing_dates": True,
    "min_date": None,  # Auto-detect from data
    "max_date": None,  # Auto-detect from data
    "remove_outliers": False,  # Never remove outliers in sales forecasting
    "apply_smoothing": False,  # Optional smoothing (disabled by default)
    "smoothing_window": 3,  # Window size if smoothing is enabled
}

# Feature engineering parameters
FEATURE_ENGINEERING_CONFIG = {
    "create_temporal_features": True,
    "create_cyclical_features": True,
    "create_lag_features": True,
    "lag_periods": [1, 2, 3, 6, 12],  # Monthly lags
    "create_moving_averages": True,
    "ma_windows": [3, 6, 12],  # Moving average windows
    "create_aggregated_features": True,
}

# Scaling parameters
SCALING_CONFIG = {
    "method": "robust",  # RobustScaler - resistant to outliers
    "exclude_columns": ["target", "year", "month", "day"],
    "feature_range": None,  # Use default for RobustScaler
}

# Data quality parameters
DATA_QUALITY_CONFIG = {
    "remove_duplicates": True,
    "handle_missing_values": True,
    "missing_value_strategy": {
        "categorical": "constant",  # Fill with constant value
        "numerical": "zero",  # Fill with zero
    },
    "filter_zero_values": True,  # Remove zero/negative sales
    "min_sales_value": 0.01,  # Minimum valid sales value
}

# ARIMA model configuration
ARIMA_CONFIG = {
    "start_p": 0,  # Starting value for autoregressive order
    "start_q": 0,  # Starting value for moving average order
    "max_p": 5,  # Maximum autoregressive order
    "max_q": 5,  # Maximum moving average order
    "max_d": 2,  # Maximum differencing order
    "seasonal": True,  # Enable seasonal ARIMA
    "seasonal_periods": 12,  # Monthly seasonality
    "max_P": 2,  # Maximum seasonal autoregressive order
    "max_Q": 2,  # Maximum seasonal moving average order
    "max_D": 1,  # Maximum seasonal differencing order
    "ic": "aic",  # Information criterion (aic, bic, hqic)
    "stepwise": True,  # Enable stepwise for faster, often better results
    "approximation": False,  # Use exact likelihood for better fit
    "seasonal_test": "seas",  # Test for seasonality
    "test": "adf",  # Stationarity test
    "suppress_warnings": True,  # Suppress convergence warnings
    "with_intercept": True,  # Include intercept in model
    "alpha": 0.05,  # Significance level for confidence intervals
    "random_state": 42,  # Random seed for reproducibility
    "test_ratio": 0.2,  # Proportion of data for testing
    # Preprocessing options
    "use_log_transform": False,  # Apply log transformation for variance stabilization
    "log_offset": 1,  # Offset to add before log transformation (handles zeros)
    "check_for_negative": True,  # Check for negative values before log transform
}

# Exponential Smoothing model configuration
EXPONENTIAL_SMOOTHING_CONFIG = {
    "trend": None,  # None for auto-detection, or 'add'/'mul'
    "seasonal": None,  # None for auto-detection, or 'add'/'mul'
    "seasonal_periods": 12,  # Monthly seasonality
    "damped": None,  # None for auto-detection
    "random_state": 42,  # Random seed for reproducibility
    "test_ratio": 0.2,  # Proportion of data for testing
}

# Theta model configuration
THETA_CONFIG = {
    "season_length": 12,  # Monthly seasonality period
    "theta": None,  # Auto-optimize theta parameter (None = automatic selection)
    "use_mle": True,  # Use Maximum Likelihood Estimation for parameter fitting
    "random_state": 42,  # Random seed for reproducibility
    "test_ratio": 0.2,  # Proportion of data for testing
}

# XGBoost model configuration
XGBOOST_CONFIG = {
    # Core XGBoost parameters
    "n_estimators": 1000,  # Number of boosting rounds
    "max_depth": 6,  # Maximum tree depth
    "learning_rate": 0.1,  # Learning rate (eta)
    "subsample": 0.8,  # Subsample ratio of training instances
    "colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree
    "reg_alpha": 0.1,  # L1 regularization term on weights
    "reg_lambda": 1.0,  # L2 regularization term on weights
    "random_state": 42,  # Random seed for reproducibility
    "test_ratio": 0.2,  # Proportion of data for testing
    # Training parameters
    "objective": "reg:squarederror",  # Regression objective
    "eval_metric": "rmse",  # Evaluation metric
    "early_stopping_rounds": 50,  # Stop if no improvement
    "verbose": False,  # Suppress training output
    # Hyperparameter tuning
    "hyperparameter_tuning": False,  # Enable hyperparameter optimization
    "cv_folds": 5,  # Cross-validation folds for hyperparameter tuning
    "param_grid": {
        "n_estimators": [500, 1000, 1500],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    # Feature engineering (used by TabularFormatter)
    "categorical_encoding": {
        "high_cardinality_threshold": 50,
        "low_cardinality_method": "onehot",
        "high_cardinality_method": "label",
    },
}

# Model output directories
MODEL_DIRS = {
    "arima": PROCESSED_DATA_DIR / "arima",
    "theta": PROCESSED_DATA_DIR / "theta",
    "exponential_smoothing": PROCESSED_DATA_DIR / "exponential_smoothing",
    "xgboost": PROCESSED_DATA_DIR / "xgboost",
}


# Ensure directories exist
def create_directories():
    """Create necessary directories for data processing."""
    for directory in [PROCESSED_DATA_DIR] + list(MODEL_DIRS.values()):
        directory.mkdir(parents=True, exist_ok=True)


def load_config():
    """
    Load complete configuration dictionary.

    Returns:
        Dictionary with all configuration parameters
    """
    return {
        "data": DATA_CONFIG,
        "anonymization": ANONYMIZATION_CONFIG,
        "time_series": TIME_SERIES_CONFIG,
        "feature_engineering": FEATURE_ENGINEERING_CONFIG,
        "scaling": SCALING_CONFIG,
        "data_quality": DATA_QUALITY_CONFIG,
        "arima": ARIMA_CONFIG,
        "exponential_smoothing": EXPONENTIAL_SMOOTHING_CONFIG,
        "theta": THETA_CONFIG,
        "xgboost": XGBOOST_CONFIG,
        "paths": {
            "project_root": PROJECT_ROOT,
            "data_dir": DATA_DIR,
            "processed_data_dir": PROCESSED_DATA_DIR,
            "raw_data_file": RAW_DATA_FILE,
            "model_dirs": MODEL_DIRS,
        },
    }


if __name__ == "__main__":
    create_directories()
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
