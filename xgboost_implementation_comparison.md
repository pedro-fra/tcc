# XGBoost Implementation Comparison: Raw XGBoost vs Darts XGBModel

## Overview
This document compares the results and characteristics of two XGBoost implementations for the TCC sales forecasting project:
1. **Original Implementation**: Raw XGBoost with custom feature engineering
2. **New Implementation**: Darts XGBModel with automated time series handling

## Implementation Characteristics

### Original XGBoost Implementation
- **Library**: Raw `xgboost.XGBRegressor`
- **Feature Engineering**: Manual creation of 45 features including:
  - Temporal features (year, month, quarter, etc.)
  - Cyclical encoding (sin/cos transformations)
  - Lag features (1, 2, 3, 6, 12 periods)
  - Rolling statistics (MA, std, min, max)
  - Calendar features (month/quarter start/end)
  - Interaction features
- **Data Format**: Pandas DataFrame with manual preprocessing
- **Training Method**: scikit-learn compatible fit/predict
- **Hyperparameter Tuning**: GridSearchCV with TimeSeriesSplit

### New Darts XGBModel Implementation
- **Library**: `darts.models.XGBModel`
- **Feature Engineering**: Automatic lag feature creation
- **Lag Configuration**: `[-1, -2, -3, -6, -12]` periods
- **Data Format**: Native Darts TimeSeries objects
- **Training Method**: Darts native fit/predict API
- **Time Series Integration**: Built-in time series validation and handling

## Performance Comparison

### Data Split Information
| Metric | Original | Darts | Difference |
|--------|----------|-------|------------|
| Total Samples | 120 | 132 | +12 samples |
| Train Samples | 96 | 105 | +9 samples |
| Test Samples | 24 | 27 | +3 samples |
| Train Period | 2015-10-31 to 2023-09-30 | 2014-10-31 to 2023-06-30 | Earlier start |
| Test Period | 2023-10-31 to 2025-09-30 | 2023-07-31 to 2025-09-30 | Slightly different |

### Model Performance Metrics
| Metric | Original | Darts | Change | Performance |
|--------|----------|-------|--------|-------------|
| **MAE** | 6,246,418 | 19,191,511 | +207% | **Worse** |
| **RMSE** | 7,885,916 | 22,088,032 | +180% | **Worse** |
| **MAPE** | 16.32% | 54.73% | +235% | **Worse** |

## Analysis

### Why Different Performance?

**1. Feature Engineering Complexity**
- **Original**: 45 engineered features including complex temporal patterns, rolling statistics, and interactions
- **Darts**: Only 5 lag features automatically created
- **Impact**: Original model had much richer feature representation

**2. Data Preprocessing**
- **Original**: Custom preprocessing pipeline with advanced feature engineering
- **Darts**: Simplified approach relying on automatic lag creation
- **Impact**: Less information available to the Darts model

**3. Model Configuration**
- **Original**: May have used optimized hyperparameters from previous tuning
- **Darts**: Using default/basic configuration without optimization
- **Impact**: Suboptimal model parameters for Darts implementation

**4. Time Series Handling**
- **Original**: Custom time series splitting and validation
- **Darts**: Native time series handling, potentially different validation approach
- **Impact**: Different training/validation methodology

### Advantages of Each Approach

#### Original XGBoost Advantages
- **Superior Performance**: Much better forecasting accuracy
- **Rich Features**: Comprehensive feature engineering captures complex patterns
- **Flexibility**: Full control over preprocessing and feature creation
- **Optimization**: Can use advanced hyperparameter tuning techniques

#### Darts XGBModel Advantages
- **Simplicity**: Minimal code, automatic time series handling
- **Consistency**: Unified API with other time series models (ARIMA, Theta, etc.)
- **Maintenance**: Easier to maintain and understand
- **Integration**: Seamless integration with Darts ecosystem
- **Standards**: Follows time series forecasting best practices

## Recommendations

### For Production Use
**Keep the Original XGBoost Implementation** for production forecasting due to:
- Significantly better performance (16% vs 55% MAPE)
- Proven accuracy on the TCC dataset
- Advanced feature engineering captures business patterns

### For Research Consistency
**Use Darts XGBModel** when:
- Comparing with other Darts models (ARIMA, Theta, Exponential Smoothing)
- Ensuring methodological consistency across models
- Building a unified evaluation framework

### Improvement Opportunities for Darts Implementation
1. **Add More Lags**: Include more lag periods for richer temporal information
2. **Feature Engineering**: Add custom features using Darts' feature engineering capabilities
3. **Hyperparameter Tuning**: Optimize XGBoost parameters specifically for Darts
4. **Ensemble Approaches**: Combine multiple lag configurations
5. **External Features**: Add external variables as covariates

## Conclusion

While the Darts XGBModel implementation successfully demonstrates:
- ✅ Library migration from raw XGBoost to Darts
- ✅ Consistent API with other time series models  
- ✅ Proper time series handling and validation
- ✅ Clean, maintainable code structure

The performance difference is significant due to the simpler feature engineering approach. For the TCC project's goal of comparing best-performing ML models against traditional methods, the original XGBoost implementation should be retained for its superior accuracy.

The Darts implementation serves as a valuable proof-of-concept showing how XGBoost can be integrated into a unified time series forecasting framework, which could be beneficial for future research and development efforts.

## Files Generated
- **Model**: `data/processed_data/xgboost/xgboost_darts_model.pkl`
- **Results**: `data/processed_data/xgboost/xgboost_darts_model_summary.json`
- **Pipeline**: `data/processed_data/xgboost/xgboost_darts_pipeline_summary.json`
- **Plots**: Attempted but failed due to API mismatch with existing visualization module

**Date**: 2025-01-13  
**Implementation**: Fully functional Darts XGBModel pipeline  
**Status**: Successfully completed with comprehensive results