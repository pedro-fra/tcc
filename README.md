# Sales Forecasting TCC Project

This project implements a sales forecasting system comparing different machine learning approaches with traditional Power BI methods.

## Models Implemented
- ARIMA
- Theta Method  
- Exponential Smoothing
- XGBoost

## Usage

1. Install dependencies:
```bash
uv sync
```

2. Run preprocessing:
```bash
uv run python preprocess_data.py
```

## Project Structure
- `src/preprocessing/` - Data preprocessing pipeline
- `src/models/` - Model implementations
- `data/` - Raw and processed data
- `processed_data/` - Output from preprocessing