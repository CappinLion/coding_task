# Docstar1000 Demand Forecasting
This repo contains an end-to-end demand forecasting pipeline for Catapult Inc's Docstar 1000 spray. It covers data loading, preprocessing (country-scaling, outliear treatment), model training, hyperparameter tuning, inference and evaluation.

## Setup
1. Install poetry
``` curl -sSL https://install.python-poetry.org | python3 - ```
2. Install pre-commit hooks
``` pre-commit install```
3. Install dependencies
`poetry install`
4. Activate the virtual environment
 
## Run the End-to-End Pipeline
- Command: `poetry run python scripts/run.py`
- Modifications: change parameters in the `conf/config.yaml`
- Unit Tests: `poetry run pytest`

The E22 Pipeline will:
1. Load raw data and scaling factors
2. Split to train/holdout data using stratification on `"Sales"`
3. Treat Outliear using percentile clipping
4. Build a preprocessing pipeline with country-scaling, standardization, 1-Hot-Encoding for seasonal factors
5. Hyperparameter search using random search over the selected estimator
6. Evaluate on the holdout set with RMSE and R_squared
7. Display top feature importances

