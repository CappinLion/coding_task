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
- Use mlflow for experiment tracking: set `cfg.model.mlflow.enable` to `true`

The E2E Pipeline will:
1. Load raw data and scaling factors
2. Split to train/holdout data using stratification on `"Sales"`
3. Treat Outliear using percentile clipping
4. Build a preprocessing pipeline with country-scaling, standardization, 1-Hot-Encoding for seasonal factors
5. Hyperparameter search using random search over the selected estimator
6. Evaluate on the holdout set with RMSE and R_squared
7. Display top feature importances
8. Create a run within a mlflow experiment and log metrics and model hyperparameters

## Technical Summary

### Pipeline
Below is the core `Pipeline` structure:

![Pipeline Structure](data/pipeline.png)

1. Preprocessor (Pipeline)
- ColumnTransformer splits our inputs depending on the datatype of the features:
    - Numerical features (AUS_1, ..., UK_3)
        1) DemandScalingTransformer
            - Applies a country-specific multiplier (from our scaling_map) to each center’s demand forecast
            - Example: if USA’s factor = 1.5, then USA_1, USA_2, USA_3 are all multiplied by 1.5
        2) StandardScaler
            - This will be applied to each scaled demand column so that all numeric inputs live on a comparable scale
    - Categorical feature (factor)
        1) SimpleImputer
            - Fills any missing seasonal factor with the placeholder "missing".
        2) OneHotEncoder
            - Converts the factor levels (e.g. factor1, factor2,..., missing) into binary indicator columns (1 or 0), dropping one level to avoid collinearity.
2. Estimator
- XGBRegressor (or whichever algorithm you choose: LightGBM, RandomForest)
- Receives the preprocessed, feature-engineered matrix and learns to predict Sales.

### Evaluation Summary

- We can observe that the average error (RMSE) during hyperparameter tuning is 12.46, while the RMSE for the unseen holdout set is **12.43**
- This closeness indicates that the model is neither over- nor underfitting
- The target feature, `Sales`, ranges roughly from 300 to 1000 with a mean around 525 (see EDA notebook). An average error of 12.4 is about 2-3% of that mean which is fairly low
- The R_squared is 0.98 which means that the model explains about 98% of the variance in the holdout set Sales values.
- As expected, the feature with the highest importance is `factor` with a value of almost 0.2

![Feature Importances](data/feat_imp.png)

### Logs
```
[2025-06-07 23:56:06,849][run_script][INFO] - Reading data...
[2025-06-07 23:56:15,079][run_script][INFO] - Data loaded successfully.
[2025-06-07 23:56:15,161][run_script][INFO] - Training begins...
[2025-06-07 23:56:15,161][run_script][INFO] - Using estimator: XGBoost
[2025-06-07 23:56:37,050][run_script][INFO] - Best CV score (neg_root_mean_squared_error): 12.46
[2025-06-07 23:56:37,050][run_script][INFO] - Running inference on test set...
[2025-06-07 23:56:37,081][run_script][INFO] - Inference completed.
[2025-06-07 23:56:37,081][run_script][INFO] - ======= Test Set Evaluation =======
[2025-06-07 23:56:37,082][run_script][INFO] - Test RMSE: 12.43
[2025-06-07 23:56:37,082][run_script][INFO] - Test R_squared: 0.9826
[2025-06-07 23:56:37,082][run_script][INFO] - ======= Top 10 Feature Importances =======
[2025-06-07 23:56:37,082][run_script][INFO] - factor: 0.1977
[2025-06-07 23:56:37,082][run_script][INFO] - FRA_2: 0.0582
[2025-06-07 23:56:37,082][run_script][INFO] - IND_3: 0.0393
[2025-06-07 23:56:37,083][run_script][INFO] - GER_1: 0.0385
[2025-06-07 23:56:37,083][run_script][INFO] - FRA_1: 0.0200
[2025-06-07 23:56:37,083][run_script][INFO] - AUS_2: 0.0188
[2025-06-07 23:56:37,083][run_script][INFO] - GER_2: 0.0055
[2025-06-07 23:56:37,083][run_script][INFO] - JAP_1: 0.0040
[2025-06-07 23:56:37,083][run_script][INFO] - GER_3: 0.0034
[2025-06-07 23:56:37,083][run_script][INFO] - UK_2: 0.0033
[2025-06-07 23:56:37,084][run_script][INFO] - The whole training and inference took: 30.23 seconds.
```

### Mlflow
- Run this command: `poetry run mlflow ui`
- Open the localhost on port 5000

![MlFlow Experiment](data/mlflow_experiment.png)
![MlFlow Run](data/mlflow_run.png)