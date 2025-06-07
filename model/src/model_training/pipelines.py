import logging
from typing import Dict

import pandas as pd
from lightgbm import LGBMRegressor
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from model.src.data_preparation.preprocessing import build_preprocessing_pipeline

default_logger = logging.getLogger(__name__)


def create_predictor(cfg: DictConfig) -> BaseEstimator:
    """
    Create a regression estimator based on configuration.

    Arguments:
        cfg (DictConfig): Configuration object with attribute `model.estimator`
            specifying one of "XGBoost", "LightGBM", or "RandomForest".

    Returns:
        BaseEstimator: Instantiated scikit-learn or XGBoost/LightGBM regressor.
    """
    if cfg.model.estimator == "XGBoost":
        return XGBRegressor()
    if cfg.model.estimator == "LightGBM":
        return LGBMRegressor()
    if cfg.model.estimator == "RandomForest":
        return RandomForestRegressor()
    raise ValueError(
        f"Unknown estimator: {cfg.model.estimator}."
        "Supported estimators are: 'XGBoost', 'LightGBM', 'RandomForest'."
    )


def make_full_pipeline(cfg: DictConfig, scaling_map: Dict[str, float]) -> Pipeline:
    """
    Build a full modeling pipeline combining preprocessing and estimator.

    Arguments:
        cfg (DictConfig): Configuration object with model settings.
        scaling_map (Dict[str, float]): Mapping from country code to scaling factor.

    Returns:
        pipeline (Pipeline): scikit-learn Pipeline with 'preprocessor' and 'estimator' steps.
    """
    estimator = create_predictor(cfg)
    preprocessor = build_preprocessing_pipeline(scaling_map)

    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return full_pipeline


def opt_params(
    pipe: Pipeline, cfg: DictConfig, X: pd.DataFrame, y: pd.Series
) -> RandomizedSearchCV:
    """
    Optimize hyperparameters using RandomizedSearchCV.

    Arguments:
        pipe (Pipeline): The pipeline to optimize.
        cfg (DictConfig): Configuration containing hyperparameter search space.
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.

    Returns:
        RandomizedSearchCV: Fitted RandomizedSearchCV object with best parameters.
    """

    params_dict = {**{k: tuple(v) for k, v in dict(cfg.model.params).items()}}

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=params_dict,
        n_iter=cfg.model.n_iter,
        cv=cfg.model.cv_folds,
        scoring=cfg.model.scoring,
        random_state=42,
        return_train_score=True,
        refit=True,
    )

    search.fit(X, y)
    return search
