from typing import Dict, Tuple

import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from model.src.model_training.pipelines import make_full_pipeline, opt_params


def get_optimized_pipeline(
    cfg: DictConfig, scaling_map: Dict[str, float], X: pd.DataFrame, y: pd.Series
) -> Tuple[RandomizedSearchCV, BaseEstimator]:
    """
    Build a full pipeline, optimize its hyperparameters, and return the results.

    Arguments:
        cfg (DictConfig): Configuration object containing model & search settings.
        scaling_map (Dict[str, float]): Mapping from country codes to scaling factors.
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target Series.

    Returns:
        Tuple[RandomizedSearchCV, BaseEstimator]:
        - cross_val_result: The fitted RandomizedSearchCV instance.
        - best_model: Pipeline with the best found parameters (fitted on full data).
    """
    pipeline = make_full_pipeline(cfg=cfg, scaling_map=scaling_map)

    cross_val_result = opt_params(pipe=pipeline, cfg=cfg, X=X, y=y)
    best_params = cross_val_result.best_params_
    best_model = cross_val_result.best_estimator_

    print("Training completed.")
    print("=== Hyperparameter Search Results ===")
    print(f"Best parameters: {best_params}")
    print(f"Best model: {best_model}")

    return cross_val_result, best_model
