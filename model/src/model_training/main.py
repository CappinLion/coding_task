from typing import Dict, Tuple

import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from model.src.model_training.pipelines import make_full_pipeline, opt_params
from model.src.utils.helpers import get_logger

logger = get_logger("model_training")


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

    logger.info("Training completed.")
    logger.info("======= Hyperparameter Search Results =======")
    logger.info(f"Best parameters: {best_params}")

    return cross_val_result, best_model
