from typing import List, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline


def get_feat_importance(model: Pipeline, df: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Extract and return feature importances from a fitted pipeline.

    Arguments:
        model (Pipeline): A fitted sklearn Pipeline with steps:
                          - "preprocessor"
                          - "estimator"
        df (pd.DataFrame): Original feature DataFrame before preprocessing

    Returns:
        List[Tuple[str, float]]: Sorted list of (feature_name, importance) tuples in descending order of importance.
    """
    importances = model.named_steps["estimator"].feature_importances_
    feature_names = list(df.columns)

    feat_list = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return feat_list


def evaluate_metrics(y_test: pd.Series, y_pred: pd.Series) -> dict:
    """
    Evaluate and return performance metrics for regression tasks.

    Arguments:
        y_test (pd.Series): True target values.
        y_pred (pd.Series): Predicted target values.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    return metrics
