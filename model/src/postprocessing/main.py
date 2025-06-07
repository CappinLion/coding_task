from typing import List, Tuple

import pandas as pd
from sklearn.pipeline import Pipeline


def get_feat_importance(model: Pipeline, df: pd.DataFrame) -> List[Tuple[str, float]]:
    """
    Extract and return feature importances from a fitted pipeline.

    Arguments:
        model (Pipeline): A fitted sklearn Pipeline with steps:
                          - 'preprocessor' (ColumnTransformer → numeric pipeline first)
                          - 'estimator'   (has attribute `feature_importances_`)
        df (pd.DataFrame): Original feature DataFrame before preprocessing,
                           used to retrieve feature names.

    Returns:
        List[Tuple[str, float]]: Sorted list of (feature_name, importance) tuples
                                 in descending order of importance.
    """
    importances = model.named_steps["estimator"].feature_importances_
    feature_names = list(df.columns)

    # Combine names and importances, sort descending
    feat_list: List[Tuple[str, float]] = sorted(zip(feature_names, importances), reverse=True)
    return feat_list
