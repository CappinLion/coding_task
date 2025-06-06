from typing import List, Tuple

from sklearn.pipeline import Pipeline


def get_feat_importance(model: Pipeline) -> List[Tuple[str, float]]:
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

    # Retrieve the demand_scaler mapping order (country codes)
    preproc = model.named_steps["preprocessor"]
    numeric_pipeline = preproc.transformers_[0][1]
    country_codes = list(numeric_pipeline.named_steps["demand_scaler"].scaling_map.keys())

    # Build the list of feature names in the same order as the pipeline input:
    # Each code has three centers: code_1, code_2, code_3
    feature_names: List[str] = [f"{code}_{i}" for code in country_codes for i in (1, 2, 3)]

    # Combine names and importances, sort descending
    feat_list: List[Tuple[str, float]] = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    return feat_list
