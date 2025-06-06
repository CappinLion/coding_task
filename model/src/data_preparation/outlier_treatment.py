from typing import List, Optional

import pandas as pd


def treat_outliers(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """
    Clip numerical columns in `df` to the [lower_quantile, upper_quantile] range.

    Arguments:
        df (pd.DataFrame): Input DataFrame.
        cols (List[str], optional): Columns to treat. If None, all numeric columns are used.
        lower_quantile (float): Lower quantile for clipping (e.g. 0.01 → 1st percentile).
        upper_quantile (float): Upper quantile for clipping (e.g. 0.99 → 99th percentile).

    Returns:
        pd.DataFrame: A copy of `df` where each specified column has been clipped
                      to its own [q_low, q_high] bounds.
    """
    df_out = df.copy()
    if cols is None:
        cols = df_out.select_dtypes(include="number").columns.tolist()

    lower_bounds = df_out[cols].quantile(lower_quantile)
    upper_bounds = df_out[cols].quantile(upper_quantile)

    for col in cols:
        lb = lower_bounds[col]
        ub = upper_bounds[col]
        df_out[col] = df_out[col].clip(lower=lb, upper=ub)

    return df_out
