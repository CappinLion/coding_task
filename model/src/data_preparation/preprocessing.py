from typing import Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# pylint: disable=unused-argument
class DemandScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies country-specific scaling factors to the three
    processing centers for each country (e.g., 'GER_1', 'GER_2', 'GER_3').
    """

    def __init__(self, scaling_map=dict):
        self.scaling_map = scaling_map

    def fit(self, X: pd.DataFrame, y=None) -> "DemandScalingTransformer":
        """No need for fitting, but signature must accept X and y."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame.")
        if not isinstance(self.scaling_map, dict):
            raise ValueError("scaling_map must be a dictionary mapping country codes to factors.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Multiply each demand column "<CODE>_<i>" by its country-specific factor.

        Arguments:
            X (pd.DataFrame): DataFrame containing demand columns (e.g. 'AUS_1', 'CAN_2', ...).

        Returns:
            pd.DataFrame: A copy of X with scaled demand columns.
        """
        X_scaled = X.copy()
        for code, factor in self.scaling_map.items():
            for i in [1, 2, 3]:
                col = f"{code}_{i}"
                if col in X_scaled.columns:
                    X_scaled[col] *= factor
        return X_scaled


def build_preprocessing_pipeline(scaling_map: Dict[str, float]) -> Pipeline:
    """
    Build a preprocessing pipeline that:
      - Scales each country's center columns by its factor, then standardizes them.
      - Imputes and one-hot encodes the 'factor' column.

    Arguments:
        scaling_map (Dict[str, float]): Mapping from 3-letter country code to scaling factor.

    Returns:
        Pipeline: An sklearn Pipeline wrapping a ColumnTransformer for preprocessing.
    """
    countries = ["AUS", "CAN", "GER", "FRA", "IND", "JAP", "USA", "UK"]
    demand_columns = [f"{code}_{i}" for code in countries for i in [1, 2, 3]]

    numeric_pipeline = Pipeline(
        [
            ("demand_scaler", DemandScalingTransformer(scaling_map)),
            ("standard_scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, demand_columns),
            ("cat", categorical_pipeline, ["factor"]),
        ],
        remainder="drop",
    )

    return Pipeline([("preprocessor", preprocessor)])


def load_data(data_path: str, scaling_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load demand dataset and scaling factors, returning raw DataFrame and scaling map.

    Arguments:
        data_path (str): File path to the demand data (Excel).
        scaling_path (str): File path to the scaling factors (Excel).

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]
        - df: Loaded demand DataFrame.
        - scaling_map: Mapping from country code to factor.
    """
    df = pd.read_excel(data_path, index_col=0).reset_index(drop=True)
    scaling = pd.read_excel(scaling_path, index_col=0).reset_index(drop=True)

    country_code_map = {
        "AUS": "Australia",
        "CAN": "Canada",
        "GER": "Germany",
        "FRA": "France",
        "IND": "India",
        "JAP": "Japan",
        "USA": "United States of America",
    }
    scaling_map = {}
    for code, country in country_code_map.items():
        scaling_map[code] = float(scaling.loc[scaling["Country"] == country, "Scaling Factor"])

    return df, scaling_map
