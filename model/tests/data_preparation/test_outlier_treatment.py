import unittest

import pandas as pd

from model.src.data_preparation.outlier_treatment import treat_outliers


class TestOutlierTreatment(unittest.TestCase):
    """
    Unit tests for the treat_outliers function.
    """

    def test_default_outlier_treatment(self):
        df = pd.DataFrame({"x": [0, 100, 200, 300, 400]})
        treated = treat_outliers(df)
        low, high = df["x"].quantile(0.01), df["x"].quantile(0.99)
        self.assertGreaterEqual(treated["x"].min(), low)
        self.assertLessEqual(treated["x"].max(), high)

    def test_specific_columns(self):
        df = pd.DataFrame({"a": [0, 5, 10, 15, 20], "b": [100, 200, 300, 400, 500]})
        treated = treat_outliers(df, cols=["a"], lower_quantile=0.2, upper_quantile=0.8)
        low_a, high_a = df["a"].quantile(0.2), df["a"].quantile(0.8)
        expected_a = df["a"].clip(lower=low_a, upper=high_a).tolist()
        self.assertListEqual(treated["a"].tolist(), expected_a)
        self.assertListEqual(treated["b"].tolist(), df["b"].tolist())
