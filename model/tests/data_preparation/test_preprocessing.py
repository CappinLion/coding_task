import unittest
from unittest.mock import patch

import pandas as pd

from model.src.data_preparation.preprocessing import (
    DemandScalingTransformer,
    build_preprocessing_pipeline,
    load_data,
)


class TestPreprocessing(unittest.TestCase):
    """
    Unit tests for DemandScalingTransformer, build_preprocessing_pipeline, and load_data.
    """

    def setUp(self):
        countries = ["AUS", "CAN", "GER", "FRA", "IND", "JAP", "USA", "UK"]
        demand_columns = [f"{code}_{i}" for code in countries for i in [1, 2, 3]]
        data = {col: [0, 0] for col in demand_columns}
        data.update(
            {
                "AUS_1": [1, 2],
                "AUS_2": [3, 4],
                "AUS_3": [5, 6],
            }
        )
        data["factor"] = ["factor1", "factor2"]
        self.df = pd.DataFrame(data)
        self.scaling_map = {"AUS": 2.0}

    def test_demand_scaling_transformer(self):
        transformer = DemandScalingTransformer(self.scaling_map)
        transformer = transformer.fit(self.df, y=None)
        transformed = transformer.transform(self.df)
        for i in [1, 2, 3]:
            col = f"AUS_{i}"
            expected = self.df[col] * 2.0
            pd.testing.assert_series_equal(transformed[col], expected)

    def test_demand_scaling_input_validation(self):
        transformer = DemandScalingTransformer(self.scaling_map)
        with self.assertRaises(ValueError):
            transformer.fit(X=[1, 2, 3], y=None)
        transformer_bad = DemandScalingTransformer("not a dict")
        with self.assertRaises(ValueError):
            transformer_bad.fit(self.df, y=None)

    def test_build_preprocessing_pipeline(self):
        pipeline = build_preprocessing_pipeline(self.scaling_map)
        output = pipeline.fit_transform(self.df)
        self.assertEqual(
            output.shape[1], len(self.df.columns) - 1 + 1
        )  # -1 for "factor", +1 for dummy
        factor_dummy = output[:, -1]  # Last column is the dummy for "factor"
        self.assertTrue(set(factor_dummy).issubset({0.0, 1.0}))

    @patch("pandas.read_excel")
    def test_load_data_excel(self, mock_read_excel):
        df_data = pd.DataFrame(
            {"AUS_1": [1], "AUS_2": [2], "AUS_3": [3], "factor": ["factor1"], "Sales": [100]}
        )
        df_scaling = pd.DataFrame({"Country": ["Australia"], "Scaling Factor": ["3.5"]})
        mock_read_excel.side_effect = [df_data, df_scaling]
        df_loaded, scaling_map_loaded = load_data("dummy_data_path.xlsx", "dummy_scaling_path.xlsx")
        self.assertIn("Sales", df_loaded.columns)
        self.assertAlmostEqual(scaling_map_loaded["AUS"], 3.5)
