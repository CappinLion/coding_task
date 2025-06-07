import unittest

import pandas as pd
from lightgbm import LGBMRegressor
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from model.src.model_training.pipelines import create_predictor, make_full_pipeline, opt_params


class TestPipelines(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "model": {
                    "estimator": "RandomForest",
                    "params": {
                        "RandomForest": {
                            "estimator__n_estimators": [5],
                            "estimator__non_valid_param": [0.5],
                        }
                    },
                    "n_iter": 1,
                    "cv_folds": 2,
                    "scoring": "r2",
                }
            }
        )
        codes = ["AUS", "CAN", "GER", "FRA", "IND", "JAP", "USA", "UK"]
        data = {}
        for c in codes:
            for i in (1, 2, 3):
                data[f"{c}_{i}"] = [1, 2, 3, 4]
        data["factor"] = ["factor1", "factor2", "factor1", "factor2"]
        self.X = pd.DataFrame(data)
        self.y = pd.Series([10.0, 20.0, 30.0, 40.0])
        self.scaling_map = {c: 1.0 for c in codes}

    def test_create_predictor(self):
        self.cfg.model.estimator = "RandomForest"
        pred = create_predictor(self.cfg)
        self.assertIsInstance(pred, RandomForestRegressor)

        self.cfg.model.estimator = "XGBoost"
        pred2 = create_predictor(self.cfg)
        self.assertIsInstance(pred2, XGBRegressor)

        self.cfg.model.estimator = "LightGBM"
        pred3 = create_predictor(self.cfg)
        self.assertIsInstance(pred3, LGBMRegressor)

        self.cfg.model.estimator = "Unknown"
        with self.assertRaises(ValueError):
            create_predictor(self.cfg)

    def test_make_full_pipeline(self):
        pipe = make_full_pipeline(self.cfg, self.scaling_map)
        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("preprocessor", pipe.named_steps)
        self.assertIn("estimator", pipe.named_steps)

    def test_opt_params_filters(self):
        pipe = make_full_pipeline(self.cfg, self.scaling_map)
        search = opt_params(pipe, self.cfg, self.X, self.y)
        params = search.param_distributions
        self.assertIn("estimator__n_estimators", params)
        self.assertNotIn("estimator__non_valid_param", params)
