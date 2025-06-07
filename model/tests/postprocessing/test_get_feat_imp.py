import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from model.src.postprocessing.main import get_feat_importance


class TestFeatureImportance(unittest.TestCase):
    def setUp(self):
        self.feature_names = ["feat1", "feat2", "feat3"]
        self.df = pd.DataFrame(np.zeros((5, 3)), columns=self.feature_names)

    def test_importances_sorted_descending(self):
        importances = np.array([0.2, 0.5, 0.3])
        estimator_mock = Mock()
        estimator_mock.feature_importances_ = importances

        pipeline_mock = Mock()
        pipeline_mock.named_steps = {"estimator": estimator_mock}

        feat_imp = get_feat_importance(pipeline_mock, self.df)

        expected = [("feat2", 0.5), ("feat3", 0.3), ("feat1", 0.2)]
        self.assertEqual(feat_imp, expected)

    def test_length_matches_columns(self):
        importances = np.array([1.0, 1.0, 1.0])
        estimator_mock = Mock()
        estimator_mock.feature_importances_ = importances

        pipeline_mock = Mock()
        pipeline_mock.named_steps = {"estimator": estimator_mock}

        feat_imp = get_feat_importance(pipeline_mock, self.df)
        self.assertEqual(len(feat_imp), 3)
        names = [name for name, _ in feat_imp]
        self.assertEqual(names, self.feature_names)
