import unittest

import numpy as np
import pandas as pd

from model.src.utils.helpers import stratified_split


class TestStratifiedSplit(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randint(0, 10, 100),
                "target": np.random.randint(0, 1000, 100),
            }
        )

    def test_shapes(self):
        X_train, X_test, y_train, y_test = stratified_split(
            self.df, "target", test_size=0.2, random_state=42
        )
        self.assertEqual(len(X_train) + len(X_test), len(self.df))
        self.assertEqual(len(y_train) + len(y_test), len(self.df))
        self.assertEqual(X_train.shape[1], self.df.shape[1] - 1)
        self.assertEqual(X_test.shape[1], self.df.shape[1] - 1)

    def test_stratification(self):
        _, _, y_train, y_test = stratified_split(
            self.df, "target", test_size=0.2, random_state=42, n_bins=5
        )
        train_bins = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
        test_bins = pd.qcut(y_test, q=5, labels=False, duplicates="drop")
        train_dist = train_bins.value_counts(normalize=True).sort_index()
        test_dist = test_bins.value_counts(normalize=True).sort_index()
        for t, s in zip(train_dist, test_dist):
            self.assertAlmostEqual(t, s, delta=0.15)

    def test_no_target_leakage(self):
        X_train, X_test, _, _ = stratified_split(self.df, "target")
        self.assertNotIn("target", X_train.columns)
        self.assertNotIn("target", X_test.columns)
