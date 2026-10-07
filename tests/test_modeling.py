"""Focused checks for the training/inference data contract."""

import unittest

import numpy as np
import pandas as pd

from modeling import FEATURES, clean_features, validate_features


class FeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.row = pd.DataFrame([{
            "age": 54, "trestbps": 0, "chol": 0, "thalch": 150, "oldpeak": 1.0,
            "sex": "Male", "cp": "asymptomatic", "fbs": "TRUE", "restecg": "normal",
            "exang": "FALSE", "slope": "flat",
        }])[FEATURES]

    def test_disguised_zero_values_are_imputed_as_missing(self):
        cleaned = clean_features(self.row)
        self.assertTrue(np.isnan(cleaned.loc[0, "trestbps"]))
        self.assertTrue(np.isnan(cleaned.loc[0, "chol"]))
        self.assertIs(cleaned.loc[0, "fbs"], True)
        self.assertIs(cleaned.loc[0, "exang"], False)

    def test_batch_validation_rejects_missing_age_and_unknown_category(self):
        invalid = self.row.copy()
        invalid.loc[0, "age"] = np.nan
        invalid.loc[0, "cp"] = "unknown chest pain"
        errors = validate_features(invalid)
        self.assertTrue(any("age" in error for error in errors))
        self.assertTrue(any("cp" in error for error in errors))

    def test_missing_optional_values_are_accepted(self):
        valid = self.row.copy()
        valid.loc[0, "oldpeak"] = np.nan
        valid.loc[0, "slope"] = np.nan
        self.assertEqual(validate_features(valid), [])


if __name__ == "__main__":
    unittest.main()
