"""Focused checks for the training/inference data contract."""

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modeling import (
    FEATURES, clean_features, deduplicate_records, split_records,
    validate_dataset, validate_features,
)


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

    def test_missing_boolean_predictors_are_allowed_but_invalid_text_is_rejected(self):
        for missing in [np.nan, None, pd.NA]:
            with self.subTest(missing=missing):
                valid = self.row.copy()
                valid["fbs"] = pd.Series([missing], dtype=object)
                valid["exang"] = pd.Series([missing], dtype=object)
                self.assertEqual(validate_features(valid), [])
        invalid = self.row.copy()
        invalid.loc[0, "fbs"] = "maybe"
        invalid.loc[0, "exang"] = "unknown"
        self.assertEqual(len(validate_features(invalid)), 2)

    def test_repeated_record_with_different_id_is_removed(self):
        first = self.row.copy()
        first.insert(0, "id", 1)
        second = first.copy()
        second["id"] = 2
        records = pd.concat([first, second], ignore_index=True)
        self.assertEqual(len(deduplicate_records(records)), 1)

    def test_only_whitelisted_predictors_are_returned(self):
        source = self.row.assign(id=1, dataset="Cleveland", num=4, ca=1, thal="normal", extra="unused")
        self.assertEqual(clean_features(source).columns.tolist(), FEATURES)
        self.assertTrue(set(clean_features(source).columns).isdisjoint({"id", "dataset", "num", "ca", "thal", "extra"}))

    def test_forbidden_columns_cannot_be_added_to_feature_list(self):
        for column in ["id", "dataset", "num", "ca", "thal"]:
            with self.subTest(column=column), patch("modeling.FEATURES", FEATURES + [column]):
                with self.assertRaisesRegex(ValueError, "Forbidden predictors"):
                    clean_features(self.row)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "id": range(101, 121), "age": range(40, 60),
            "trestbps": 130, "chol": 245, "thalch": 150, "oldpeak": 1.0,
            "sex": "Male", "cp": "asymptomatic", "fbs": True,
            "restecg": "normal", "exang": False, "slope": "flat",
            "dataset": ["Cleveland", "Hungary"] * 10,
            "ca": np.nan, "thal": np.nan, "num": [0, 1, 2, 3, 4] * 4,
        })

    def test_all_original_target_values_are_accepted_without_mutation(self):
        before = self.data.copy(deep=True)
        validate_dataset(self.data)
        pd.testing.assert_frame_equal(self.data, before)

    def test_invalid_targets_are_rejected(self):
        for value in [np.nan, -1, 5, 0.5, np.inf, "1", "invalid", True]:
            with self.subTest(value=value):
                invalid = self.data.copy()
                invalid["num"] = invalid["num"].astype(object)
                invalid.loc[0, "num"] = value
                with self.assertRaisesRegex(ValueError, "num must contain"):
                    validate_dataset(invalid)

    def test_missing_schema_column_is_rejected(self):
        for column in ["id", "dataset", "num", "age", "ca", "thal"]:
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Missing dataset columns"):
                validate_dataset(self.data.drop(columns=column))

    def test_empty_dataset_and_duplicate_column_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_dataset(self.data.iloc[:0])
        with self.assertRaisesRegex(ValueError, "duplicate column"):
            validate_dataset(pd.concat([self.data, self.data[["num"]]], axis=1))

    def test_missing_or_repeated_source_ids_are_rejected(self):
        for value in [None, self.data.loc[1, "id"]]:
            with self.subTest(value=value):
                invalid = self.data.copy()
                invalid.loc[0, "id"] = value
                with self.assertRaisesRegex(ValueError, "Source row IDs"):
                    validate_dataset(invalid)

    def test_missing_or_blank_hospital_is_rejected(self):
        for value in [None, "", "  "]:
            with self.subTest(value=value):
                invalid = self.data.copy()
                invalid.loc[0, "dataset"] = value
                with self.assertRaisesRegex(ValueError, "Hospital names"):
                    validate_dataset(invalid)

    def test_existing_feature_validation_is_reused(self):
        invalid = self.data.copy()
        invalid.loc[0, "cp"] = "unrecognized"
        with self.assertRaisesRegex(ValueError, "cp"):
            validate_dataset(invalid)

    def test_hospital_missingness_counts_cleaned_values_without_writing(self):
        import eda

        self.data.loc[0, "chol"] = 0
        self.data.loc[1, "slope"] = np.nan
        with patch("eda.pd.read_csv", return_value=self.data), patch("eda.Path.write_text") as write, patch("builtins.print"):
            eda.main()
        write.assert_called_once()
        summary = json.loads(write.call_args.args[0])
        hospitals = {row["site"]: row for row in summary["site_summary"]}
        self.assertEqual(hospitals["Cleveland"]["missing_after_zero_cleaning"]["chol"], 1)
        self.assertEqual(hospitals["Hungary"]["missing_after_zero_cleaning"]["slope"], 1)
        for feature in FEATURES:
            self.assertEqual(
                sum(row["missing_after_zero_cleaning"][feature] for row in hospitals.values()),
                summary["missing_after_zero_cleaning"][feature],
            )

    def test_actual_dataset_passes_validation(self):
        source = pd.read_csv(Path(__file__).resolve().parents[1] / "heart_disease_uci.csv")
        validate_dataset(source)
        self.assertEqual(len(deduplicate_records(source)), 918)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed in this interpreter")
    def test_split_is_repeatable_disjoint_and_exhaustive(self):
        train, test = split_records(self.data)
        repeated_train, repeated_test = split_records(self.data)
        self.assertEqual(train["id"].tolist(), repeated_train["id"].tolist())
        self.assertEqual(test["id"].tolist(), repeated_test["id"].tolist())
        self.assertTrue(set(train["id"]).isdisjoint(test["id"]))
        self.assertEqual(set(train["id"]) | set(test["id"]), set(self.data["id"]))
        self.assertEqual((len(train), len(test)), (16, 4))
        self.assertEqual(set((train["num"] > 0).astype(int)), {0, 1})
        self.assertEqual(set((test["num"] > 0).astype(int)), {0, 1})

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed in this interpreter")
    def test_duplicate_indices_are_rejected_before_alignment(self):
        invalid = self.data.copy()
        invalid.index = [0] * len(invalid)
        with self.assertRaisesRegex(ValueError, "Row indices must be unique"):
            split_records(invalid)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed in this interpreter")
    def test_actual_split_matches_original_training_code(self):
        from sklearn.model_selection import train_test_split

        source = pd.read_csv(Path(__file__).resolve().parents[1] / "heart_disease_uci.csv")
        data = deduplicate_records(source)
        x = clean_features(data)
        y = (data["num"] > 0).astype(int)
        old_x_train, old_x_test, old_y_train, old_y_test, _, _ = train_test_split(
            x, y, data["dataset"], test_size=0.2, stratify=y, random_state=42
        )
        train, test = split_records(data, random_state=42)
        self.assertEqual(train.index.tolist(), old_x_train.index.tolist())
        self.assertEqual(test.index.tolist(), old_x_test.index.tolist())
        pd.testing.assert_frame_equal(clean_features(train), old_x_train)
        pd.testing.assert_frame_equal(clean_features(test), old_x_test)
        self.assertEqual((len(train), len(test)), (734, 184))
        self.assertEqual((int(old_y_train.sum()), int(old_y_test.sum())), (406, 102))
        self.assertTrue(set(train["id"]).isdisjoint(test["id"]))


if __name__ == "__main__":
    unittest.main()
