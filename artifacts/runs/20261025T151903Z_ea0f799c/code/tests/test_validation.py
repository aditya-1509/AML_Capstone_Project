"""Trace actual search/prediction calls on synthetic rows; no real-data fitting."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from bayesian_network import HeartBayesianNetwork
from advanced_modeling import clean_features, make_preprocessor
from train import (
    PROTOCOL, best_configuration, choose_threshold, develop, evaluate_frozen,
    evaluate_hospitals, make_candidates, prepare_partitions,
)

EVENTS = []


class RecordingPreprocessor(TransformerMixin, BaseEstimator):
    def fit(self, x, y=None):
        EVENTS.append(("preprocess_fit", set(x.index), set()))
        return self

    def transform(self, x):
        return x


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, offset=0.0):
        self.offset = offset

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        self.fit_ids_ = set(x.index)
        EVENTS.append(("model_fit", self.fit_ids_, set()))
        return self

    def predict_proba(self, x):
        EVENTS.append(("predict", self.fit_ids_, set(x.index)))
        p = np.clip(0.15 + 0.7 * (x.index.to_numpy() % 2) + self.offset, 0.01, 0.99)
        return np.column_stack([1 - p, p])


class ValidationIsolationTests(unittest.TestCase):
    def setUp(self):
        i = np.arange(160)
        self.raw = pd.DataFrame({
            "id": i + 1000, "age": 40 + i % 30, "trestbps": 120 + i % 20,
            "chol": 180 + i % 50, "thalch": 130 + i % 40, "oldpeak": (i % 4).astype(float),
            "sex": "Male", "cp": "asymptomatic", "fbs": False, "restecg": "normal",
            "exang": False, "slope": "flat", "ca": np.nan, "thal": np.nan,
            "num": i % 2, "dataset": np.repeat(PROTOCOL["hospital_evaluation"]["site_order"], 40),
        })
        self.development, self.test = prepare_partitions(self.raw)
        self.candidates = {
            name: (Pipeline([("preprocess", RecordingPreprocessor()), ("model", RecordingClassifier())]),
                   {"model__offset": [0.0, 0.05]})
            for name in ["Logistic regression", "Random forest"]
        }
        EVENTS.clear()

    def run_development(self):
        return develop(self.development, self.candidates, forbidden_ids=self.test["id"])

    def test_nested_search_and_all_preprocessing_exclude_outer_rows_and_cutoff_rows(self):
        result = self.run_development()
        selection = set(result["selection_source_ids"])
        cutoff = set(result["cutoff_source_ids"])
        self.assertTrue(selection.isdisjoint(cutoff | set(self.test["id"])))
        index_by_id = dict(zip(self.raw["id"], self.raw.index))
        allowed_indices = {index_by_id[i] for i in selection}
        for kind, fitted, predicted in EVENTS:
            if kind.endswith("fit"):
                self.assertTrue(fitted <= allowed_indices)
            else:
                self.assertTrue(fitted.isdisjoint(predicted))
        for name in self.candidates:
            records = [row for row in result["oof"] if row["model"] == name]
            self.assertEqual(len(records), len(selection))
            self.assertEqual({row["source_id"] for row in records}, selection)
            for search in [s for s in result["searches"] if s["model"] == name and s["fold"] is not None]:
                held = {r["source_id"] for r in records if r["fold"] == search["fold"]}
                fitted = set(search["train_source_ids"])
                self.assertTrue(fitted.isdisjoint(held))
                self.assertEqual(fitted | held, selection)
            final = [s for s in result["searches"] if s["model"] == name and s["fold"] is None]
            self.assertEqual(set(final[0]["train_source_ids"]), selection)

    def test_cutoff_labels_cannot_change_family_or_hyperparameter_selection(self):
        first = self.run_development()
        index_by_id = dict(zip(self.raw["id"], self.raw.index))
        selection = self.raw.loc[[index_by_id[i] for i in first["selection_source_ids"]]].copy()
        cutoff = self.raw.loc[[index_by_id[i] for i in first["cutoff_source_ids"]]].copy()
        cutoff["num"] = 1 - cutoff["num"]
        # Hold the already assigned partitions fixed; only C's labels change.
        with patch("train.train_test_split", return_value=(selection, cutoff)):
            second = self.run_development()
        self.assertEqual(first["selected_model"], second["selected_model"])
        self.assertEqual(first["comparison"], second["comparison"])
        self.assertEqual(first["oof"], second["oof"])
        self.assertEqual(first["cutoff_probabilities"], second["cutoff_probabilities"])
        for name in self.candidates:
            self.assertEqual(first["models"][name].get_params()["model__offset"], second["models"][name].get_params()["model__offset"])
            self.assertNotEqual(first["thresholds"][name]["threshold"], second["thresholds"][name]["threshold"])

    def test_hospitals_select_locally_without_retrospective_or_held_site_rows(self):
        actual_develop = develop
        calls = []

        def trace_local(data, *args, **kwargs):
            start = len(EVENTS)
            local = actual_develop(data, *args, **kwargs)
            selection_indices = set(data.loc[data["id"].isin(local["selection_source_ids"])].index)
            for kind, fitted, _ in EVENTS[start:]:
                if kind.endswith("fit"):
                    self.assertTrue(fitted <= selection_indices)
            calls.append(set(data["dataset"]))
            return local

        with patch("train.develop", side_effect=trace_local):
            results = evaluate_hospitals(self.development, self.candidates, retrospective_ids=self.test["id"])
        for i, site in enumerate(PROTOCOL["hospital_evaluation"]["site_order"]):
            self.assertNotIn(site, calls[i])
            local = results[site]["development"]
            used = set(local["selection_source_ids"]) | set(local["cutoff_source_ids"])
            held = set(results[site]["evaluation"]["source_ids"])
            self.assertTrue(used.isdisjoint(held | set(self.test["id"])))
            self.assertEqual(held, set(self.development.loc[self.development["dataset"] == site, "id"]))
            self.assertEqual(local["selected_model"], results[site]["evaluation"]["selected_model"])
        with self.assertRaisesRegex(ValueError, "retrospective"):
            evaluate_hospitals(self.raw, self.candidates, retrospective_ids=self.test["id"])

    def test_retrospective_scoring_does_not_fit_select_or_retune(self):
        result = self.run_development()
        EVENTS.clear()
        with patch("train.search_model", side_effect=AssertionError("Evaluation searched")), patch("train.choose_threshold", side_effect=AssertionError("Evaluation retuned")):
            scored = evaluate_frozen(self.test, result)
        self.assertIn("Retrospective", scored["label"])
        self.assertEqual(scored["selected_model"], result["selected_model"])
        self.assertTrue(all(kind == "predict" for kind, _, _ in EVENTS))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate_frozen(self.development, result)

    def test_small_bn_grid_uses_the_same_nested_selection_helpers(self):
        candidates = {
            "Bayesian network": (HeartBayesianNetwork(allow_provisional=True), {"alpha": [1.0]}),
            "Logistic regression": (
                Pipeline([("preprocess", make_preprocessor()), ("model", LogisticRegression(solver="liblinear", random_state=42))]),
                {"model__C": [1.0]},
            ),
        }
        result = develop(self.development, candidates, forbidden_ids=self.test["id"])
        self.assertEqual(set(result["models"]), set(candidates))
        self.assertTrue(result["frozen"])
        self.assertEqual(len(result["oof"]), len(candidates) * len(result["selection_source_ids"]))
        self.assertFalse(result["models"]["Bayesian network"].graph_["approved"])
        # Actual existing preprocessing learned its median on S, not C or T.
        selection = self.development.loc[self.development["id"].isin(result["selection_source_ids"])]
        imputer = result["models"]["Logistic regression"]["preprocess"].named_transformers_["numeric"]["impute"]
        np.testing.assert_allclose(imputer.statistics_, clean_features(selection).iloc[:, :5].median().to_numpy())

    def test_declared_tie_rules_and_unresolved_real_grid_gate(self):
        self.assertEqual(best_configuration({"mean_test_f1": [0.5, 0.5], "mean_test_roc_auc": [0.6, 0.6], "params": [{"x": 2}, {"x": 1}]}), 1)
        self.assertEqual(choose_threshold(np.array([0, 1]), np.array([0.2, 0.8]))["grid_index"], 50)
        with self.assertRaisesRegex(ValueError, "unresolved"):
            make_candidates()

    def test_experiment_entry_requires_both_explicit_flags(self):
        from train import main

        with patch("experiment.run_experiment") as run:
            for arguments in [["train.py"], ["train.py", "--run-experiment"]]:
                with patch("sys.argv", arguments), self.assertRaises(SystemExit):
                    main()
            run.assert_not_called()
            with patch("sys.argv", ["train.py", "--run-experiment", "--allow-provisional-bn"]):
                main()
            run.assert_called_once_with(allow_provisional=True)

    def test_saved_prediction_bootstrap_matches_sklearn_with_ties(self):
        from experiment import bootstrap_intervals
        from sklearn.metrics import average_precision_score, roc_auc_score

        y = np.array([0, 1] * 10)
        p = np.array([0.2, 0.2, 0.6, 0.6, 0.8] * 4)
        actual = bootstrap_intervals(y, p, 0.5, repetitions=1050)
        samples = np.random.default_rng(47).integers(0, len(y), size=(1050, len(y)))
        auc, ap = [], []
        for sample in samples:
            if len(np.unique(y[sample])) == 2:
                auc.append(roc_auc_score(y[sample], p[sample]))
            ap.append(average_precision_score(y[sample], p[sample]))
        self.assertEqual(actual["roc_auc"]["valid_resamples"], len(auc))
        np.testing.assert_allclose(actual["roc_auc"]["percentile_95"], np.percentile(auc, [2.5, 97.5]))
        np.testing.assert_allclose(actual["average_precision"]["percentile_95"], np.percentile(ap, [2.5, 97.5]))


if __name__ == "__main__":
    unittest.main()
