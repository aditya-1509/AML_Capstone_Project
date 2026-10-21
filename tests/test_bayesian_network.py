"""Essential BN checks using synthetic rows only; no real-data fitting or writes."""

import copy
import pickle
import unittest

import numpy as np
import pandas as pd

from bayesian_network import HeartBayesianNetwork, validate_graph
from modeling import FEATURES


class BayesianNetworkTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({
            "age": [40, 42, 55, 60, 65, 70], "trestbps": [120, 125, 130, 135, 140, 145],
            "chol": [180, 190, 210, 220, 230, 240], "thalch": [170, 165, 150, 145, 140, 130],
            "oldpeak": [0, 0, 1, 2, 2, 3], "sex": "Male",
            "cp": ["typical angina"] * 2 + ["asymptomatic"] * 4,
            "fbs": [False, False, False, True, True, True], "restecg": "normal",
            "exang": [False, False, True, True, True, True],
            "slope": ["upsloping", "upsloping", "flat", "flat", "downsloping", "downsloping"],
        })
        self.y = pd.Series([0, 0, 1, 1, 1, 1])
        self.model = HeartBayesianNetwork(allow_provisional=True).fit(self.x, self.y)

    def test_graph_validity_and_explicit_provisional_opt_in(self):
        graph = self.model.graph_
        self.assertFalse(graph["approved"])
        self.assertEqual(graph["status"], "provisional")
        self.assertEqual(set(validate_graph(graph["edges"])), {"disease", *FEATURES})
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_graph(graph["edges"] + [["slope", "disease"]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_graph(graph["edges"] + [graph["edges"][0]])
        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_graph(graph["edges"] + [["id", "disease"]])
        with self.assertRaisesRegex(ValueError, "not approved"):
            HeartBayesianNetwork().fit(self.x, self.y)

    def test_smoothed_probability_tables_are_positive_and_normalized(self):
        for node, table in self.model.cpds_.items():
            with self.subTest(node=node):
                self.assertTrue(np.isfinite(table).all())
                self.assertTrue((table > 0).all())
                np.testing.assert_allclose(table.sum(axis=0), 1)
        np.testing.assert_allclose(self.model.cpds_["disease"], [3 / 8, 5 / 8])
        self.assertAlmostEqual(self.model.cpds_["cp"][3, 1], 5 / 8)

    def test_inference_matches_hand_calculated_posterior(self):
        # P(D=1)=5/8, P(cp=asymptomatic|D=1)=5/8 and P(cp|D=0)=1/6.
        self.assertAlmostEqual(self.model.query({"cp": "asymptomatic"}), 25 / 29)

    def test_missing_oldpeak_is_summed_in_the_dependent_slope_branch(self):
        slope = self.model.states_["slope"].index("flat")
        joint = np.array([
            self.model.cpds_["disease"][d] * sum(
                self.model.cpds_["oldpeak"][o, d] * self.model.cpds_["slope"][slope, d, o]
                for o in range(len(self.model.states_["oldpeak"]))
            ) for d in [0, 1]
        ])
        self.assertAlmostEqual(self.model.query({"slope": "flat"}), joint[1] / joint.sum())
        self.assertAlmostEqual(self.model.query({}), 5 / 8)
        self.assertAlmostEqual(self.model.query({"oldpeak": np.nan, "slope": None}), 5 / 8)

    def test_unseen_valid_states_unknown_categories_and_missing_training_values(self):
        # Female is a valid state absent from training, retained through smoothing.
        self.assertTrue(0 < self.model.query({"sex": "Female"}) < 1)
        with self.assertRaisesRegex(ValueError, "unrecognized category"):
            self.model.query({"cp": "unknown"})
        with self.assertRaisesRegex(ValueError, "unrecognized category"):
            self.model.query({"fbs": "maybe"})
        with self.assertRaisesRegex(ValueError, "eligible features"):
            self.model.query({"num": 1})
        incomplete = self.x.copy()
        incomplete.loc[0, "chol"] = 0  # Existing zero-to-missing convention.
        incomplete["fbs"] = incomplete["fbs"].astype(object)
        incomplete.loc[0, "fbs"] = None
        fitted = HeartBayesianNetwork(allow_provisional=True).fit(incomplete, self.y)
        self.assertTrue(np.isfinite(fitted.predict_proba(incomplete)).all())
        incomplete["age"] = np.nan
        with self.assertRaisesRegex(ValueError, "all training values are missing"):
            HeartBayesianNetwork(allow_provisional=True).fit(incomplete, self.y)

    def test_prediction_consistency_serialization_and_frozen_training_bins(self):
        before = copy.deepcopy(self.model.bin_edges_)
        probabilities = self.model.predict_proba(self.x)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(self.model.predict(self.x), probabilities[:, 1] >= 0.5)
        np.testing.assert_array_equal(self.model.predict_proba(self.x), probabilities)
        restored = pickle.loads(pickle.dumps(self.model))  # Memory only.
        np.testing.assert_array_equal(restored.predict_proba(self.x), probabilities)
        self.model.query({"age": 1000})
        for name in before:
            np.testing.assert_array_equal(before[name], self.model.bin_edges_[name])


if __name__ == "__main__":
    unittest.main()
