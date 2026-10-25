"""Check preprocessing leakage, partition isolation and baseline metric evidence."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from data_loader import clean_features, load_dataset
from modeling import make_baseline_models, make_preprocessor
from train_baselines import (
    classification_scores, cross_validate_baselines, prepare_partitions, run_baselines,
)


@pytest.fixture
def records():
    size = 40
    return pd.DataFrame({
        "id": np.arange(1, size + 1),
        "age": np.arange(40, 40 + size),
        "trestbps": [120, 140] * (size // 2),
        "chol": [0, 220, 245, 260] * (size // 4),
        "thalch": np.arange(120, 120 + size),
        "oldpeak": np.arange(size) / 20,
        "sex": ["Female", "Male"] * (size // 2),
        "cp": ["typical angina", "asymptomatic"] * (size // 2),
        "fbs": [True, False] * (size // 2),
        "restecg": "normal", "exang": False, "slope": "flat",
        "dataset": ["Cleveland", "Hungary"] * (size // 2),
        "ca": np.nan, "thal": np.nan,
        "num": [0, 2] * (size // 2),
    })


def test_preprocessor_does_not_learn_from_validation_or_forbidden_columns(records):
    training = clean_features(records.iloc[:20])
    validation = clean_features(records.iloc[20:]).assign(num=999, dataset="unused", id=-1)
    validation["age"] = 10000
    validation["cp"] = "unseen category"
    validation["fbs"] = np.nan
    validation["chol"] = np.nan
    preprocessor = make_preprocessor().fit(training)
    imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    original_statistics = imputer.statistics_.copy()
    transformed = preprocessor.transform(validation)
    assert imputer.statistics_[0] == training["age"].median()
    np.testing.assert_array_equal(imputer.statistics_, original_statistics)
    assert np.isfinite(transformed).all()
    feature_names = preprocessor.get_feature_names_out()
    assert not any("num" == name.split("__")[-1] for name in feature_names)
    assert not any(name.endswith("__id") or "dataset" in name for name in feature_names)


def test_models_have_independent_preprocessors_and_can_fit_missing_values(records):
    models = make_baseline_models()
    first, second = models.values()
    assert first.named_steps["preprocess"] is not second.named_steps["preprocess"]
    features = clean_features(records)
    for model in models.values():
        model.fit(features, (records["num"] > 0).astype(int))
        probabilities = model.predict_proba(features.iloc[:5])
        assert probabilities.shape == (5, 2)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_partitions_are_disjoint_reproducible_and_source_is_unchanged():
    raw = load_dataset()
    before = raw.copy(deep=True)
    partitions = prepare_partitions(raw)
    again = prepare_partitions(raw)
    assert {name: len(part) for name, part in partitions.items()} == {
        "selection": 550, "cutoff": 184, "retrospective": 184,
    }
    ids = [set(part["id"]) for part in partitions.values()]
    assert len(set.union(*ids)) == 918
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for name in partitions:
        pd.testing.assert_frame_equal(partitions[name], again[name])
    pd.testing.assert_frame_equal(raw, before)


def test_duplicate_records_stay_out_of_all_partitions(records):
    duplicate = records.iloc[[0]].assign(id=999)
    partitions = prepare_partitions(pd.concat([records, duplicate], ignore_index=True))
    combined = pd.concat(partitions.values())
    assert len(combined) == len(records)
    assert 999 not in combined["id"].values


def test_oof_evidence_reconstructs_fold_metrics_with_common_folds(records):
    results, predictions, folds = cross_validate_baselines(records)
    for name, result in results.items():
        rows = predictions[predictions["model"] == name]
        assert set(rows["source_id"]) == set(records["id"])
        assert rows["source_id"].is_unique
        for fold, saved_scores in zip(folds, result["fold_scores"]):
            scored = rows[rows["fold"] == fold["fold"]]
            assert set(scored["source_id"]) == set(fold["validation_source_ids"])
            actual = classification_scores(scored["target"], scored["probability"])
            for metric, score in actual.items():
                assert saved_scores[metric] == pytest.approx(score)
        for metric, mean in result["mean"].items():
            assert mean == pytest.approx(np.mean([row[metric] for row in result["fold_scores"]]))


def test_cv_rejects_insufficient_class_support(records):
    records["num"] = 0
    records.loc[0, "num"] = 1
    with pytest.raises(ValueError, match="Each class needs"):
        cross_validate_baselines(records)


def test_runner_scores_only_selection_and_preserves_existing_outputs(records, tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "run"
    records.to_csv(source, index=False)
    selected_ids = set(prepare_partitions(records)["selection"]["id"])
    original_cv = cross_validate_baselines

    def check_selection(selection):
        assert set(selection["id"]) == selected_ids
        return original_cv(selection)

    with patch("train_baselines.cross_validate_baselines", side_effect=check_selection) as cv:
        summary = run_baselines(source, output)
        cv.assert_called_once()
    manifest = json.loads((output / "partitions.json").read_text())
    saved = pd.read_csv(output / "oof_predictions.csv")
    assert set(saved["source_id"]) == selected_ids
    assert manifest["source_sha256"] == summary["source_sha256"]
    assert set(manifest["partition_source_ids"]["selection"]) == selected_ids
    original_metrics = (output / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_baselines(source, output)
    assert (output / "metrics.json").read_bytes() == original_metrics


def test_probability_threshold_includes_exact_half():
    metrics = classification_scores([0, 1], [0.49, 0.5])
    assert metrics["f1"] == 1
