"""Run development-only ML baselines: python train_baselines.py --help.

Uses the local project's historical 80/20 split (seed 42), then a 75/25
selection/cutoff split of development records (seed 43). Only selection rows
enter five-fold CV (seed 44); cutoff and retrospective evaluation rows are
reserved. Preprocessing is learned afresh inside every training fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from data_loader import (
    DEFAULT_DATA_PATH, FEATURES, clean_features, deduplicate_records,
    load_dataset, validate_dataset,
)
from modeling import make_baseline_models


def prepare_partitions(raw):
    """Validate and deduplicate before reproducing the shared partition rules."""
    validate_dataset(raw)
    data = deduplicate_records(raw)
    development, retrospective = train_test_split(
        data, test_size=0.2, stratify=(data["num"] > 0).astype(int), random_state=42,
    )
    selection, cutoff = train_test_split(
        development, test_size=0.25,
        stratify=(development["num"] > 0).astype(int), random_state=43,
    )
    return {"selection": selection, "cutoff": cutoff, "retrospective": retrospective}


def classification_scores(target, probability):
    """Score binary disease presence at the fixed probability cutoff of 0.50."""
    predicted = np.asarray(probability) >= 0.5
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
    }


def cross_validate_baselines(selection, *, n_splits=5):
    """Fit fixed models on common folds and retain out-of-fold probabilities.

    No candidate ranking, tuning, threshold selection or final holdout scoring
    happens here. Fold SD describes variation, not a confidence interval.
    """
    validate_dataset(selection)
    target = (selection["num"] > 0).astype(int)
    if n_splits < 2 or set(target.unique()) != {0, 1} or target.value_counts().min() < n_splits:
        raise ValueError("Each class needs at least n_splits records for stratified CV")
    features = clean_features(selection)
    folds = list(StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=44,
    ).split(features, target))
    fold_manifest = [
        {"fold": number, "validation_source_ids": selection.iloc[valid]["id"].tolist()}
        for number, (_, valid) in enumerate(folds, start=1)
    ]
    results, predictions = {}, []
    for name, pipeline in make_baseline_models().items():
        fold_scores = []
        for number, (train, valid) in enumerate(folds, start=1):
            fitted = clone(pipeline).fit(features.iloc[train], target.iloc[train])
            if not np.array_equal(fitted.classes_, [0, 1]):
                raise ValueError("Expected probability columns in class order [0, 1]")
            probability = fitted.predict_proba(features.iloc[valid])[:, 1]
            if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                raise ValueError("Predicted probabilities must be finite and between 0 and 1")
            fold_scores.append({"fold": number, **classification_scores(target.iloc[valid], probability)})
            predictions.extend(
                {
                    "model": name, "fold": number, "source_id": int(source_id),
                    "target": int(label), "probability": float(value),
                    "prediction": int(value >= 0.5),
                }
                for source_id, label, value in zip(
                    selection.iloc[valid]["id"], target.iloc[valid], probability,
                )
            )
        results[name] = {
            "parameters": pipeline.named_steps["model"].get_params(),
            "fold_scores": fold_scores,
            "mean": {
                metric: float(np.mean([score[metric] for score in fold_scores]))
                for metric in fold_scores[0] if metric != "fold"
            },
            "sd": {
                metric: float(np.std([score[metric] for score in fold_scores], ddof=1))
                for metric in fold_scores[0] if metric != "fold"
            },
        }
    return results, pd.DataFrame(predictions), fold_manifest


def run_baselines(data_path=DEFAULT_DATA_PATH, output_dir=None):
    """Execute baselines and write a new result directory without overwriting."""
    output = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "artifacts/ml_baselines"
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new --output directory: {output}")
    source = Path(data_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = load_dataset(source)
    partitions = prepare_partitions(raw)
    results, predictions, folds = cross_validate_baselines(partitions["selection"])
    summary = {
        "stage": "Fixed baseline cross-validation on selection records only",
        "target": "Disease presence: num > 0",
        "features": FEATURES,
        "source_sha256": source_hash,
        "source_rows": len(raw),
        "retained_rows": sum(len(part) for part in partitions.values()),
        "partition_sizes": {name: len(part) for name, part in partitions.items()},
        "seeds": {"historical_split": 42, "selection_cutoff_split": 43, "cv": 44, "estimator": 42},
        "cv_folds": 5,
        "decision_threshold": 0.5,
        "metric_aggregation": "Unweighted fold mean; sample SD (ddof=1), not a confidence interval",
        "preprocessing": "Training-fold median imputation, numeric missing indicators/scaling, categorical mode/one-hot encoding",
        "evaluation_status": "Cutoff and retrospective partitions were not fitted or scored by this run. The historical retrospective partition was previously inspected in the source project; it is not a new untouched test.",
        "validation_scope": "Row-stratified CV; no hospital-disjoint or clinical validation",
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        },
        "models": results,
    }
    manifest = {
        "source_sha256": source_hash,
        "partition_source_ids": {name: part["id"].tolist() for name, part in partitions.items()},
        "cv_folds": folds,
        "fold_rule": "Training IDs are selection IDs excluding the listed validation IDs, preserving selection order",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "partitions.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    rows = [
        {"model": name, **{f"{metric}_{stat}": value for stat in ("mean", "sd") for metric, value in result[stat].items()}}
        for name, result in results.items()
    ]
    pd.DataFrame(rows).to_csv(output / "cv_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Raw UCI CSV")
    parser.add_argument("--output", type=Path, help="New output directory (default: artifacts/ml_baselines)")
    args = parser.parse_args()
    summary = run_baselines(args.data, args.output)
    for name, result in summary["models"].items():
        print(f"{name}: CV F1={result['mean']['f1']:.4f}, ROC-AUC={result['mean']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
