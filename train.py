"""Train, compare and package a reproducible UCI heart-disease classifier."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from modeling import FEATURES, clean_features, make_preprocessor

ROOT = Path(__file__).resolve().parent
SEED = 42


def scores(y_true, probability, threshold=0.5):
    predicted = (np.asarray(probability) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }


def main():
    data = pd.read_csv(ROOT / "heart_disease_uci.csv")
    x = clean_features(data)
    y = (data["num"] > 0).astype(int)
    sites = data["dataset"]
    x_train, x_test, y_train, y_test, site_train, site_test = train_test_split(
        x, y, sites, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "Logistic regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED),
            {"model__C": [0.05, 0.2, 1, 5], "model__l1_ratio": [0, 1]},
        ),
        "Random forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [5, 10, None],
             "model__min_samples_leaf": [2, 5], "model__max_features": ["sqrt"]},
        ),
        "Histogram gradient boosting": (
            HistGradientBoostingClassifier(random_state=SEED),
            {"model__learning_rate": [0.03, 0.08], "model__max_leaf_nodes": [7, 15],
             "model__l2_regularization": [0, 1]},
        ),
        "XGBoost": (
            XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          random_state=SEED, n_jobs=1),
            {"model__n_estimators": [150, 300], "model__max_depth": [2, 4],
             "model__learning_rate": [0.03, 0.1]},
        ),
    }
    comparisons = []
    winners = {}
    for name, (classifier, grid) in candidates.items():
        search = GridSearchCV(
            Pipeline([("preprocess", make_preprocessor()), ("model", classifier)]),
            grid, cv=cv, scoring={"f1": "f1", "roc_auc": "roc_auc"},
            refit="f1", n_jobs=-1, error_score="raise",
        )
        search.fit(x_train, y_train)
        row = search.best_index_
        comparisons.append({
            "model": name,
            "cv_f1": float(search.cv_results_["mean_test_f1"][row]),
            "cv_auc": float(search.cv_results_["mean_test_roc_auc"][row]),
            "best_parameters": search.best_params_,
        })
        winners[name] = search.best_estimator_

    selected = max(comparisons, key=lambda item: (item["cv_f1"], item["cv_auc"]))
    model = winners[selected["model"]]
    # Select the decision cutoff from training-only out-of-fold predictions.
    oof = cross_val_predict(model, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thresholds = np.arange(0.30, 0.705, 0.01)
    threshold = float(max(thresholds, key=lambda value: (f1_score(y_train, oof >= value), -abs(value - 0.5))))
    probability = model.predict_proba(x_test)[:, 1]
    test_scores = scores(y_test, probability, threshold)
    fpr, tpr, _ = roc_curve(y_test, probability)

    # The source hospital is excluded as a predictor. This check exposes transport limitations.
    site_checks = []
    for site in sorted(sites.unique()):
        train_mask = sites != site
        test_mask = sites == site
        site_classifier, site_grid = candidates[selected["model"]]
        site_search = GridSearchCV(
            Pipeline([("preprocess", make_preprocessor()), ("model", site_classifier)]),
            site_grid, cv=cv, scoring="f1", refit=True, n_jobs=-1, error_score="raise",
        )
        site_search.fit(x.loc[train_mask], y.loc[train_mask])
        p = site_search.best_estimator_.predict_proba(x.loc[test_mask])[:, 1]
        result = scores(y.loc[test_mask], p, 0.5)
        site_checks.append({"site": site, "n": int(test_mask.sum()),
                            "positive_rate": float(y.loc[test_mask].mean()),
                            "f1": result["f1"], "recall": result["recall"],
                            "roc_auc": result["roc_auc"]})

    importance = permutation_importance(
        model, x_test, y_test, scoring="roc_auc", n_repeats=12, random_state=SEED, n_jobs=-1
    )
    importance_rows = sorted(
        [{"feature": name, "mean_auc_drop": float(mean), "std": float(std)}
         for name, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
        key=lambda row: row["mean_auc_drop"], reverse=True,
    )
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(data)), "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)), "positive_rate": float(y.mean()),
        "selected_model": selected["model"], "threshold": threshold,
        "comparison": comparisons, "test": test_scores,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "site_checks": site_checks, "importance": importance_rows,
    }
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, output / "heart_model.joblib")
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": selected["model"], "threshold": threshold,
                      "test": test_scores, "site_checks": site_checks}, indent=2))


if __name__ == "__main__":
    main()
