"""Read-only access to the verified study and its frozen inference models."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from advanced_modeling import FEATURES, clean_features, deduplicate_records, validate_features

ROOT = Path(__file__).resolve().parent
RUN_ID = "20261025T151903Z_ea0f799c"
RUN = ROOT / "artifacts" / "runs" / RUN_ID
MODEL_NAMES = ("Random forest", "Logistic regression", "Histogram gradient boosting", "XGBoost", "Bayesian network")
GROUPS = ("pooled", "hospital_Cleveland", "hospital_Hungary", "hospital_Switzerland", "hospital_VA_Long_Beach")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def evidence():
    """Fail closed on missing, mismatched or unverified study artifacts."""
    manifest = read_json(RUN / "manifest.json")
    verification = read_json(RUN / "verification.json")
    if manifest["status"] != "complete" or verification["status"] != "PASS":
        raise ValueError("The saved study has not passed verification")
    if hashlib.sha256((ROOT / "heart_disease_uci.csv").read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("The source dataset does not match the verified study")
    freeze = read_json(RUN / "pooled/freeze.json")
    metrics = read_json(RUN / "pooled/metrics.json")
    if freeze["selected_model"] != "Random forest" or metrics["selected_model"] != freeze["selected_model"]:
        raise ValueError("The saved model selection is inconsistent")
    thresholds = read_json(RUN / "pooled/thresholds.json")
    for name in MODEL_NAMES:
        cutoff = freeze["thresholds"][name]
        if not 0 <= cutoff <= 1 or cutoff != metrics["families"][name]["threshold"] or cutoff != thresholds[name]["threshold"]:
            raise ValueError(f"The saved cutoff for {name} is inconsistent")
    return {"manifest": manifest, "freeze": freeze, "metrics": metrics,
            "comparison": read_json(RUN / "pooled/comparison.json")}


@lru_cache(maxsize=5)
def frozen_model(name):
    if name not in MODEL_NAMES:
        raise ValueError("Choose a model from the verified study")
    study = evidence()
    path = RUN / "pooled/models" / (name.replace(" ", "_") + ".joblib")
    expected = study["freeze"]["model_sha256"][name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"The saved {name} model failed its integrity check")
    model = joblib.load(path)
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("The saved model has an unexpected class order")
    return model


def predict_profile(profile, *, compare=False):
    """Infer on a fictional profile. Never fit, log or persist profile values."""
    row = pd.DataFrame([profile], columns=FEATURES)
    errors = validate_features(row)
    if errors:
        raise ValueError("; ".join(errors))
    features = clean_features(row)
    names = MODEL_NAMES if compare else ("Random forest",)
    predictions = []
    for name in names:
        probability = float(frozen_model(name).predict_proba(features)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The model returned an invalid score")
        cutoff = evidence()["freeze"]["thresholds"][name]
        predictions.append({"Model": name, "Score": probability, "Study cutoff": cutoff,
                            "Decision": "Above cutoff" if probability >= cutoff else "Below cutoff"})
    return pd.DataFrame(predictions)


@lru_cache(maxsize=1)
def cohort():
    data = deduplicate_records(pd.read_csv(ROOT / "heart_disease_uci.csv"))
    cleaned = clean_features(data)
    data = data.copy()
    data[FEATURES] = cleaned
    data["Disease presence"] = np.where(data["num"] > 0, "Recorded positive", "Recorded negative")
    return data


@lru_cache(maxsize=5)
def evaluation_rows(group="pooled"):
    if group not in GROUPS:
        raise ValueError("Unknown evaluation group")
    return pd.read_csv(RUN / group / "evaluation_predictions.csv")


def threshold_scores(model, cutoff, group="pooled"):
    """Explore saved predictions without changing the study's frozen choices."""
    if model not in MODEL_NAMES or not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Choose a known model and a cutoff between zero and one")
    rows = evaluation_rows(group)
    rows = rows.loc[rows["model"] == model]
    y, probability = rows["target"], rows["probability"]
    predicted = probability >= cutoff
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": len(rows)}


def model_table():
    source = evidence()
    cv = {row["model"]: row["cv_f1"] for row in source["comparison"]}
    return pd.DataFrame([
        {"Model": name, "Nested CV F1": cv[name],
         **{key: family["tuned"][key] for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc")},
         "Study cutoff": family["threshold"]}
        for name in MODEL_NAMES for family in [source["metrics"]["families"][name]]
    ])


def hospital_table():
    rows = []
    for group in GROUPS[1:]:
        data = read_json(RUN / group / "metrics.json")
        name = data["selected_model"]
        score = data["families"][name]["tuned"]
        rows.append({"Hospital": group.removeprefix("hospital_").replace("_", " "),
                     "Model": name, "Records": data["n"], "Positive": data["positive"],
                     "F1": score["f1"], "Always positive F1": 2 * data["positive"] / (data["n"] + data["positive"]),
                     "Specificity": score["specificity"], "ROC-AUC": score["roc_auc"]})
    return pd.DataFrame(rows)
