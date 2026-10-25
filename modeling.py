"""Training-fold preprocessing and fixed conventional ML baselines."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor():
    """Return unfitted transforms; fit these inside each training fold only."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_baseline_models():
    """Return independent, unfitted pipelines with fixed baseline settings.

    Inputs follow data_loader.clean_features(). These settings are a starting
    point for later tuning, rather than the result of a hyperparameter search.
    """
    estimators = {
        "Logistic regression": LogisticRegression(
            solver="liblinear", C=1.0, max_iter=2000, random_state=42,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=1,
        ),
    }
    return {
        name: Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }
