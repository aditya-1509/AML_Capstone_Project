"""Data ingestion and schema definitions for the UCI Heart Disease dataset."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# Core feature definitions
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "num"
ID_COL = "id"
SITE_COL = "dataset"
EXCLUDED_COLS = ["ca", "thal"]  # Excluded due to severe cross-hospital missingness

CATEGORY_OPTIONS = {
    "sex": ["Female", "Male"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "fbs": [False, True],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "exang": [False, True],
    "slope": ["upsloping", "flat", "downsloping"],
}

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "heart_disease_uci.csv"


def load_dataset(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load raw Heart Disease UCI dataset from CSV into a pandas DataFrame."""
    path = Path(filepath) if filepath else DEFAULT_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """Compute high-level dataset dimensions and site distribution."""
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "sites": df[SITE_COL].value_counts().to_dict() if SITE_COL in df.columns else {},
        "target_distribution": df[TARGET].value_counts().to_dict() if TARGET in df.columns else {},
    }


if __name__ == "__main__":
    df = load_dataset()
    overview = get_dataset_overview(df)
    print("=" * 60)
    print("Heart Disease UCI Dataset Ingestion Overview")
    print("=" * 60)
    print(f"Total Rows:    {overview['total_rows']}")
    print(f"Total Columns: {overview['total_columns']}")
    print(f"Sites:         {overview['sites']}")
    print(f"Target Counts: {overview['target_distribution']}")
    print("=" * 60)
