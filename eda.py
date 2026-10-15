"""Create an auditable dataset summary without using the target to fit a model."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modeling import FEATURES, clean_features, deduplicate_records

ROOT = Path(__file__).resolve().parent


def main() -> None:
    raw = pd.read_csv(ROOT / "heart_disease_uci.csv")
    data = deduplicate_records(raw)
    cleaned = clean_features(data)
    data["positive"] = (data["num"] > 0).astype(int)

    summary = {
        "source_rows": len(raw),
        "duplicate_records_removed": len(raw) - len(data),
        "analysis_rows": len(data),
        "features": FEATURES,
        "target_counts": {str(k): int(v) for k, v in data["positive"].value_counts().sort_index().items()},
        "missing_after_zero_cleaning": {k: int(v) for k, v in cleaned.isna().sum().items()},
        "site_summary": [
            {"site": str(site), "rows": len(group), "positive": int(group["positive"].sum()),
             "positive_rate": float(group["positive"].mean())}
            for site, group in data.groupby("dataset")
        ],
        "numeric_summary": {
            name: {"nonmissing": int(cleaned[name].count()),
                   "median": float(cleaned[name].median()),
                   "min": float(cleaned[name].min()),
                   "max": float(cleaned[name].max())}
            for name in ["age", "trestbps", "chol", "thalch", "oldpeak"]
        },
        "excluded_predictors": {
            "id": "row identifier", "dataset": "site reserved for shift check",
            "ca": f"{int(data['ca'].isna().sum())} of {len(data)} missing",
            "thal": f"{int(data['thal'].isna().sum())} of {len(data)} missing",
        },
    }
    output = ROOT / "artifacts" / "eda.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
