"""Create a reproducible, descriptive data audit without fitting transforms.

Run ``python eda.py`` to write JSON and SVG summaries under
``artifacts/eda_current``. The original ``artifacts/eda.json`` is preserved.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import pandas as pd

from modeling import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    clean_features,
    deduplicate_records,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parent
TARGET = "num"


def _write_bar_svg(path: Path, title: str, labels: list[str], values: list[float],
                   *, value_label: str = "Count") -> None:
    """Write a compact labelled horizontal bar chart using only the stdlib."""
    width, left, right, top, row_h = 900, 190, 30, 64, 34
    height = top + row_h * max(1, len(labels)) + 42
    max_value = max(values, default=0) or 1
    plot_width = width - left - right
    rows = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_h
        bar_width = max(0, plot_width * float(value) / max_value)
        rows.append(
            f'<text x="{left - 12}" y="{y + 19}" text-anchor="end" class="label">'
            f'{html.escape(str(label))}</text>'
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="22" rx="3" fill="#2f6f9f"/>'
            f'<text x="{left + bar_width + 8:.2f}" y="{y + 17}" class="value">{value:.3g}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font-family:Arial,sans-serif;fill:#18212b}}.title{{font-size:22px;font-weight:700}}.label{{font-size:14px}}.value{{font-size:13px}}</style>
<rect width="100%" height="100%" fill="white"/><text x="24" y="34" class="title">{html.escape(title)}</text>
{''.join(rows)}<text x="{left}" y="{height - 12}" class="label">{html.escape(value_label)}</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def build_report(csv_path: Path) -> tuple[dict, dict[str, tuple[str, list[str], list[float]]]]:
    raw = pd.read_csv(csv_path)
    validate_dataset(raw)
    data = deduplicate_records(raw)
    cleaned = clean_features(data)

    duplicate_removed = len(raw) - len(data)
    positive = (data[TARGET] > 0).astype(int)
    target_counts = positive.value_counts().reindex([0, 1], fill_value=0)
    class_count = int(target_counts.sum())
    missing = cleaned.isna().sum()
    sample_records = []
    for record in raw.head(5).to_dict(orient="records"):
        sample_records.append({
            name: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for name, value in record.items()
        })
    feature_types = {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "excluded_from_predictors": {
            "id": "source row identifier",
            "dataset": "collection site; retained for descriptive shift checks",
            "num": "outcome; never included in the feature matrix",
            "ca": f"excluded: {int(data['ca'].isna().sum())}/{len(data)} missing",
            "thal": f"excluded: {int(data['thal'].isna().sum())}/{len(data)} missing",
        },
    }
    site_rows = []
    for site, indices in data.groupby("dataset", sort=True).groups.items():
        site_mask = data.index.isin(indices)
        site_features = cleaned.loc[site_mask]
        site_target = positive.loc[site_mask]
        site_rows.append({
            "site": str(site),
            "rows": int(site_mask.sum()),
            "positive": int(site_target.sum()),
            "negative": int(site_target.eq(0).sum()),
            "positive_rate": float(site_target.mean()),
            "missing_by_feature": {name: int(count) for name, count in site_features.isna().sum().items()},
        })

    report = {
        "source_file": csv_path.name,
        "source_rows": int(len(raw)),
        "source_columns": list(raw.columns),
        "source_dtypes": {name: str(dtype) for name, dtype in raw.dtypes.items()},
        "sample_records": sample_records,
        "duplicate_analysis": {
            "duplicate_records_ignoring_id": duplicate_removed,
            "analysis_rows_after_deduplication": int(len(data)),
            "deduplication_rule": "exact duplicate across all columns except id; retained first source occurrence",
        },
        "data_quality_checks": {
            "schema_and_target_validation": "passed; required columns/types/domains checked by validate_dataset",
            "invalid_categories_or_out_of_range_values": "none detected; invalid records cause validation to stop",
            "unresolved_exact_duplicates_ignoring_id": 0,
            "source_ids_missing_or_repeated": 0,
            "site_labels_missing_or_blank": 0,
        },
        "feature_types": feature_types,
        "target": {
            "source_column": TARGET,
            "validated_values": [0, 1, 2, 3, 4],
            "binary_definition": "positive = num > 0; negative = num == 0",
            "counts": {str(k): int(v) for k, v in target_counts.items()},
            "positive_rate": float(target_counts.loc[1] / class_count) if class_count else None,
            "imbalance_ratio_majority_to_minority": float(target_counts.max() / target_counts.min()) if target_counts.min() else None,
        },
        "missing_values_after_documented_cleaning": {
            "rule": "zero trestbps/chol readings treated as missing; no rows dropped or values imputed",
            "by_feature": {name: int(count) for name, count in missing.items()},
            "percent_by_feature": {name: float(count / len(cleaned) * 100) for name, count in missing.items()},
        },
        "duplicate_rows_remaining_after_deduplication": int(data.duplicated(subset=data.columns.difference(["id"])).sum()),
        "categorical_values": {
            name: {str(k): int(v) for k, v in data[name].value_counts(dropna=False).items()}
            for name in CATEGORICAL_FEATURES
        },
        "numeric_descriptives_after_documented_cleaning": {
            name: {key: (None if pd.isna(value) else float(value)) for key, value in cleaned[name].describe().to_dict().items()}
            for name in NUMERIC_FEATURES
        },
        "numeric_pearson_correlations_after_documented_cleaning": {
            name: {other: (None if pd.isna(value) else float(value)) for other, value in row.items()}
            for name, row in cleaned[NUMERIC_FEATURES].corr().items()
        },
        "site_summary": site_rows,
        "preprocessing_handoff": {
            "deterministic_before_split": ["validate required schema and target domain", "deduplicate exact records ignoring id", "convert numeric fields", "treat zero trestbps/chol as missing"],
            "fit_only_inside_training_partitions": ["numeric median imputation", "missingness indicators", "numeric scaling", "categorical mode imputation", "one-hot encoding"],
            "excluded_predictors": ["id", "dataset", "num", "ca", "thal"],
            "no_train_test_split_or_learned_transform_fitted_by_this_report": True,
        },
    }
    plots = {
        "target_distribution.svg": ("Binary target distribution", ["Negative (num = 0)", "Positive (num > 0)"], [int(target_counts.loc[0]), int(target_counts.loc[1])]),
        "missing_values.svg": ("Missing values after documented cleaning", list(missing.index), [int(v) for v in missing.values]),
        "positive_rate_by_site.svg": ("Observed positive rate by collection site", [row["site"] for row in site_rows], [100 * row["positive_rate"] for row in site_rows]),
    }
    return report, plots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "heart_disease_uci.csv", help="CSV dataset path")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "eda_current", help="directory for JSON and SVG outputs")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report, plots = build_report(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for filename, (title, labels, values) in plots.items():
        _write_bar_svg(output_dir / filename, title, labels, values,
                       value_label="Positive rate (%)" if filename == "positive_rate_by_site.svg" else "Records")
    print(f"Wrote data audit and {len(plots)} plots to {output_dir}")


if __name__ == "__main__":
    main()
