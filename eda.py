"""Exploratory Data Analysis (EDA) pipeline for UCI Heart Disease project.

Produces a structured data audit (JSON) and clean vector visualizations (SVG)
under the ``artifacts/`` directory without fitting machine learning models or
leaking test set assumptions.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from data_loader import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    SITE_COL,
    TARGET,
    clean_features,
    deduplicate_records,
    load_dataset,
    validate_dataset,
)

ROOT = Path(__file__).resolve().parent


def _write_horizontal_bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    *,
    value_label: str = "Count",
    bar_color: str = "#2563eb",
    format_pct: bool = False,
) -> None:
    """Generate a clean vector horizontal bar chart using standard library XML."""
    width, left, right, top, row_h = 880, 210, 60, 70, 36
    height = top + row_h * max(1, len(labels)) + 45
    max_value = max(values, default=0) or 1
    plot_width = width - left - right

    rows: list[str] = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_h
        bar_w = max(0, plot_width * float(value) / max_value)
        val_str = f"{value:.1f}%" if format_pct else f"{value:,.0f}" if value.is_integer() else f"{value:.2f}"
        rows.append(
            f'  <text x="{left - 12}" y="{y + 20}" text-anchor="end" class="label">{html.escape(str(label))}</text>\n'
            f'  <rect x="{left}" y="{y + 2}" width="{bar_w:.2f}" height="22" rx="4" fill="{bar_color}"/>\n'
            f'  <text x="{left + bar_w + 10:.2f}" y="{y + 18}" class="val">{val_str}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; fill: #1e293b; }}
    .title {{ font-size: 19px; font-weight: 700; fill: #0f172a; }}
    .subtitle {{ font-size: 12px; fill: #64748b; }}
    .label {{ font-size: 13px; font-weight: 500; fill: #334155; }}
    .val {{ font-size: 12px; font-weight: 600; fill: #1e293b; }}
    .axis {{ font-size: 11px; fill: #64748b; font-weight: 600; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" rx="8"/>
  <text x="28" y="34" class="title">{html.escape(title)}</text>
  <text x="28" y="52" class="subtitle">UCI Heart Disease Clinical Dataset Audit</text>
{chr(10).join(rows)}
  <text x="{left}" y="{height - 14}" class="axis">{html.escape(value_label)}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_heatmap_svg(
    path: Path,
    title: str,
    features: list[str],
    corr_matrix: list[list[float]],
) -> None:
    """Generate a clean SVG correlation matrix heatmap."""
    size = len(features)
    cell_size = 72
    margin_left, margin_top = 130, 80
    width = margin_left + size * cell_size + 40
    height = margin_top + size * cell_size + 50

    cells: list[str] = []
    # Axis labels
    for i, name in enumerate(features):
        y = margin_top + i * cell_size + cell_size // 2
        cells.append(f'  <text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" class="label">{name}</text>')
        x = margin_left + i * cell_size + cell_size // 2
        cells.append(f'  <text x="{x}" y="{margin_top - 12}" text-anchor="middle" class="label">{name}</text>')

    for i in range(size):
        for j in range(size):
            r = corr_matrix[i][j]
            x = margin_left + j * cell_size
            y = margin_top + i * cell_size

            # Color gradient: positive (blue #2563eb), negative (red #dc2626), zero (white #ffffff)
            if r >= 0:
                alpha = min(1.0, max(0.0, r))
                color = f"rgba(37, 99, 235, {alpha * 0.85 + 0.05:.2f})"
            else:
                alpha = min(1.0, max(0.0, -r))
                color = f"rgba(220, 38, 38, {alpha * 0.85 + 0.05:.2f})"

            cells.append(f'  <rect x="{x}" y="{y}" width="{cell_size - 2}" height="{cell_size - 2}" rx="3" fill="{color}"/>')
            cells.append(f'  <text x="{x + cell_size//2}" y="{y + cell_size//2 + 5}" text-anchor="middle" class="val">{r:+.2f}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; fill: #1e293b; }}
    .title {{ font-size: 19px; font-weight: 700; fill: #0f172a; }}
    .subtitle {{ font-size: 12px; fill: #64748b; }}
    .label {{ font-size: 13px; font-weight: 600; fill: #334155; }}
    .val {{ font-size: 13px; font-weight: 700; fill: #0f172a; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff" rx="8"/>
  <text x="28" y="34" class="title">{html.escape(title)}</text>
  <text x="28" y="52" class="subtitle">Pearson Correlation Matrix across Continuous Predictors</text>
{chr(10).join(cells)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def generate_eda_report(raw: pd.DataFrame) -> tuple[dict, dict[str, tuple]]:
    """Perform full exploratory analysis, returning summary dictionary and plot specs."""
    validate_dataset(raw)
    data = deduplicate_records(raw)
    cleaned = clean_features(data)

    duplicates_removed = len(raw) - len(data)
    positive = (data[TARGET] > 0).astype(int)
    target_counts = positive.value_counts().reindex([0, 1], fill_value=0)
    multiclass_counts = data[TARGET].value_counts().sort_index().to_dict()

    missing_series = cleaned.isna().sum()

    # Per-hospital breakdown
    site_summaries = []
    for site, group in data.groupby(SITE_COL, sort=True):
        site_positive = (group[TARGET] > 0).astype(int)
        site_cleaned = cleaned.loc[group.index]
        site_summaries.append({
            "site": str(site),
            "records": len(group),
            "positive_count": int(site_positive.sum()),
            "negative_count": int((site_positive == 0).sum()),
            "positive_rate": float(site_positive.mean()),
            "missing_by_feature": {k: int(v) for k, v in site_cleaned.isna().sum().items()},
        })

    # Numeric descriptive statistics
    descriptives = {}
    for col in NUMERIC_FEATURES:
        series = cleaned[col].dropna()
        descriptives[col] = {
            "nonmissing_count": int(series.count()),
            "missing_count": int(cleaned[col].isna().sum()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "25%": float(series.quantile(0.25)),
            "50%_median": float(series.median()),
            "75%": float(series.quantile(0.75)),
            "max": float(series.max()),
        }

    # Correlation matrix
    corr_df = cleaned[NUMERIC_FEATURES].corr()
    corr_matrix = [[float(corr_df.loc[r, c]) for c in NUMERIC_FEATURES] for r in NUMERIC_FEATURES]

    # Categorical distributions by target
    cat_summary = {}
    for col in CATEGORICAL_FEATURES:
        cross = pd.crosstab(data[col], positive, dropna=False)
        cat_summary[col] = {
            str(cat): {
                "negative": int(cross.loc[cat, 0]) if 0 in cross.columns and cat in cross.index else 0,
                "positive": int(cross.loc[cat, 1]) if 1 in cross.columns and cat in cross.index else 0,
            }
            for cat in cross.index
        }

    report = {
        "dataset_name": "UCI Heart Disease (Multi-Center)",
        "source_rows": len(raw),
        "deduplicated_analysis_rows": len(data),
        "duplicates_removed": duplicates_removed,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded_predictors": {
                "id": "Arbitrary patient identifier",
                "dataset": "Clinical site / hospital cohort (reserved for shift validation)",
                "ca": f"Excluded due to severe missingness ({int(data['ca'].isna().sum())}/{len(data)} missing)",
                "thal": f"Excluded due to severe missingness ({int(data['thal'].isna().sum())}/{len(data)} missing)",
            },
        },
        "target": {
            "definition": "num > 0 indicates presence of heart disease (positive); num == 0 indicates absence (negative)",
            "binary_counts": {"negative_0": int(target_counts.loc[0]), "positive_1": int(target_counts.loc[1])},
            "positive_prevalence": float(target_counts.loc[1] / len(data)),
            "multiclass_stages": {f"stage_{k}": int(v) for k, v in multiclass_counts.items()},
        },
        "site_heterogeneity": site_summaries,
        "numeric_descriptives": descriptives,
        "categorical_breakdown": cat_summary,
        "missingness_summary": {
            "rule": "0 trestbps and 0 chol values converted to NaN prior to analysis",
            "missing_counts": {k: int(v) for k, v in missing_series.items()},
            "missing_percentages": {k: float(v / len(cleaned) * 100) for k, v in missing_series.items()},
        },
        "pearson_correlations": {
            row: {col: float(corr_df.loc[row, col]) for col in NUMERIC_FEATURES}
            for row in NUMERIC_FEATURES
        },
    }

    # Plot specifications: (type, title, labels/args, values/args)
    plots = {
        "target_distribution.svg": (
            "bar",
            "Target Class Distribution (num > 0 vs num == 0)",
            ["Negative (No Disease, num=0)", "Positive (Heart Disease, num>0)"],
            [float(target_counts.loc[0]), float(target_counts.loc[1])],
            {"value_label": "Number of Records", "bar_color": "#2563eb", "format_pct": False},
        ),
        "missing_values.svg": (
            "bar",
            "Missing Values by Feature (After Zero-as-Missing Cleaning)",
            list(missing_series.index),
            [float(v) for v in missing_series.values],
            {"value_label": "Missing Record Count", "bar_color": "#e11d48", "format_pct": False},
        ),
        "positive_rate_by_site.svg": (
            "bar",
            "Observed Heart Disease Positive Rate by Clinical Site",
            [s["site"] for s in site_summaries],
            [s["positive_rate"] * 100 for s in site_summaries],
            {"value_label": "Positive Prevalence (%)", "bar_color": "#059669", "format_pct": True},
        ),
        "correlation_heatmap.svg": (
            "heatmap",
            "Inter-Feature Pearson Correlation Heatmap",
            NUMERIC_FEATURES,
            corr_matrix,
            {},
        ),
    }

    return report, plots


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "heart_disease_uci.csv", help="Path to raw dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts", help="Directory for output audit & plots")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and analyzing dataset from: {data_path}")
    raw = load_dataset(data_path)
    report, plots = generate_eda_report(raw)

    audit_path = output_dir / "data_audit.json"
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote structured data audit to: {audit_path}")

    for filename, (plot_type, title, arg1, arg2, kwargs) in plots.items():
        file_path = output_dir / filename
        if plot_type == "bar":
            _write_horizontal_bar_svg(file_path, title, arg1, arg2, **kwargs)
        elif plot_type == "heatmap":
            _write_heatmap_svg(file_path, title, arg1, arg2)
        print(f"Generated plot: {file_path}")

    print("EDA audit pipeline completed successfully!")


if __name__ == "__main__":
    main()
