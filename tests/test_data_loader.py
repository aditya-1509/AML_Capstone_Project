"""Unit and sanity tests for data loading, schema validation, and cleaning rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    FEATURES,
    NUMERIC_FEATURES,
    clean_features,
    deduplicate_records,
    load_clean_dataset,
    load_dataset,
    validate_dataset,
    validate_features,
)


def test_load_dataset_loads_uci_file():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 920
    assert "num" in df.columns
    assert "dataset" in df.columns


def test_deduplicate_records():
    df = load_dataset()
    dedup = deduplicate_records(df)
    assert len(dedup) == 918
    assert len(df) - len(dedup) == 2


def test_clean_features_zero_replacement():
    df = load_dataset()
    dedup = deduplicate_records(df)
    cleaned = clean_features(dedup)

    # In raw data, trestbps has 0s or chol has 0s (especially in Hungary and VA cohorts)
    assert (cleaned["trestbps"] == 0).sum() == 0
    assert (cleaned["chol"] == 0).sum() == 0
    assert cleaned["trestbps"].isna().sum() > 0
    assert cleaned["chol"].isna().sum() > 0


def test_validate_dataset_validates_raw():
    df = load_dataset()
    # Should complete without error
    validate_dataset(df)


def test_validate_dataset_catches_invalid_target():
    df = load_dataset().copy()
    df.loc[0, "num"] = 99  # Invalid severity value
    with pytest.raises(ValueError, match="Target 'num' must contain only integers"):
        validate_dataset(df)


def test_validate_dataset_catches_duplicate_ids():
    df = load_dataset().copy()
    df.loc[1, "id"] = df.loc[0, "id"]
    with pytest.raises(ValueError, match="Source patient row IDs must be unique"):
        validate_dataset(df)


def test_validate_features_catches_out_of_bounds():
    df = load_dataset().copy()
    df.loc[0, "age"] = 150  # Out of range [18, 100]
    errors = validate_features(df)
    assert any("age" in err and "outside" in err for err in errors)


def test_load_clean_dataset_pipeline():
    dedup, cleaned = load_clean_dataset()
    assert len(dedup) == 918
    assert len(cleaned) == 918
    assert list(cleaned.columns) == FEATURES
