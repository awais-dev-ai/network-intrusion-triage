"""Tests for dataset loading, schema validation, and stratified sampling."""

import pandas as pd
import pytest

from src.data_loader import load_parquet, sample_stratified, validate_schema


def test_load_parquet_missing_file_raises() -> None:
    """Missing Parquet path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_parquet("data/raw/does_not_exist.parquet")


def test_validate_schema_empty_dataframe_raises() -> None:
    """Empty DataFrame should raise ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        validate_schema(empty_df, ["dur", "proto"])


def test_validate_schema_missing_columns_raises() -> None:
    """DataFrame missing required columns should raise ValueError."""
    df = pd.DataFrame({"dur": [0.1, 0.2], "proto": ["tcp", "udp"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df, ["dur", "proto", "sbytes"])


def test_sample_stratified_preserves_size_and_classes() -> None:
    """Stratified sampling should return exact max_rows and both classes."""
    df = pd.DataFrame(
        {
            "dur": list(range(100)),
            "label": [0] * 40 + [1] * 60,
        }
    )
    sampled = sample_stratified(df, target_col="label", max_rows=20, seed=42)
    assert len(sampled) == 20
    assert (sampled["label"] == 0).sum() == 8
    assert (sampled["label"] == 1).sum() == 12
