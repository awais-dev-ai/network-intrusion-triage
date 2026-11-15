"""Dataset loading, schema validation, and deterministic stratified sampling."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import resolve_path


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a Parquet dataset and verify it is non-empty.

    Args:
        path: Path to the Parquet file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read or contains zero rows.
    """
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_parquet(file_path)
    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {file_path}")
    return df


def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Ensure all required columns are present in the DataFrame.

    Args:
        df: Input DataFrame.
        required_cols: Column names that must exist.

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sample_stratified(
    df: pd.DataFrame,
    target_col: str,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    """Return a deterministic stratified subset when max_rows is smaller than len(df)."""
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    sampled_df, _ = train_test_split(
        df,
        train_size=max_rows,
        stratify=df[target_col],
        random_state=seed,
    )
    return sampled_df.reset_index(drop=True)


def load_train_test(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and optionally sample the raw train and test splits."""
    schema = config["schema"]
    required_cols = (
        schema["categorical_cols"] + schema["numeric_cols"] + [schema["target_col"]]
    )

    train_df = load_parquet(config["paths"]["raw_train"])
    test_df = load_parquet(config["paths"]["raw_test"])

    validate_schema(train_df, required_cols)
    validate_schema(test_df, required_cols)

    seed = int(config["project"]["seed"])
    sampling = config["sampling"]
    train_df = sample_stratified(
        train_df, schema["target_col"], sampling.get("max_train_rows"), seed
    )
    test_df = sample_stratified(
        test_df, schema["target_col"], sampling.get("max_test_rows"), seed
    )
    return train_df, test_df
