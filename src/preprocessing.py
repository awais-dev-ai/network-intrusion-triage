"""Leakage-free feature preparation and scikit-learn preprocessing pipeline."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import validate_schema


def prepare_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Extract clean feature matrix X and optional target series y.

    Drops leakage/metadata columns and coerces feature dtypes consistently.
    """
    schema = config["schema"]
    cat_cols: list[str] = schema["categorical_cols"]
    num_cols: list[str] = schema["numeric_cols"]
    target_col: str = schema["target_col"]

    required = cat_cols + num_cols + ([target_col] if require_target else [])
    validate_schema(df, required)

    feature_cols = cat_cols + num_cols
    X = df[feature_cols].copy()

    for col in cat_cols:
        X[col] = X[col].astype("string").fillna("unknown").astype(str)

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = df[target_col].astype(int) if require_target else None
    return X, y


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    """Build a leakage-free ColumnTransformer for numeric and categorical features."""
    schema = config["schema"]
    num_cols: list[str] = schema["numeric_cols"]
    cat_cols: list[str] = schema["categorical_cols"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
