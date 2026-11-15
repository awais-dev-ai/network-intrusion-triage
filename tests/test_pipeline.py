"""Tests for preprocessing, model pipelines, and edge-case handling."""

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import build_models
from src.preprocessing import prepare_features


def _make_sample_df(config: dict, n_rows: int = 20) -> pd.DataFrame:
    """Build a small synthetic DataFrame matching the configured schema."""
    rng = np.random.default_rng(42)
    data = {}
    for col in config["schema"]["categorical_cols"]:
        data[col] = rng.choice(["tcp", "udp", "http"], size=n_rows)
    for col in config["schema"]["numeric_cols"]:
        data[col] = rng.uniform(0.0, 100.0, size=n_rows)
    data["attack_cat"] = ["Normal" if i % 2 == 0 else "Exploits" for i in range(n_rows)]
    data["label"] = [0 if i % 2 == 0 else 1 for i in range(n_rows)]
    return pd.DataFrame(data)


def test_prepare_features_drops_leakage_and_supports_inference() -> None:
    """prepare_features should drop attack_cat and allow require_target=False."""
    config = load_config()
    df = _make_sample_df(config, n_rows=10)
    X_train, y_train = prepare_features(df, config, require_target=True)

    assert "attack_cat" not in X_train.columns
    assert "label" not in X_train.columns
    assert y_train is not None and len(y_train) == 10

    df_infer = df.drop(columns=["label", "attack_cat"])
    X_infer, y_infer = prepare_features(df_infer, config, require_target=False)
    assert y_infer is None
    assert list(X_infer.columns) == list(X_train.columns)


def test_pipeline_handles_nan_corrupt_and_unseen_categories() -> None:
    """Fitted pipeline should handle NaN/corrupt numbers and unseen categories."""
    config = load_config()
    train_df = _make_sample_df(config, n_rows=30)
    X_train, y_train = prepare_features(train_df, config, require_target=True)

    lr_pipeline = build_models(config)["logistic_regression"]
    lr_pipeline.fit(X_train, y_train)

    edge_df = _make_sample_df(config, n_rows=4).drop(columns=["label", "attack_cat"])
    edge_df.loc[0, "proto"] = "unseen_protocol"
    edge_df.loc[1, "service"] = "unseen_service"
    edge_df.loc[2, "dur"] = np.nan
    edge_df["sbytes"] = edge_df["sbytes"].astype(object)
    edge_df.loc[3, "sbytes"] = "corrupt_text"

    X_edge, _ = prepare_features(edge_df, config, require_target=False)
    preds = lr_pipeline.predict(X_edge)
    probs = lr_pipeline.predict_proba(X_edge)[:, 1]

    assert len(preds) == 4
    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))
