"""Model factory returning leakage-free scikit-learn pipelines."""

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor


def build_models(config: dict[str, Any]) -> dict[str, Pipeline]:
    """Construct baseline and improved classification pipelines from config."""
    seed = int(config["project"]["seed"])
    model_cfg = config["models"]

    lr_cfg = model_cfg["logistic_regression"]
    rf_cfg = model_cfg["random_forest"]
    hgb_cfg = model_cfg["hist_gradient_boosting"]

    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=int(lr_cfg["max_iter"]),
            C=float(lr_cfg["C"]),
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=int(rf_cfg["n_estimators"]),
            max_depth=int(rf_cfg["max_depth"]),
            n_jobs=int(rf_cfg["n_jobs"]),
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=int(hgb_cfg["max_iter"]),
            learning_rate=float(hgb_cfg["learning_rate"]),
            max_depth=int(hgb_cfg["max_depth"]),
            class_weight=hgb_cfg["class_weight"],
            random_state=seed,
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(config)),
                ("classifier", estimator),
            ]
        )
        for name, estimator in estimators.items()
    }
