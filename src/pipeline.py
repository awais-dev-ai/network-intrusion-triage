"""Authoritative end-to-end training, evaluation, and artifact-saving pipeline."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import load_config, resolve_path
from src.data_loader import load_train_test
from src.eda import generate_eda_figures
from src.evaluate import evaluate_model, save_evaluation_artifacts
from src.models import build_models
from src.preprocessing import prepare_features


def run_pipeline(config_path: Path | str | None = None) -> dict[str, Any]:
    """Execute EDA, model training, evaluation, and artifact persistence."""
    config = load_config(config_path)
    train_df, test_df = load_train_test(config)

    eda_summary = generate_eda_figures(train_df, config)

    X_train, y_train = prepare_features(train_df, config, require_target=True)
    X_test, y_test = prepare_features(test_df, config, require_target=True)

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_out = X_train.copy()
    train_out[config["schema"]["target_col"]] = y_train
    test_out = X_test.copy()
    test_out[config["schema"]["target_col"]] = y_test
    train_out.to_parquet(processed_dir / "train_prepared.parquet", index=False)
    test_out.to_parquet(processed_dir / "test_prepared.parquet", index=False)

    pipelines = build_models(config)
    results: dict[str, dict[str, Any]] = {}
    probas_by_model: dict[str, np.ndarray] = {}
    fitted_models = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics, _, y_prob = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        probas_by_model[name] = y_prob
        fitted_models[name] = pipeline

    priority_metric = config["evaluation"]["priority_metric"]
    best_name = max(
        results.keys(),
        key=lambda k: (results[k][priority_metric], results[k]["f1_score"]),
    )

    best_model_path = resolve_path(config["paths"]["best_model"])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], best_model_path)

    save_evaluation_artifacts(results, best_name, y_test, probas_by_model, config)

    return {
        "eda": eda_summary,
        "best_model": best_name,
        "results": results,
        "best_model_path": str(best_model_path),
    }


def main() -> None:
    """CLI entrypoint for running the full intrusion triage pipeline."""
    summary = run_pipeline()
    print(f"Best model selected: {summary['best_model']}")
    print(f"Saved model to: {summary['best_model_path']}")
    for name, metrics in summary["results"].items():
        print(
            f"  {name:24s} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
