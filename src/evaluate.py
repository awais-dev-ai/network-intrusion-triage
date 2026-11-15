"""Model evaluation metrics, JSON reporting, and comparison plots."""

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.config import resolve_path

matplotlib.use("Agg")


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate a fitted pipeline on test data and return metrics, predictions, and probabilities."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, y_pred, y_prob


def save_evaluation_artifacts(
    results: dict[str, dict[str, Any]],
    best_model_name: str,
    y_test: pd.Series,
    probas_by_model: dict[str, np.ndarray],
    config: dict[str, Any],
) -> None:
    """Write metrics.json and generate comparison, confusion matrix, and ROC plots."""
    metrics_path = resolve_path(config["paths"]["metrics_file"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "best_model": best_model_name,
        "priority_metric": config["evaluation"]["priority_metric"],
        "sampling": config["sampling"],
        "models": results,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    figures_dir = resolve_path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_model_comparison(results, figures_dir)
    _plot_confusion_matrix(
        np.array(results[best_model_name]["confusion_matrix"]),
        best_model_name,
        figures_dir,
    )
    _plot_roc_curves(y_test, probas_by_model, results, figures_dir)


def _plot_model_comparison(
    results: dict[str, dict[str, Any]], out_dir: Path
) -> None:
    model_names = list(results.keys())
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, metric in enumerate(metric_keys):
        values = [results[name][metric] for name in model_names]
        ax.bar(x + idx * width, values, width, label=metric.replace("_", " ").title())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([name.replace("_", " ").title() for name in model_names])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Experiment Comparison on Unseen Test Split")
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(
    cm: np.ndarray, model_name: str, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Suspicious (1)"])
    ax.set_yticklabels(["Normal (0)", "Suspicious (1)"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ').title()}")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_best.png", dpi=150)
    plt.close(fig)


def _plot_roc_curves(
    y_test: pd.Series,
    probas_by_model: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in probas_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = results[name]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves Across Experiments")
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
