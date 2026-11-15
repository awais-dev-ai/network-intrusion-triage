"""Exploratory data analysis summary and Matplotlib visualizations."""

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import resolve_path

matplotlib.use("Agg")


def generate_eda_figures(
    train_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate and save four EDA figures and return summary statistics."""
    figures_dir = resolve_path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    target_col = config["schema"]["target_col"]
    saved_files: list[str] = []

    saved_files.append(_plot_class_distribution(train_df, target_col, figures_dir))
    saved_files.append(_plot_service_attack_rate(train_df, target_col, figures_dir))
    saved_files.append(_plot_numeric_distributions(train_df, target_col, figures_dir))
    saved_files.append(_plot_correlation_heatmap(train_df, config, figures_dir))

    return {
        "train_rows": int(len(train_df)),
        "normal_count": int((train_df[target_col] == 0).sum()),
        "suspicious_count": int((train_df[target_col] == 1).sum()),
        "figures": saved_files,
    }


def _plot_class_distribution(
    df: pd.DataFrame, target_col: str, out_dir: Path
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts = df[target_col].value_counts().sort_index()
    labels = ["Normal (0)", "Suspicious (1)"]
    colors = ["#2b7bba", "#d95f02"]
    axes[0].bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors)
    axes[0].set_title("Binary Target Distribution")
    axes[0].set_ylabel("Flow Count")

    if "attack_cat" in df.columns:
        cat_counts = df["attack_cat"].astype(str).value_counts()
        axes[1].barh(cat_counts.index[::-1], cat_counts.values[::-1], color="#4c72b0")
        axes[1].set_title("Attack Category Breakdown")
        axes[1].set_xlabel("Flow Count")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    out_path = out_dir / "eda_class_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def _plot_service_attack_rate(
    df: pd.DataFrame, target_col: str, out_dir: Path
) -> str:
    service_stats = (
        df.assign(service=df["service"].astype(str))
        .groupby("service", observed=False)[target_col]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .head(10)
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(service_stats.index, service_stats["mean"] * 100.0, color="#c44e52")
    ax.set_xlabel("Suspicious Flow Rate (%)")
    ax.set_title("Suspicious Traffic Rate Across Top 10 Services")
    ax.set_xlim(0, 105)
    fig.tight_layout()

    out_path = out_dir / "eda_service_attack_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def _plot_numeric_distributions(
    df: pd.DataFrame, target_col: str, out_dir: Path
) -> str:
    features = ["dur", "sbytes", "dbytes", "rate"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    normal_mask = df[target_col] == 0
    attack_mask = df[target_col] == 1

    for ax, feat in zip(axes.ravel(), features):
        normal_vals = np.log1p(df.loc[normal_mask, feat].clip(lower=0))
        attack_vals = np.log1p(df.loc[attack_mask, feat].clip(lower=0))
        ax.hist(normal_vals, bins=35, alpha=0.6, label="Normal (0)", color="#2b7bba")
        ax.hist(attack_vals, bins=35, alpha=0.6, label="Suspicious (1)", color="#d95f02")
        ax.set_title(f"log1p({feat}) Distribution by Class")
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.tight_layout()
    out_path = out_dir / "eda_numeric_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def _plot_correlation_heatmap(
    df: pd.DataFrame, config: dict[str, Any], out_dir: Path
) -> str:
    num_cols = config["schema"]["numeric_cols"]
    target_col = config["schema"]["target_col"]

    corr_with_target = (
        df[num_cols + [target_col]]
        .corr(numeric_only=True)[target_col]
        .drop(target_col)
        .abs()
        .sort_values(ascending=False)
    )
    top_cols = list(corr_with_target.head(10).index) + [target_col]
    corr_matrix = df[top_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_cols)))
    ax.set_yticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(top_cols, fontsize=8.5)
    ax.set_title("Correlation Heatmap (Top 10 Features with Target)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = out_dir / "eda_correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)
