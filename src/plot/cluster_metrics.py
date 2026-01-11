"""
Visualization functions for multi-metric cluster validation results.

This module provides plotting functions to visualize silhouette score,
Calinski-Harabasz index, and Davies-Bouldin index across different
numbers of clusters, helping users understand and interpret cluster
optimization results.
"""

import logging
from typing import Optional

import dataframely as dy
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import polars as pl

from src.dataframes.geocode_cluster_metrics import (
    GeocodeClusterMetricsSchema,
    get_metric_interpretations,
)

logger = logging.getLogger(__name__)


def plot_all_metrics_vs_k(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    optimal_k: Optional[int] = None,
    figsize: tuple[float, float] = (14, 10),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a multi-panel plot showing all cluster validation metrics vs k.

    Args:
        metrics_df: DataFrame containing cluster metrics for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Sort by num_clusters for consistent plotting
    df = metrics_df.sort("num_clusters")
    k_values = df["num_clusters"].to_list()

    # Color scheme
    colors = {
        "silhouette": "#2E86AB",
        "calinski_harabasz": "#A23B72",
        "davies_bouldin": "#F18F01",
        "combined": "#06A77D",
        "optimal": "#E63946",
    }

    # Plot 1: Silhouette Score
    ax1 = axes[0, 0]
    sil_scores = df["silhouette_score"].to_list()
    ax1.plot(
        k_values,
        sil_scores,
        marker="o",
        linewidth=2,
        markersize=8,
        color=colors["silhouette"],
        label="Silhouette Score",
    )
    ax1.axhline(y=0.25, color="#E63946", linestyle="--", linewidth=1.5, alpha=0.7, label="Threshold (0.25)")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Silhouette Score", fontsize=11)
    ax1.set_title("Silhouette Score", fontsize=12, weight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    _highlight_optimal(ax1, optimal_k, k_values, sil_scores, colors["optimal"])

    # Plot 2: Calinski-Harabasz Index
    ax2 = axes[0, 1]
    ch_scores = df["calinski_harabasz_score"].to_list()
    ax2.plot(
        k_values,
        ch_scores,
        marker="s",
        linewidth=2,
        markersize=8,
        color=colors["calinski_harabasz"],
        label="Calinski-Harabasz",
    )
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Calinski-Harabasz Index", fontsize=11)
    ax2.set_title("Calinski-Harabasz Index (higher = better)", fontsize=12, weight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    _highlight_optimal(ax2, optimal_k, k_values, ch_scores, colors["optimal"])

    # Plot 3: Davies-Bouldin Index
    ax3 = axes[1, 0]
    db_scores = df["davies_bouldin_score"].to_list()
    ax3.plot(
        k_values,
        db_scores,
        marker="^",
        linewidth=2,
        markersize=8,
        color=colors["davies_bouldin"],
        label="Davies-Bouldin",
    )
    ax3.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax3.set_ylabel("Davies-Bouldin Index", fontsize=11)
    ax3.set_title("Davies-Bouldin Index (lower = better)", fontsize=12, weight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9)
    _highlight_optimal(ax3, optimal_k, k_values, db_scores, colors["optimal"])

    # Plot 4: Combined Score
    ax4 = axes[1, 1]
    combined_scores = df["combined_score"].to_list()
    ax4.plot(
        k_values,
        combined_scores,
        marker="D",
        linewidth=2,
        markersize=8,
        color=colors["combined"],
        label="Combined Score",
    )
    ax4.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax4.set_ylabel("Combined Score", fontsize=11)
    ax4.set_title("Combined Weighted Score (higher = better)", fontsize=12, weight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best", fontsize=9)
    _highlight_optimal(ax4, optimal_k, k_values, combined_scores, colors["optimal"])

    plt.suptitle(
        "Multi-Metric Cluster Validation",
        fontsize=16,
        weight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving multi-metric plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _highlight_optimal(
    ax: Axes,
    optimal_k: Optional[int],
    k_values: list[int],
    scores: list[float],
    color: str,
) -> None:
    """Helper to highlight optimal k on a plot axis."""
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        optimal_score = scores[idx]
        ax.axvline(x=optimal_k, color=color, linestyle=":", linewidth=2, alpha=0.7)
        ax.scatter(
            [optimal_k],
            [optimal_score],
            color=color,
            s=200,
            zorder=5,
            edgecolors="white",
            linewidths=2,
        )


def plot_normalized_metrics(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    optimal_k: Optional[int] = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a plot showing normalized metrics on the same scale [0, 1].

    This allows direct comparison of all metrics since they're normalized.

    Args:
        metrics_df: DataFrame containing cluster metrics for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    df = metrics_df.sort("num_clusters")
    k_values = df["num_clusters"].to_list()

    # Plot normalized metrics
    metrics_config = [
        ("silhouette_normalized", "Silhouette", "#2E86AB", "o"),
        ("calinski_harabasz_normalized", "Calinski-Harabasz", "#A23B72", "s"),
        ("davies_bouldin_normalized", "Davies-Bouldin (inv)", "#F18F01", "^"),
        ("combined_score", "Combined Score", "#06A77D", "D"),
    ]

    for col, label, color, marker in metrics_config:
        scores = df[col].to_list()
        linewidth = 3 if col == "combined_score" else 1.5
        alpha = 1.0 if col == "combined_score" else 0.7
        ax.plot(
            k_values,
            scores,
            marker=marker,
            linewidth=linewidth,
            markersize=7,
            color=color,
            label=label,
            alpha=alpha,
        )

    # Highlight optimal k
    if optimal_k is not None and optimal_k in k_values:
        ax.axvline(
            x=optimal_k,
            color="#E63946",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Optimal k={optimal_k}",
        )

    ax.set_xlabel("Number of Clusters (k)", fontsize=12, weight="bold")
    ax.set_ylabel("Normalized Score [0, 1]", fontsize=12, weight="bold")
    ax.set_title(
        "Normalized Cluster Validation Metrics",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving normalized metrics plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metric_rankings(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    optimal_k: Optional[int] = None,
    figsize: tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a heatmap showing how each k ranks across different metrics.

    This visualization helps identify k values that perform consistently
    well across all metrics vs those that excel in only one area.

    Args:
        metrics_df: DataFrame containing cluster metrics for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object
    """
    df = metrics_df.sort("num_clusters")

    # Compute ranks for each metric (1 = best)
    df_with_ranks = df.with_columns(
        pl.col("silhouette_score").rank(descending=True).alias("sil_rank"),
        pl.col("calinski_harabasz_score").rank(descending=True).alias("ch_rank"),
        pl.col("davies_bouldin_score").rank(descending=False).alias("db_rank"),  # Lower is better
        pl.col("combined_score").rank(descending=True).alias("combined_rank"),
    )

    k_values = df_with_ranks["num_clusters"].to_list()
    rank_matrix = np.array(
        [
            df_with_ranks["sil_rank"].to_list(),
            df_with_ranks["ch_rank"].to_list(),
            df_with_ranks["db_rank"].to_list(),
            df_with_ranks["combined_rank"].to_list(),
        ]
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap (lower rank = better = darker color)
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "Combined"])

    # Add rank values as text
    for i in range(4):
        for j in range(len(k_values)):
            rank = int(rank_matrix[i, j])
            text_color = "white" if rank <= len(k_values) / 2 else "black"
            ax.text(j, i, str(rank), ha="center", va="center", color=text_color, fontsize=10)

    # Highlight optimal k column
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax.axvline(x=idx - 0.5, color="#E63946", linewidth=3)
        ax.axvline(x=idx + 0.5, color="#E63946", linewidth=3)

    ax.set_xlabel("Number of Clusters (k)", fontsize=12, weight="bold")
    ax.set_title(
        "Metric Rankings by k (1 = Best)",
        fontsize=14,
        weight="bold",
        pad=15,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rank (lower = better)", fontsize=10)

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving metric rankings plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metrics_summary(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    optimal_k: Optional[int] = None,
    figsize: tuple[float, float] = (16, 10),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a comprehensive summary visualization with all metric plots.

    Combines the all-metrics plot with normalized metrics for a complete overview.

    Args:
        metrics_df: DataFrame containing cluster metrics for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    df = metrics_df.sort("num_clusters")
    k_values = df["num_clusters"].to_list()

    colors = {
        "silhouette": "#2E86AB",
        "calinski_harabasz": "#A23B72",
        "davies_bouldin": "#F18F01",
        "combined": "#06A77D",
        "optimal": "#E63946",
    }

    # Top row: Individual metrics
    # Silhouette
    ax1 = fig.add_subplot(gs[0, 0])
    sil_scores = df["silhouette_score"].to_list()
    ax1.plot(k_values, sil_scores, marker="o", linewidth=2, markersize=6, color=colors["silhouette"])
    ax1.axhline(y=0.25, color=colors["optimal"], linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("k", fontsize=10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_title("Silhouette", fontsize=11, weight="bold")
    ax1.grid(True, alpha=0.3)
    _highlight_optimal(ax1, optimal_k, k_values, sil_scores, colors["optimal"])

    # Calinski-Harabasz
    ax2 = fig.add_subplot(gs[0, 1])
    ch_scores = df["calinski_harabasz_score"].to_list()
    ax2.plot(k_values, ch_scores, marker="s", linewidth=2, markersize=6, color=colors["calinski_harabasz"])
    ax2.set_xlabel("k", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Calinski-Harabasz (↑)", fontsize=11, weight="bold")
    ax2.grid(True, alpha=0.3)
    _highlight_optimal(ax2, optimal_k, k_values, ch_scores, colors["optimal"])

    # Davies-Bouldin
    ax3 = fig.add_subplot(gs[0, 2])
    db_scores = df["davies_bouldin_score"].to_list()
    ax3.plot(k_values, db_scores, marker="^", linewidth=2, markersize=6, color=colors["davies_bouldin"])
    ax3.set_xlabel("k", fontsize=10)
    ax3.set_ylabel("Score", fontsize=10)
    ax3.set_title("Davies-Bouldin (↓)", fontsize=11, weight="bold")
    ax3.grid(True, alpha=0.3)
    _highlight_optimal(ax3, optimal_k, k_values, db_scores, colors["optimal"])

    # Bottom left: Combined score (larger)
    ax4 = fig.add_subplot(gs[1, :2])
    combined_scores = df["combined_score"].to_list()
    ax4.plot(k_values, combined_scores, marker="D", linewidth=3, markersize=10, color=colors["combined"])
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        optimal_score = combined_scores[idx]
        ax4.axvline(x=optimal_k, color=colors["optimal"], linestyle="--", linewidth=2, alpha=0.8)
        ax4.scatter([optimal_k], [optimal_score], color=colors["optimal"], s=250, zorder=5, edgecolors="white", linewidths=2)
        ax4.annotate(
            f"k={optimal_k}\nscore={optimal_score:.3f}",
            xy=(optimal_k, optimal_score),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors["optimal"], alpha=0.9),
            color="white",
            weight="bold",
        )
    ax4.set_xlabel("Number of Clusters (k)", fontsize=11, weight="bold")
    ax4.set_ylabel("Combined Score", fontsize=11, weight="bold")
    ax4.set_title("Combined Weighted Score", fontsize=12, weight="bold")
    ax4.grid(True, alpha=0.3)

    # Bottom right: Interpretation guide
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    interpretations = get_metric_interpretations()
    guide_text = "Metric Interpretation Guide\n" + "=" * 30 + "\n\n"
    guide_text += f"Selected: k={optimal_k}\n\n" if optimal_k else ""
    guide_text += "Silhouette: [-1, 1]\n  Higher = better separation\n  >0.25 = acceptable\n\n"
    guide_text += "Calinski-Harabasz: [0, ∞)\n  Higher = better defined\n\n"
    guide_text += "Davies-Bouldin: [0, ∞)\n  Lower = better clustering\n  <1 = good separation\n\n"
    guide_text += "Combined: [0, 1]\n  Weighted average\n  Higher = better overall"

    ax5.text(
        0.1, 0.95, guide_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.suptitle(
        "Multi-Metric Cluster Optimization Summary",
        fontsize=16,
        weight="bold",
        y=1.01,
    )

    if save_path:
        logger.info(f"Saving metrics summary plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
