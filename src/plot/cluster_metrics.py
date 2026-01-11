"""
Visualization functions for multi-metric cluster validation results.

This module provides plotting functions to visualize silhouette score,
Calinski-Harabasz index, Davies-Bouldin index, and inertia (for elbow method)
across different numbers of clusters, helping users understand and interpret
cluster optimization results.
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
    ElbowAnalysisResult,
    GeocodeClusterMetricsSchema,
    get_metric_interpretations,
    get_elbow_analysis,
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


def plot_elbow_curve(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    optimal_k: Optional[int] = None,
    show_distances: bool = True,
    figsize: tuple[float, float] = (12, 8),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create an elbow curve plot showing inertia vs number of clusters.

    The elbow method plots the within-cluster sum of squares (inertia) against
    the number of clusters. The optimal k is at the "elbow" point where the
    rate of decrease sharply changes.

    Args:
        metrics_df: DataFrame containing cluster metrics with inertia values
        optimal_k: The optimal number of clusters to highlight (optional).
                   If None, will attempt to detect the elbow point automatically.
        show_distances: Whether to show the perpendicular distances used for
                        elbow detection (default: True)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object
    """
    elbow_data: ElbowAnalysisResult = get_elbow_analysis(metrics_df)
    k_values: list[int] = elbow_data["k_values"]
    inertia_values: list[float] = elbow_data["inertia_values"]
    distances: list[float] = elbow_data["distances"]

    # Use detected elbow if optimal_k not provided
    if optimal_k is None:
        elbow_k_val = elbow_data["elbow_k"]
        optimal_k = elbow_k_val if elbow_k_val != 0 else None

    # Color scheme
    colors = {
        "inertia": "#2E86AB",
        "elbow": "#E63946",
        "distance": "#06A77D",
        "line": "#AAAAAA",
    }

    if show_distances:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.7))
        ax2 = None

    # Main elbow curve
    ax1.plot(
        k_values,
        inertia_values,
        marker="o",
        linewidth=2.5,
        markersize=10,
        color=colors["inertia"],
        label="Inertia (WCSS)",
    )

    # Draw reference line from first to last point
    ax1.plot(
        [k_values[0], k_values[-1]],
        [inertia_values[0], inertia_values[-1]],
        linestyle="--",
        linewidth=1.5,
        color=colors["line"],
        alpha=0.7,
        label="Reference line",
    )

    # Highlight elbow point
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        elbow_inertia = inertia_values[idx]

        ax1.axvline(
            x=optimal_k,
            color=colors["elbow"],
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )
        ax1.scatter(
            [optimal_k],
            [elbow_inertia],
            color=colors["elbow"],
            s=300,
            zorder=5,
            edgecolors="white",
            linewidths=3,
            label=f"Elbow (k={optimal_k})",
        )

        # Add annotation
        ax1.annotate(
            f"Elbow: k={optimal_k}\nInertia={elbow_inertia:.1f}",
            xy=(optimal_k, elbow_inertia),
            xytext=(20, 20),
            textcoords="offset points",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors["elbow"], alpha=0.9),
            color="white",
            weight="bold",
            arrowprops=dict(arrowstyle="->", color=colors["elbow"], lw=2),
        )

    ax1.set_xlabel("Number of Clusters (k)", fontsize=12, weight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12, weight="bold")
    ax1.set_title("Elbow Method for Optimal k Selection", fontsize=14, weight="bold", pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Plot distances (if requested)
    if ax2 is not None:
        ax2.bar(
            k_values,
            distances,
            color=colors["distance"],
            alpha=0.7,
            edgecolor="white",
            linewidth=1,
        )

        if optimal_k is not None and optimal_k in k_values:
            idx = k_values.index(optimal_k)
            ax2.bar(
                [optimal_k],
                [distances[idx]],
                color=colors["elbow"],
                alpha=0.9,
                edgecolor="white",
                linewidth=2,
            )

        ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax2.set_ylabel("Distance from Line", fontsize=11)
        ax2.set_title("Perpendicular Distance (larger = more likely elbow)", fontsize=11, weight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving elbow curve plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_elbow_derivatives(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    optimal_k: Optional[int] = None,
    figsize: tuple[float, float] = (14, 5),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Plot the first and second derivatives of the inertia curve.

    This provides additional insight into the rate of change of inertia,
    which can help identify the elbow point more precisely.

    Args:
        metrics_df: DataFrame containing cluster metrics with inertia values
        optimal_k: The optimal number of clusters to highlight (optional)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object
    """
    elbow_data: ElbowAnalysisResult = get_elbow_analysis(metrics_df)
    k_values: list[int] = elbow_data["k_values"]
    inertia_values: list[float] = elbow_data["inertia_values"]
    inertia_deltas: list[float] = elbow_data["inertia_deltas"]
    inertia_delta2: list[float] = elbow_data["inertia_delta2"]

    if optimal_k is None:
        elbow_k_val = elbow_data["elbow_k"]
        optimal_k = elbow_k_val if elbow_k_val != 0 else None

    colors = {
        "inertia": "#2E86AB",
        "delta1": "#A23B72",
        "delta2": "#F18F01",
        "optimal": "#E63946",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Inertia
    ax1.plot(k_values, inertia_values, marker="o", linewidth=2, markersize=6, color=colors["inertia"])
    ax1.set_xlabel("k", fontsize=10)
    ax1.set_ylabel("Inertia", fontsize=10)
    ax1.set_title("Inertia (WCSS)", fontsize=11, weight="bold")
    ax1.grid(True, alpha=0.3)
    _highlight_optimal(ax1, optimal_k, k_values, inertia_values, colors["optimal"])

    # Plot 2: First derivative (rate of change)
    ax2.plot(k_values, inertia_deltas, marker="s", linewidth=2, markersize=6, color=colors["delta1"])
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax2.set_xlabel("k", fontsize=10)
    ax2.set_ylabel("Δ Inertia", fontsize=10)
    ax2.set_title("1st Derivative (Rate of Change)", fontsize=11, weight="bold")
    ax2.grid(True, alpha=0.3)
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax2.axvline(x=optimal_k, color=colors["optimal"], linestyle=":", linewidth=2, alpha=0.7)

    # Plot 3: Second derivative (acceleration)
    ax3.plot(k_values, inertia_delta2, marker="^", linewidth=2, markersize=6, color=colors["delta2"])
    ax3.axhline(y=0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax3.set_xlabel("k", fontsize=10)
    ax3.set_ylabel("Δ² Inertia", fontsize=10)
    ax3.set_title("2nd Derivative (Acceleration)", fontsize=11, weight="bold")
    ax3.grid(True, alpha=0.3)
    if optimal_k is not None and optimal_k in k_values:
        ax3.axvline(x=optimal_k, color=colors["optimal"], linestyle=":", linewidth=2, alpha=0.7)

    plt.suptitle(
        "Elbow Method: Inertia and Derivatives",
        fontsize=14,
        weight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving elbow derivatives plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_normalized_metrics(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    optimal_k: Optional[int] = None,
    include_inertia: bool = False,
    figsize: tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a plot showing normalized metrics on the same scale [0, 1].

    This allows direct comparison of all metrics since they're normalized.

    Args:
        metrics_df: DataFrame containing cluster metrics for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        include_inertia: Whether to include normalized inertia in the plot
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

    if include_inertia:
        metrics_config.insert(3, ("inertia_normalized", "Inertia (inv)", "#8B5CF6", "v"))

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
        pl.col("inertia").rank(descending=False).alias("inertia_rank"),  # Lower is better
        pl.col("combined_score").rank(descending=True).alias("combined_rank"),
    )

    k_values = df_with_ranks["num_clusters"].to_list()
    rank_matrix = np.array(
        [
            df_with_ranks["sil_rank"].to_list(),
            df_with_ranks["ch_rank"].to_list(),
            df_with_ranks["db_rank"].to_list(),
            df_with_ranks["inertia_rank"].to_list(),
            df_with_ranks["combined_rank"].to_list(),
        ]
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap (lower rank = better = darker color)
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto")

    # Set ticks and labels
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(range(5))
    ax.set_yticklabels(["Silhouette", "Calinski-Harabasz", "Davies-Bouldin", "Inertia", "Combined"])

    # Add rank values as text
    for i in range(5):
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
    selection_method: str = "combined",
    figsize: tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a comprehensive summary visualization with all metric plots.

    Combines the all-metrics plot with normalized metrics and elbow curve.

    Args:
        metrics_df: DataFrame containing cluster metrics for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        selection_method: The method used for selection (for display purposes)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    df = metrics_df.sort("num_clusters")
    k_values = df["num_clusters"].to_list()

    colors = {
        "silhouette": "#2E86AB",
        "calinski_harabasz": "#A23B72",
        "davies_bouldin": "#F18F01",
        "inertia": "#8B5CF6",
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

    # Middle left: Elbow curve (larger)
    ax4 = fig.add_subplot(gs[1, :2])
    inertia_values = df["inertia"].to_list()
    ax4.plot(k_values, inertia_values, marker="v", linewidth=2.5, markersize=8, color=colors["inertia"])

    # Draw reference line for elbow detection
    ax4.plot(
        [k_values[0], k_values[-1]],
        [inertia_values[0], inertia_values[-1]],
        linestyle="--",
        linewidth=1.5,
        color="#AAAAAA",
        alpha=0.7,
    )

    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        optimal_inertia = inertia_values[idx]
        ax4.axvline(x=optimal_k, color=colors["optimal"], linestyle="--", linewidth=2, alpha=0.8)
        ax4.scatter([optimal_k], [optimal_inertia], color=colors["optimal"], s=200, zorder=5, edgecolors="white", linewidths=2)
    ax4.set_xlabel("Number of Clusters (k)", fontsize=11, weight="bold")
    ax4.set_ylabel("Inertia (WCSS)", fontsize=11, weight="bold")
    ax4.set_title("Elbow Curve", fontsize=12, weight="bold")
    ax4.grid(True, alpha=0.3)

    # Middle right: Method info
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    method_descriptions = {
        "combined": "Uses weighted average of\nnormalized metrics",
        "silhouette": "Uses silhouette score only",
        "elbow": "Finds point of maximum\ncurvature in inertia curve",
    }

    info_text = f"Selection Method: {selection_method}\n"
    info_text += "─" * 25 + "\n\n"
    info_text += method_descriptions.get(selection_method, "Unknown method") + "\n\n"
    if optimal_k is not None:
        info_text += f"Selected k = {optimal_k}\n\n"
        if optimal_k in k_values:
            idx = k_values.index(optimal_k)
            info_text += f"Silhouette: {sil_scores[idx]:.3f}\n"
            info_text += f"Calinski-H: {ch_scores[idx]:.1f}\n"
            info_text += f"Davies-B: {db_scores[idx]:.3f}\n"
            info_text += f"Inertia: {inertia_values[idx]:.1f}\n"

    ax5.text(
        0.1, 0.95, info_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Bottom row: Combined score (larger)
    ax6 = fig.add_subplot(gs[2, :2])
    combined_scores = df["combined_score"].to_list()
    ax6.plot(k_values, combined_scores, marker="D", linewidth=3, markersize=10, color=colors["combined"])
    if optimal_k is not None and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        optimal_score = combined_scores[idx]
        ax6.axvline(x=optimal_k, color=colors["optimal"], linestyle="--", linewidth=2, alpha=0.8)
        ax6.scatter([optimal_k], [optimal_score], color=colors["optimal"], s=250, zorder=5, edgecolors="white", linewidths=2)
        ax6.annotate(
            f"k={optimal_k}\nscore={optimal_score:.3f}",
            xy=(optimal_k, optimal_score),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=colors["optimal"], alpha=0.9),
            color="white",
            weight="bold",
        )
    ax6.set_xlabel("Number of Clusters (k)", fontsize=11, weight="bold")
    ax6.set_ylabel("Combined Score", fontsize=11, weight="bold")
    ax6.set_title("Combined Weighted Score", fontsize=12, weight="bold")
    ax6.grid(True, alpha=0.3)

    # Bottom right: Interpretation guide
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    guide_text = "Metric Interpretation\n" + "─" * 20 + "\n\n"
    guide_text += "Silhouette: [-1, 1]\n  >0.25 = acceptable\n\n"
    guide_text += "Calinski-H: [0, ∞)\n  Higher = better\n\n"
    guide_text += "Davies-B: [0, ∞)\n  Lower = better\n\n"
    guide_text += "Inertia: [0, ∞)\n  Look for elbow\n\n"
    guide_text += "Combined: [0, 1]\n  Higher = better"

    ax7.text(
        0.1, 0.95, guide_text,
        transform=ax7.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
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
