import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.cluster_optimization import ClusterMetrics, OptimalKResult

logger = logging.getLogger(__name__)


def plot_optimization_metrics(
    result: OptimalKResult,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Create comprehensive visualization of cluster optimization metrics.

    Creates a multi-panel figure showing:
    1. Silhouette scores across k values
    2. Davies-Bouldin index across k values
    3. Calinski-Harabasz score across k values
    4. Inertia (elbow plot)
    5. Cluster size statistics
    6. Combined ranking visualization

    Args:
        result: OptimalKResult from ClusterOptimizer
        figsize: Figure size in inches (width, height)

    Returns:
        matplotlib.Figure with all subplots
    """
    metrics_list = result.all_metrics
    k_values = [m.k for m in metrics_list]
    optimal_k = result.optimal_k

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        f"Cluster Optimization Analysis\nOptimal k={optimal_k} ({result.selection_method})",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Silhouette Score (top left)
    ax = axes[0, 0]
    silhouette_scores = [m.silhouette for m in metrics_list]
    ax.plot(k_values, silhouette_scores, "o-", linewidth=2, markersize=8)
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.7, label="Optimal k")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score (Higher is Better)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add horizontal reference line at 0.5 (good quality threshold)
    ax.axhline(0.5, color="green", linestyle=":", alpha=0.5, label="Good quality")
    ax.axhline(0.3, color="orange", linestyle=":", alpha=0.5, label="Fair quality")

    # 2. Davies-Bouldin Index (top center)
    ax = axes[0, 1]
    db_scores = [m.davies_bouldin for m in metrics_list]
    ax.plot(k_values, db_scores, "o-", linewidth=2, markersize=8, color="orange")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.7, label="Optimal k")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Davies-Bouldin Index", fontsize=11)
    ax.set_title(
        "Davies-Bouldin Index (Lower is Better)", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Calinski-Harabasz Score (top right)
    ax = axes[0, 2]
    ch_scores = [m.calinski_harabasz for m in metrics_list]
    ax.plot(k_values, ch_scores, "o-", linewidth=2, markersize=8, color="green")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.7, label="Optimal k")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Calinski-Harabasz Score", fontsize=11)
    ax.set_title(
        "Calinski-Harabasz Score (Higher is Better)", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Inertia / Elbow Plot (bottom left)
    ax = axes[1, 0]
    inertias = [m.inertia for m in metrics_list]
    ax.plot(k_values, inertias, "o-", linewidth=2, markersize=8, color="purple")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.7, label="Optimal k")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Inertia", fontsize=11)
    ax.set_title(
        "Inertia - Elbow Method (Lower is Better)", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add elbow detection visualization
    if len(k_values) > 2:
        # Calculate and show rate of change
        first_deriv = np.diff(inertias)
        # Normalize for visualization
        normalized_deriv = (first_deriv - np.min(first_deriv)) / (
            np.max(first_deriv) - np.min(first_deriv)
        )
        ax2 = ax.twinx()
        ax2.plot(
            k_values[1:],
            normalized_deriv,
            "s--",
            color="gray",
            alpha=0.5,
            label="Rate of change",
        )
        ax2.set_ylabel("Normalized Rate of Change", fontsize=10, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    # 5. Cluster Size Statistics (bottom center)
    ax = axes[1, 1]
    mean_sizes = [m.mean_cluster_size for m in metrics_list]
    min_sizes = [m.min_cluster_size for m in metrics_list]
    max_sizes = [m.max_cluster_size for m in metrics_list]

    ax.plot(k_values, mean_sizes, "o-", linewidth=2, markersize=6, label="Mean size")
    ax.plot(k_values, min_sizes, "s--", linewidth=1.5, markersize=5, label="Min size")
    ax.plot(k_values, max_sizes, "^--", linewidth=1.5, markersize=5, label="Max size")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.7, label="Optimal k")
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Cluster Size (# hexagons)", fontsize=11)
    ax.set_title("Cluster Size Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 6. Multi-Metric Summary (bottom right)
    ax = axes[1, 2]

    # Normalize all metrics to 0-1 scale for comparison
    def normalize(values):
        arr = np.array(values)
        return (
            (arr - arr.min()) / (arr.max() - arr.min())
            if arr.max() != arr.min()
            else arr
        )

    # Higher is better for silhouette and CH
    norm_silhouette = normalize(silhouette_scores)
    norm_ch = normalize(ch_scores)

    # Lower is better for DB and inertia, so invert
    norm_db = 1 - normalize(db_scores)
    norm_inertia = 1 - normalize(inertias)

    # Compute average normalized score
    avg_scores = (norm_silhouette + norm_ch + norm_db + norm_inertia) / 4

    ax.plot(k_values, avg_scores, "o-", linewidth=3, markersize=10, color="darkblue")
    ax.axvline(optimal_k, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Combined Score (normalized)", fontsize=11)
    ax.set_title("Multi-Metric Combined Score", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Highlight the optimal point
    optimal_idx = k_values.index(optimal_k)
    ax.scatter(
        [optimal_k],
        [avg_scores[optimal_idx]],
        s=300,
        color="red",
        marker="*",
        zorder=5,
        label=f"Optimal k={optimal_k}",
    )
    ax.legend()

    plt.tight_layout()
    return fig


def plot_silhouette_comparison(
    metrics_list: List[ClusterMetrics],
    optimal_k: int,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Create a focused comparison plot for silhouette scores.

    Shows both the silhouette scores and their interpretation with
    quality threshold bands.

    Args:
        metrics_list: List of ClusterMetrics
        optimal_k: The selected optimal k value
        figsize: Figure size in inches

    Returns:
        matplotlib.Figure
    """
    k_values = [m.k for m in metrics_list]
    silhouette_scores = [m.silhouette for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        f"Silhouette Score Analysis (Optimal k={optimal_k})",
        fontsize=14,
        fontweight="bold",
    )

    # Left plot: Line plot with quality bands
    ax1.fill_between(
        k_values, 0.7, 1.0, alpha=0.2, color="green", label="Excellent (>0.7)"
    )
    ax1.fill_between(
        k_values, 0.5, 0.7, alpha=0.2, color="lightgreen", label="Good (0.5-0.7)"
    )
    ax1.fill_between(
        k_values, 0.3, 0.5, alpha=0.2, color="yellow", label="Fair (0.3-0.5)"
    )
    ax1.fill_between(
        k_values, 0.0, 0.3, alpha=0.2, color="orange", label="Weak (0.0-0.3)"
    )

    ax1.plot(k_values, silhouette_scores, "o-", linewidth=2, markersize=8, color="blue")
    ax1.axvline(optimal_k, color="red", linestyle="--", linewidth=2, label="Optimal k")

    # Highlight optimal point
    optimal_idx = k_values.index(optimal_k)
    ax1.scatter(
        [optimal_k],
        [silhouette_scores[optimal_idx]],
        s=200,
        color="red",
        marker="*",
        zorder=5,
    )

    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Silhouette Score", fontsize=11)
    ax1.set_title("Silhouette Score with Quality Bands", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.set_ylim(-0.05, 1.05)

    # Right plot: Bar chart
    colors = ["red" if k == optimal_k else "steelblue" for k in k_values]
    bars = ax2.bar(
        k_values, silhouette_scores, color=colors, alpha=0.7, edgecolor="black"
    )

    # Add value labels on bars
    for bar, score in zip(bars, silhouette_scores):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Silhouette Score by k", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, max(silhouette_scores) * 1.15)

    plt.tight_layout()
    return fig


def plot_elbow_detail(
    metrics_list: List[ClusterMetrics],
    optimal_k: int,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Create detailed elbow plot with derivative analysis.

    Shows inertia curve and its first/second derivatives to visualize
    the elbow point.

    Args:
        metrics_list: List of ClusterMetrics
        optimal_k: The selected optimal k value
        figsize: Figure size in inches

    Returns:
        matplotlib.Figure
    """
    k_values = np.array([m.k for m in metrics_list])
    inertias = np.array([m.inertia for m in metrics_list])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"Elbow Method Analysis (Optimal k={optimal_k})", fontsize=14, fontweight="bold"
    )

    # Plot 1: Inertia
    ax1.plot(k_values, inertias, "o-", linewidth=2, markersize=8, color="purple")
    ax1.axvline(optimal_k, color="red", linestyle="--", linewidth=2, label="Optimal k")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Inertia Curve")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: First derivative (rate of change)
    if len(k_values) > 1:
        first_deriv = np.diff(inertias)
        ax2.plot(
            k_values[1:],
            -first_deriv,
            "s-",
            linewidth=2,
            markersize=7,
            color="orange",
        )
        ax2.axvline(
            optimal_k, color="red", linestyle="--", linewidth=2, label="Optimal k"
        )
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Rate of Improvement")
        ax2.set_title("First Derivative (Marginal Gain)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # Plot 3: Second derivative (acceleration)
    if len(k_values) > 2:
        first_deriv = np.diff(inertias)
        second_deriv = np.diff(first_deriv)
        ax3.plot(
            k_values[2:],
            second_deriv,
            "^-",
            linewidth=2,
            markersize=7,
            color="green",
        )
        ax3.axvline(
            optimal_k, color="red", linestyle="--", linewidth=2, label="Optimal k"
        )
        ax3.axhline(0, color="black", linestyle=":", alpha=0.5)
        ax3.set_xlabel("Number of Clusters (k)")
        ax3.set_ylabel("Acceleration")
        ax3.set_title("Second Derivative (Elbow Detection)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    plt.tight_layout()
    return fig
