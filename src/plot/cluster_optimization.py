"""
Visualization functions for cluster optimization results.

This module provides plotting functions to visualize silhouette scores
across different numbers of clusters, helping users understand and
interpret cluster optimization results.
"""

import logging
from typing import Optional

import dataframely as dy
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from src.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreSchema

logger = logging.getLogger(__name__)


def plot_silhouette_vs_k(
    silhouette_df: dy.DataFrame[GeocodeSilhouetteScoreSchema],
    optimal_k: Optional[int] = None,
    min_threshold: float = 0.25,
    figsize: tuple[float, float] = (10, 6),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a line plot showing silhouette score vs number of clusters.

    Args:
        silhouette_df: DataFrame containing silhouette scores for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        min_threshold: Minimum acceptable silhouette score threshold (default: 0.25)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_silhouette_vs_k(
        ...     combined_scores,
        ...     optimal_k=8,
        ...     save_path="output/silhouette_vs_k.png"
        ... )
    """
    overall_scores = silhouette_df.filter(pl.col("geocode").is_null())

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the line
    ax.plot(
        overall_scores["num_clusters"],
        overall_scores["silhouette_score"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Silhouette Score",
    )

    # Add horizontal threshold line
    ax.axhline(
        y=min_threshold,
        color="#E63946",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Threshold ({min_threshold})",
    )

    # Highlight optimal k if provided
    if optimal_k is not None:
        optimal_score_df = overall_scores.filter(pl.col("num_clusters") == optimal_k)
        if len(optimal_score_df) > 0:
            optimal_score = optimal_score_df["silhouette_score"][0]
            ax.axvline(
                x=optimal_k,
                color="#06A77D",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                label=f"Optimal k={optimal_k}",
            )
            ax.scatter(
                [optimal_k],
                [optimal_score],
                color="#06A77D",
                s=200,
                zorder=5,
                edgecolors="white",
                linewidths=2,
            )
            # Add annotation
            ax.annotate(
                f"k={optimal_k}\nscore={optimal_score:.3f}",
                xy=(optimal_k, optimal_score),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#06A77D", alpha=0.8),
                color="white",
                weight="bold",
                ha="left",
            )

    # Styling
    ax.set_xlabel("Number of Clusters (k)", fontsize=12, weight="bold")
    ax.set_ylabel("Silhouette Score", fontsize=12, weight="bold")
    ax.set_title(
        "Cluster Quality vs Number of Clusters", fontsize=14, weight="bold", pad=20
    )
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Set y-axis limits to show context
    all_scores = overall_scores["silhouette_score"].to_list()
    y_min = min(all_scores + [min_threshold - 0.1, -0.2])
    y_max = max(all_scores + [min_threshold + 0.1, 0.8])
    ax.set_ylim(y_min, y_max)

    # Add interpretation guide as text box
    interpretation_text = (
        "Silhouette Score Interpretation:\n"
        "  0.7-1.0: Strong structure\n"
        "  0.5-0.7: Reasonable structure\n"
        "  0.25-0.5: Weak structure\n"
        "  <0.25: No substantial structure"
    )
    ax.text(
        0.02,
        0.98,
        interpretation_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving silhouette vs k plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_silhouette_distributions(
    silhouette_df: dy.DataFrame[GeocodeSilhouetteScoreSchema],
    top_n: int = 5,
    figsize: tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create box plots showing per-geocode silhouette score distributions.

    Shows the distribution of silhouette scores for individual geocodes
    for the top N cluster configurations, helping identify how consistently
    good the clustering is across all data points.

    Args:
        silhouette_df: DataFrame containing silhouette scores for multiple k values
        top_n: Number of top k values to display (default: 5)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_silhouette_distributions(
        ...     combined_scores,
        ...     top_n=3,
        ...     save_path="output/silhouette_distributions.png"
        ... )
    """
    overall_scores = silhouette_df.filter(pl.col("geocode").is_null())
    top_k_values = overall_scores["num_clusters"].head(top_n).to_list()

    # Filter to per-geocode scores for top k values
    per_geocode_scores = silhouette_df.filter(
        pl.col("geocode").is_not_null() & pl.col("num_clusters").is_in(top_k_values)
    )

    # Convert to pandas for seaborn
    plot_data = per_geocode_scores.select(
        ["num_clusters", "silhouette_score"]
    ).to_pandas()

    fig, ax = plt.subplots(figsize=figsize)

    # Create box plot
    sns.boxplot(
        data=plot_data,
        x="num_clusters",
        y="silhouette_score",
        hue="num_clusters",
        palette="Set2",
        ax=ax,
        width=0.6,
        legend=False,
    )

    # Overlay violin plot for density
    sns.violinplot(
        data=plot_data,
        x="num_clusters",
        y="silhouette_score",
        hue="num_clusters",
        palette="Set2",
        ax=ax,
        alpha=0.3,
        inner=None,
        legend=False,
    )

    # Add overall scores as scatter points
    for k in top_k_values:
        overall_score = overall_scores.filter(pl.col("num_clusters") == k)[
            "silhouette_score"
        ][0]
        k_index = top_k_values.index(k)
        ax.scatter(
            [k_index],
            [overall_score],
            color="red",
            s=150,
            zorder=10,
            marker="D",
            edgecolors="darkred",
            linewidths=2,
            label="Overall Score" if k_index == 0 else "",
        )

    # Add horizontal line at 0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel("Number of Clusters (k)", fontsize=12, weight="bold")
    ax.set_ylabel("Silhouette Score", fontsize=12, weight="bold")
    ax.set_title(
        f"Per-Geocode Silhouette Score Distributions (Top {top_n} Configurations)",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    # Add summary statistics as text
    summary_stats = []
    for k in top_k_values:
        k_scores = per_geocode_scores.filter(pl.col("num_clusters") == k)[
            "silhouette_score"
        ]
        mean_score = k_scores.mean()
        std_score = k_scores.std()
        summary_stats.append(f"k={k}: μ={mean_score:.3f}, σ={std_score:.3f}")

    summary_text = "Per-Geocode Statistics:\n" + "\n".join(summary_stats)
    ax.text(
        0.98,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        logger.info(f"Saving silhouette distributions plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_optimization_summary(
    silhouette_df: dy.DataFrame[GeocodeSilhouetteScoreSchema],
    optimal_k: Optional[int] = None,
    min_threshold: float = 0.25,
    top_n: int = 5,
    figsize: tuple[float, float] = (16, 6),
    save_path: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """
    Create a comprehensive summary plot with both silhouette vs k and distributions.

    Combines the line plot and box plot into a single figure for a complete
    overview of cluster optimization results.

    Args:
        silhouette_df: DataFrame containing silhouette scores for multiple k values
        optimal_k: The optimal number of clusters to highlight (optional)
        min_threshold: Minimum acceptable silhouette score threshold (default: 0.25)
        top_n: Number of top k values to display in distributions (default: 5)
        figsize: Figure size as (width, height) in inches
        save_path: Path to save the figure (optional)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_optimization_summary(
        ...     combined_scores,
        ...     optimal_k=8,
        ...     save_path="output/optimization_summary.png"
        ... )
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # Left plot: Silhouette vs k
    ax1 = fig.add_subplot(gs[0, 0])
    overall_scores = silhouette_df.filter(pl.col("geocode").is_null())

    ax1.plot(
        overall_scores["num_clusters"],
        overall_scores["silhouette_score"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Silhouette Score",
    )

    ax1.axhline(
        y=min_threshold,
        color="#E63946",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Threshold ({min_threshold})",
    )

    if optimal_k is not None:
        optimal_score_df = overall_scores.filter(pl.col("num_clusters") == optimal_k)
        if len(optimal_score_df) > 0:
            optimal_score = optimal_score_df["silhouette_score"][0]
            ax1.axvline(
                x=optimal_k,
                color="#06A77D",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                label=f"Optimal k={optimal_k}",
            )
            ax1.scatter(
                [optimal_k],
                [optimal_score],
                color="#06A77D",
                s=200,
                zorder=5,
                edgecolors="white",
                linewidths=2,
            )

    ax1.set_xlabel("Number of Clusters (k)", fontsize=11, weight="bold")
    ax1.set_ylabel("Silhouette Score", fontsize=11, weight="bold")
    ax1.set_title("Overall Cluster Quality", fontsize=12, weight="bold", pad=15)
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)

    # Right plot: Distributions
    ax2 = fig.add_subplot(gs[0, 1])
    top_k_values = overall_scores["num_clusters"].head(top_n).to_list()

    per_geocode_scores = silhouette_df.filter(
        pl.col("geocode").is_not_null() & pl.col("num_clusters").is_in(top_k_values)
    )

    plot_data = per_geocode_scores.select(
        ["num_clusters", "silhouette_score"]
    ).to_pandas()

    sns.boxplot(
        data=plot_data,
        x="num_clusters",
        y="silhouette_score",
        palette="Set2",
        ax=ax2,
        width=0.5,
    )

    sns.violinplot(
        data=plot_data,
        x="num_clusters",
        y="silhouette_score",
        palette="Set2",
        ax=ax2,
        alpha=0.3,
        inner=None,
    )

    for k in top_k_values:
        overall_score = overall_scores.filter(pl.col("num_clusters") == k)[
            "silhouette_score"
        ][0]
        k_index = top_k_values.index(k)
        ax2.scatter(
            [k_index],
            [overall_score],
            color="red",
            s=100,
            zorder=10,
            marker="D",
            edgecolors="darkred",
            linewidths=1.5,
            label="Overall" if k_index == 0 else "",
        )

    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11, weight="bold")
    ax2.set_ylabel("Silhouette Score", fontsize=11, weight="bold")
    ax2.set_title(
        f"Score Distributions (Top {top_n})", fontsize=12, weight="bold", pad=15
    )
    ax2.grid(True, alpha=0.3, axis="y", linestyle="-", linewidth=0.5)
    ax2.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.suptitle("Cluster Optimization Summary", fontsize=16, weight="bold", y=1.02)

    if save_path:
        logger.info(f"Saving optimization summary plot to {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
