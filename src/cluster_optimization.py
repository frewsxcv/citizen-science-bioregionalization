"""
Cluster optimization module for automatically determining the optimal number of clusters.

This module uses the elbow method (Kneedle algorithm) to automatically select
the optimal number of clusters by finding the point where adding more clusters
stops providing significant reduction in within-cluster variance (inertia).
"""

import logging
from typing import Tuple

import dataframely as dy
import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterMultiKSchema
from src.dataframes.geocode_cluster_metrics import (
    GeocodeClusterMetricsSchema,
    build_geocode_cluster_metrics_df,
    select_optimal_k_elbow,
)
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)


def optimize_num_clusters(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterMultiKSchema],
    elbow_sensitivity: float = 1.0,
) -> Tuple[int, dy.DataFrame[GeocodeClusterMetricsSchema]]:
    """
    Find optimal number of clusters using the elbow method (Kneedle algorithm).

    This function computes cluster validation metrics for all k values and uses
    the elbow method to find the point where adding more clusters stops providing
    significant reduction in within-cluster variance (inertia).

    Args:
        distance_matrix: Precomputed distance matrix between geocodes
        geocode_cluster_df: DataFrame with clustering results for all k values to test
        elbow_sensitivity: Kneedle algorithm sensitivity parameter (S). Default 1.0.
                          Higher values (e.g., 2.0) make detection more conservative,
                          lower values (e.g., 0.5) make it more aggressive.

    Returns:
        A tuple containing:
        - optimal_k: The number of clusters at the elbow point
        - metrics_df: DataFrame with all metrics for all tested k values

    Example:
        >>> cluster_df = build_geocode_cluster_multi_k_df(
        ...     geocode_lf, distance_matrix, connectivity_matrix,
        ...     min_k=2, max_k=15
        ... )
        >>> optimal_k, metrics_df = optimize_num_clusters(
        ...     distance_matrix, cluster_df
        ... )
        >>> print(f"Optimal number of clusters: {optimal_k}")
    """
    # Compute all cluster validation metrics (including inertia for elbow method)
    metrics_df = build_geocode_cluster_metrics_df(
        distance_matrix,
        geocode_cluster_df,
    )

    # Select optimal k using elbow method
    optimal_k = select_optimal_k_elbow(
        metrics_df,
        sensitivity=elbow_sensitivity,
    )

    if optimal_k is None:
        logger.warning(
            "Elbow method could not find a clear elbow point. "
            "Selecting k with highest combined score as fallback."
        )
        # Fallback to k with highest combined score
        best_row = metrics_df.sort("combined_score", descending=True).head(1)
        optimal_k = int(best_row["num_clusters"][0])

    # Log the metrics for the selected k
    selected_metrics = metrics_df.filter(pl.col("num_clusters") == optimal_k)
    logger.info(
        f"Optimal k={optimal_k} selected via elbow method:\n"
        f"  Silhouette: {selected_metrics['silhouette_score'][0]:.4f}\n"
        f"  Calinski-Harabasz: {selected_metrics['calinski_harabasz_score'][0]:.2f}\n"
        f"  Davies-Bouldin: {selected_metrics['davies_bouldin_score'][0]:.4f}\n"
        f"  Inertia: {selected_metrics['inertia'][0]:.2f}"
    )

    return optimal_k, metrics_df


# Alias for backwards compatibility
optimize_num_clusters_multi_metric = optimize_num_clusters
