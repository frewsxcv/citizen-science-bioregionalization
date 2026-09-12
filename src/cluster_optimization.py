"""
Cluster optimization module for automatically determining the optimal number of clusters.

k is chosen by the combined score, which aggregates the silhouette,
Calinski-Harabasz and Davies-Bouldin indices. The elbow method (Kneedle on
inertia) is retained only as a fallback: it was the primary selector, and
routinely disagreed with every other metric.
"""

import logging
from typing import Tuple

import polars as pl

from src.dataframes.geocode_cluster_metrics import (
    build_geocode_cluster_metrics_df,
    select_optimal_k_elbow,
)
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)

# Below this, a silhouette is conventionally read as "no substantial structure".
# The chosen partition is still returned -- the caller asked for one -- but the
# run warns rather than presenting it as a result.
MIN_SILHOUETTE_THRESHOLD = 0.25


def optimize_num_clusters(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
    elbow_sensitivity: float = 1.0,
) -> Tuple[int, pl.DataFrame]:
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

    # Select k on the combined score, which aggregates silhouette,
    # Calinski-Harabasz and Davies-Bouldin. Kneedle-on-inertia was previously the
    # primary selector and routinely disagreed with every other metric -- on the
    # Colombia data it chose k=5-6 while all three, and the combined score, ranked
    # k=2 highest. It is kept as a fallback for a degenerate score column.
    selection_method = "combined score"
    best_row = metrics_df.sort("combined_score", descending=True).head(1)
    optimal_k = int(best_row["num_clusters"][0])

    if metrics_df["combined_score"].n_unique() <= 1:
        selection_method = "elbow method"
        logger.warning(
            "Combined score is constant across k; falling back to the elbow method."
        )
        elbow_k = select_optimal_k_elbow(metrics_df, sensitivity=elbow_sensitivity)
        if elbow_k is not None:
            optimal_k = elbow_k

    # Log the metrics for the selected k
    selected_metrics = metrics_df.filter(pl.col("num_clusters") == optimal_k)
    silhouette = selected_metrics["silhouette_score"][0]
    logger.info(
        f"Optimal k={optimal_k} selected via {selection_method}:\n"
        # Measured on Bray-Curtis distances over the counts themselves. It used
        # to be measured on the UMAP embedding and read far higher -- 0.3897
        # against 0.0564 for the same partition on the published dataset -- so
        # figures from before UMAP was removed are not comparable with these.
        f"  Silhouette: {silhouette:.4f}\n"
        f"  Calinski-Harabasz: {selected_metrics['calinski_harabasz_score'][0]:.2f}\n"
        f"  Davies-Bouldin: {selected_metrics['davies_bouldin_score'][0]:.4f}\n"
        f"  Inertia: {selected_metrics['inertia'][0]:.2f}"
    )

    # A silhouette this low means the partition is not supported by the data,
    # whatever k was chosen. Say so rather than presenting it as a finding.
    if silhouette < MIN_SILHOUETTE_THRESHOLD:
        logger.warning(
            f"Silhouette for k={optimal_k} is {silhouette:.4f}, below "
            f"{MIN_SILHOUETTE_THRESHOLD}: the data show no substantial cluster "
            f"structure and this partition should not be read as a set of "
            f"well-separated bioregions."
        )

    return optimal_k, metrics_df


# Alias for backwards compatibility
optimize_num_clusters_multi_metric = optimize_num_clusters
