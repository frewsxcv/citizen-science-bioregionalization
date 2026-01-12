"""
Cluster optimization module for automatically determining the optimal number of clusters.

This module provides functionality to test multiple clustering configurations and
select the optimal number of clusters based on validation metrics.

Two optimization approaches are available:
1. Single-metric (silhouette score only) - via optimize_num_clusters()
2. Multi-metric (silhouette + Calinski-Harabasz + Davies-Bouldin) - via optimize_num_clusters_multi_metric()

The multi-metric approach is recommended for more robust cluster selection.
"""

import logging
from typing import Tuple

import dataframely as dy
import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterMultiKSchema
from src.dataframes.geocode_cluster_metrics import (
    GeocodeClusterMetricsSchema,
    build_geocode_cluster_metrics_df,
    select_optimal_k_multi_metric,
)
from src.dataframes.geocode_silhouette_score import (
    GeocodeSilhouetteScoreSchema,
    build_geocode_silhouette_score_df,
)
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)


def optimize_num_clusters(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterMultiKSchema],
) -> Tuple[int, dy.DataFrame[GeocodeSilhouetteScoreSchema]]:
    """
    Find optimal number of clusters from pre-computed clustering results.

    This function analyzes clustering results for multiple k values,
    computes silhouette scores, and returns the k with the highest overall
    silhouette score.

    Args:
        distance_matrix: Precomputed distance matrix between geocodes
        geocode_cluster_df: DataFrame with clustering results for all k values to test

    Returns:
        A tuple containing:
        - optimal_k: The number of clusters with the highest silhouette score
        - silhouette_scores_df: DataFrame with silhouette scores for all tested k values

    Example:
        >>> # Build clustering for k=2 to k=15
        >>> cluster_df = GeocodeClusterSchema.build_df(
        ...     geocode_lf,
        ...     distance_matrix,
        ...     connectivity_matrix,
        ...     min_k=2,
        ...     max_k=15
        ... )
        >>> # Find optimal k
        >>> optimal_k, scores_df = optimize_num_clusters(
        ...     distance_matrix,
        ...     cluster_df,
        ... )
        >>> print(f"Optimal number of clusters: {optimal_k}")
    """
    # Compute silhouette scores for all clustering results
    silhouette_scores_df = build_geocode_silhouette_score_df(
        distance_matrix,
        geocode_cluster_df,
    )

    # Find optimal k based on highest overall silhouette score
    optimal_k = select_optimal_k(silhouette_scores_df)

    if optimal_k is None:
        logger.warning(
            "No k value met minimum threshold. Selecting k with highest score."
        )
        optimal_k = select_optimal_k(silhouette_scores_df, min_threshold=None)
        if optimal_k is None:
            raise RuntimeError(
                "Could not determine optimal k - all silhouette scores may be invalid"
            )

    logger.info(f"Optimal number of clusters: k={optimal_k}")

    return optimal_k, silhouette_scores_df


def optimize_num_clusters_multi_metric(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterMultiKSchema],
    weights: dict[str, float] | None = None,
    min_silhouette_threshold: float | None = 0.25,
    selection_method: str = "combined",
    elbow_sensitivity: float = 1.0,
) -> Tuple[int, dy.DataFrame[GeocodeClusterMetricsSchema]]:
    """
    Find optimal number of clusters using multiple validation metrics.

    This function uses three complementary metrics to select the optimal k:
    - Silhouette Score: Measures cluster cohesion and separation
    - Calinski-Harabasz Index: Ratio of between-cluster to within-cluster variance
    - Davies-Bouldin Index: Average similarity between clusters (lower is better)

    Using multiple metrics provides more robust k selection than silhouette alone.

    Args:
        distance_matrix: Precomputed distance matrix between geocodes
        geocode_cluster_df: DataFrame with clustering results for all k values to test
        weights: Optional dict with metric weights for combined score.
                 Keys: "silhouette", "calinski_harabasz", "davies_bouldin"
                 Default: {"silhouette": 0.4, "calinski_harabasz": 0.3, "davies_bouldin": 0.3}
        min_silhouette_threshold: Minimum acceptable silhouette score (default: 0.25)
                                  Set to None to disable threshold filtering.
        selection_method: Method for selecting k:
            - "combined": Use combined weighted score (default)
            - "silhouette": Use silhouette score only
            - "elbow": Find the elbow point in the inertia curve using Kneedle algorithm
        elbow_sensitivity: Kneedle algorithm sensitivity parameter (S). Default 1.0.
                          Higher values (e.g., 2.0) make detection more conservative,
                          lower values (e.g., 0.5) make it more aggressive.
                          Only used when selection_method="elbow".

    Returns:
        A tuple containing:
        - optimal_k: The number of clusters with the best combined score
        - metrics_df: DataFrame with all metrics for all tested k values

    Example:
        >>> cluster_df = build_geocode_cluster_multi_k_df(
        ...     geocode_lf, distance_matrix, connectivity_matrix,
        ...     min_k=2, max_k=15
        ... )
        >>> optimal_k, metrics_df = optimize_num_clusters_multi_metric(
        ...     distance_matrix, cluster_df
        ... )
        >>> print(f"Optimal number of clusters: {optimal_k}")
    """
    # Compute all cluster validation metrics
    metrics_df = build_geocode_cluster_metrics_df(
        distance_matrix,
        geocode_cluster_df,
        weights=weights,
    )

    # Select optimal k using multi-metric criteria
    optimal_k = select_optimal_k_multi_metric(
        metrics_df,
        min_silhouette_threshold=min_silhouette_threshold,
        selection_method=selection_method,
        elbow_sensitivity=elbow_sensitivity,
    )

    if optimal_k is None:
        # If elbow method failed to find a knee, fall back to combined method
        fallback_method = "combined" if selection_method == "elbow" else selection_method
        logger.warning(
            f"{selection_method} method could not determine optimal k. "
            f"Falling back to {fallback_method} method with no threshold."
        )
        optimal_k = select_optimal_k_multi_metric(
            metrics_df,
            min_silhouette_threshold=None,
            selection_method=fallback_method,
            elbow_sensitivity=elbow_sensitivity,
        )
        if optimal_k is None:
            raise RuntimeError(
                "Could not determine optimal k - all metrics may be invalid"
            )

    # Log the metrics for the selected k
    selected_metrics = metrics_df.filter(pl.col("num_clusters") == optimal_k)
    logger.info(
        f"Optimal k={optimal_k} selected via {selection_method} method:\n"
        f"  Silhouette: {selected_metrics['silhouette_score'][0]:.4f}\n"
        f"  Calinski-Harabasz: {selected_metrics['calinski_harabasz_score'][0]:.2f}\n"
        f"  Davies-Bouldin: {selected_metrics['davies_bouldin_score'][0]:.4f}\n"
        f"  Combined Score: {selected_metrics['combined_score'][0]:.4f}"
    )

    return optimal_k, metrics_df


def get_overall_silhouette_scores(
    silhouette_df: dy.DataFrame[GeocodeSilhouetteScoreSchema],
) -> pl.DataFrame:
    """
    Extract overall silhouette scores (one per num_clusters value).

    Filters the silhouette score dataframe to return only the rows where
    geocode is null, which represent the overall score for each clustering.

    Args:
        silhouette_df: DataFrame containing silhouette scores for multiple k values

    Returns:
        DataFrame with columns: num_clusters, silhouette_score
        Sorted by silhouette_score descending

    Example:
        >>> overall_scores = get_overall_silhouette_scores(combined_scores)
        >>> print(overall_scores)
        ┌──────────────┬──────────────────┐
        │ num_clusters ┆ silhouette_score │
        ├──────────────┼──────────────────┤
        │ 8            ┆ 0.42            │
        │ 7            ┆ 0.39            │
        │ 9            ┆ 0.38            │
        └──────────────┴──────────────────┘
    """
    return (
        silhouette_df.filter(pl.col("geocode").is_null())
        .select(["num_clusters", "silhouette_score"])
        .sort("silhouette_score", descending=True)
    )


def select_optimal_k(
    silhouette_df: dy.DataFrame[GeocodeSilhouetteScoreSchema],
    min_threshold: float | None = 0.25,
) -> int | None:
    """
    Select the optimal number of clusters based on silhouette scores.

    Returns the k value with the highest overall silhouette score that meets
    the minimum threshold. If no k meets the threshold, returns None.

    Args:
        silhouette_df: DataFrame containing silhouette scores for multiple k values
        min_threshold: Minimum acceptable silhouette score (default: 0.25)
                      Set to None to disable threshold filtering

    Returns:
        The optimal number of clusters, or None if no k meets the threshold

    Notes:
        Silhouette score interpretation:
        - 1.0: Perfect clustering
        - 0.7-1.0: Strong structure
        - 0.5-0.7: Reasonable structure
        - 0.25-0.5: Weak structure
        - < 0.25: No substantial structure
        - Negative: Points may be in wrong clusters

    Example:
        >>> optimal_k = select_optimal_k(combined_scores, min_threshold=0.25)
        >>> if optimal_k is None:
        ...     print("No clustering meets minimum quality threshold")
    """
    overall_scores = get_overall_silhouette_scores(silhouette_df)

    if min_threshold is not None:
        # Filter to scores above threshold
        valid_scores = overall_scores.filter(
            pl.col("silhouette_score") >= min_threshold
        )
        if len(valid_scores) == 0:
            return None
        return int(valid_scores["num_clusters"][0])
    else:
        # Return k with highest score regardless of threshold
        return int(overall_scores["num_clusters"][0])


def format_optimization_results(
    silhouette_df: dy.DataFrame[GeocodeSilhouetteScoreSchema],
) -> pl.DataFrame:
    """
    Format optimization results as a summary table.

    Creates a formatted table showing the silhouette score for each tested
    number of clusters, ranked by score.

    Args:
        silhouette_df: DataFrame containing silhouette scores for multiple k values

    Returns:
        DataFrame with columns: rank, num_clusters, silhouette_score
        Sorted by silhouette_score descending

    Example:
        >>> results_table = format_optimization_results(combined_scores)
        >>> print(results_table)
        ┌──────┬──────────────┬──────────────────┐
        │ rank ┆ num_clusters ┆ silhouette_score │
        ├──────┼──────────────┼──────────────────┤
        │ 1    ┆ 8            ┆ 0.4234          │
        │ 2    ┆ 7            ┆ 0.3912          │
        │ 3    ┆ 9            ┆ 0.3845          │
        └──────┴──────────────┴──────────────────┘
    """
    overall_scores = get_overall_silhouette_scores(silhouette_df)

    # Add rank column
    results = overall_scores.with_columns(
        (pl.int_range(1, pl.len() + 1)).alias("rank")
    ).select(["rank", "num_clusters", "silhouette_score"])

    return results
