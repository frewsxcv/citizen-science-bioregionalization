"""
Cluster optimization module for automatically determining the optimal number of clusters.

This module provides functionality to test multiple clustering configurations and
select the optimal number of clusters based on silhouette scores.
"""

import logging
from typing import Tuple

import dataframely as dy
import polars as pl

from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreSchema
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)


def optimize_num_clusters(
    distance_matrix: GeocodeDistanceMatrix,
    connectivity_matrix: GeocodeConnectivityMatrix,
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
    min_k: int = 2,
    max_k: int = 20,
) -> Tuple[int, dy.DataFrame[GeocodeSilhouetteScoreSchema]]:
    """
    Test clustering with k from min_k to max_k and find optimal number of clusters.

    This function iterates through different values of k (number of clusters),
    performs hierarchical clustering for each k, computes silhouette scores,
    and returns the k with the highest overall silhouette score.

    Args:
        distance_matrix: Precomputed distance matrix between geocodes
        connectivity_matrix: Spatial connectivity constraints for clustering
        geocode_lf: LazyFrame containing geocode information
        min_k: Minimum number of clusters to test (default: 2)
        max_k: Maximum number of clusters to test (default: 20)

    Returns:
        A tuple containing:
        - optimal_k: The number of clusters with the highest silhouette score
        - combined_silhouette_scores_df: DataFrame with silhouette scores for all tested k values

    Raises:
        ValueError: If min_k < 2 or max_k < min_k

    Example:
        >>> optimal_k, scores_df = optimize_num_clusters(
        ...     distance_matrix,
        ...     connectivity_matrix,
        ...     geocode_lf,
        ...     min_k=2,
        ...     max_k=15
        ... )
        >>> print(f"Optimal number of clusters: {optimal_k}")
    """
    if min_k < 2:
        raise ValueError(f"min_k must be at least 2, got {min_k}")
    if max_k < min_k:
        raise ValueError(f"max_k ({max_k}) must be >= min_k ({min_k})")

    # Get number of geocodes to validate k range
    num_geocodes = geocode_lf.select(pl.len()).collect().item()
    if max_k >= num_geocodes:
        logger.warning(
            f"max_k ({max_k}) is >= number of geocodes ({num_geocodes}). "
            f"Reducing max_k to {num_geocodes - 1}"
        )
        max_k = num_geocodes - 1

    logger.info(
        f"Testing {max_k - min_k + 1} cluster configurations (k={min_k} to k={max_k})"
    )

    all_silhouette_dfs = []

    for k in range(min_k, max_k + 1):
        logger.info(f"Testing k={k}...")

        # Perform clustering with k clusters
        geocode_cluster_df = GeocodeClusterSchema.build_df(
            geocode_lf,
            distance_matrix,
            connectivity_matrix,
            num_clusters=k,
        )

        # Compute silhouette scores
        silhouette_df = GeocodeSilhouetteScoreSchema.build_df(
            distance_matrix,
            geocode_cluster_df,
            num_clusters=k,
        )

        all_silhouette_dfs.append(silhouette_df)

        overall_score = silhouette_df.filter(pl.col("geocode").is_null())[
            "silhouette_score"
        ][0]
        logger.info(f"Completed k={k} - Silhouette score: {overall_score:.4f}")

    # Combine all silhouette scores into a single dataframe
    combined_silhouette_scores = pl.concat(all_silhouette_dfs)

    # Find optimal k based on highest overall silhouette score
    optimal_k = select_optimal_k(combined_silhouette_scores)

    if optimal_k is None:
        logger.warning(
            "No k value met minimum threshold. Selecting k with highest score."
        )
        optimal_k = select_optimal_k(combined_silhouette_scores, min_threshold=None)
        if optimal_k is None:
            raise RuntimeError(
                "Could not determine optimal k - all silhouette scores may be invalid"
            )

    logger.info(f"Optimal number of clusters: k={optimal_k}")

    return optimal_k, combined_silhouette_scores


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
