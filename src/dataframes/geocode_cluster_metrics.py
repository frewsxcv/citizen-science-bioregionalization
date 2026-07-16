"""
Multi-metric cluster validation schema and builder.

This module provides a comprehensive set of cluster validation metrics
to enable more robust automatic cluster number selection:

- Silhouette Score: Measures how similar objects are to their own cluster
  compared to other clusters. Range: [-1, 1], higher is better.

- Calinski-Harabasz Index: Ratio of between-cluster to within-cluster variance.
  Higher values indicate better-defined clusters. No upper bound.

- Davies-Bouldin Index: Average similarity between clusters, where similarity
  compares distance between clusters with cluster size. Lower is better.

- Inertia (WCSS): Within-cluster sum of squares. Lower is better.
  Used for elbow method detection.

Using multiple metrics provides more robust k selection than any single metric alone.
"""

import logging
from typing import TypedDict

import dataframely as dy
import numpy as np
import polars as pl
from kneed import KneeLocator  # typed: ignore

import bioregion_rs
from src.dataframes.geocode_cluster import GeocodeClusterMultiKSchema
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)


class ElbowAnalysisResult(TypedDict):
    """Type for elbow analysis return value."""

    k_values: list[int]
    inertia_values: list[float]
    elbow_k: int
    distances: list[float]
    inertia_deltas: list[float]
    inertia_delta2: list[float]


class GeocodeClusterMetricsSchema(dy.Schema):
    """
    Schema for multi-metric cluster validation results.

    Stores overall cluster quality metrics for each tested k value.
    All metrics are computed at the clustering level (one row per k).
    """

    num_clusters = dy.UInt32(nullable=False)
    silhouette_score = dy.Float64(nullable=False)
    calinski_harabasz_score = dy.Float64(nullable=False)
    davies_bouldin_score = dy.Float64(nullable=False)
    inertia = dy.Float64(nullable=False)
    # Normalized scores for combining metrics (all scaled to [0, 1])
    silhouette_normalized = dy.Float64(nullable=False)
    calinski_harabasz_normalized = dy.Float64(nullable=False)
    davies_bouldin_normalized = dy.Float64(nullable=False)
    inertia_normalized = dy.Float64(nullable=False)
    # Combined score using weighted average of normalized scores
    combined_score = dy.Float64(nullable=False)


def build_geocode_cluster_metrics_df(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterMultiKSchema],
    weights: dict[str, float] | None = None,
) -> dy.DataFrame[GeocodeClusterMetricsSchema]:
    """
    Build multi-metric cluster validation scores for all clustering results.

    Computes silhouette score, Calinski-Harabasz index, Davies-Bouldin index,
    and inertia (WCSS) for each k value in the clustering results. Also computes
    normalized versions of each metric and a combined weighted score.

    Args:
        distance_matrix: Precomputed distance matrix between geocodes
        geocode_cluster_df: DataFrame with clustering results for all k values
        weights: Optional dict with metric weights for combined score.
                 Keys: "silhouette", "calinski_harabasz", "davies_bouldin"
                 Default: {"silhouette": 0.4, "calinski_harabasz": 0.3, "davies_bouldin": 0.3}
                 Note: inertia is not included in combined score (used for elbow method)

    Returns:
        DataFrame with validation metrics for all k values tested

    Notes:
        - Silhouette uses precomputed distance matrix directly
        - Calinski-Harabasz and Davies-Bouldin use the squareform distance matrix
          as a feature representation (each row = distances to all other points)
        - Davies-Bouldin is inverted for normalization (since lower is better)
        - Inertia is computed as within-cluster sum of squared distances
        - Combined score provides a single metric for ranking k values
    """
    if weights is None:
        weights = {
            "silhouette": 0.4,
            "calinski_harabasz": 0.3,
            "davies_bouldin": 0.3,
        }

    k_values = geocode_cluster_df["num_clusters"].unique().sort().to_list()
    logger.info(
        f"Computing cluster metrics for {len(k_values)} k values: {k_values[0]} to {k_values[-1]}"
    )

    df = bioregion_rs.build_geocode_cluster_metrics(
        distance_matrix.condensed().tolist(),
        geocode_cluster_df,
        weights["silhouette"],
        weights["calinski_harabasz"],
        weights["davies_bouldin"],
    )

    logger.info(
        f"Computed cluster metrics. Best combined score at k="
        f"{df.sort('combined_score', descending=True)['num_clusters'][0]}"
    )

    return GeocodeClusterMetricsSchema.validate(df)


def _compute_inertia(dm_square: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute within-cluster sum of squares (WCSS/inertia).

    For each cluster, computes the sum of squared distances from each point
    to the cluster centroid. Since we're working with a distance matrix rather
    than raw coordinates, we approximate by computing the mean pairwise distance
    within each cluster.

    Args:
        dm_square: Square distance matrix (n_samples x n_samples)
        labels: Cluster labels for each sample

    Returns:
        Total within-cluster sum of squares
    """
    unique_labels = np.unique(labels)
    total_inertia = 0.0

    for label in unique_labels:
        # Get indices of points in this cluster
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) <= 1:
            # Single-point clusters have zero inertia
            continue

        # Extract pairwise distances within this cluster
        cluster_distances = dm_square[np.ix_(cluster_indices, cluster_indices)]

        # Sum of squared distances within cluster (divide by 2 to avoid double counting)
        # Using squared distances directly since dm_square contains distances
        inertia = np.sum(cluster_distances**2) / (2 * len(cluster_indices))
        total_inertia += inertia

    return float(total_inertia)


def select_optimal_k_elbow(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    sensitivity: float = 1.0,
) -> int | None:
    """
    Select optimal k using the elbow method (Kneedle algorithm).

    Finds the point of maximum curvature in the inertia vs k plot. This is where
    adding more clusters stops providing significant reduction in within-cluster variance.

    Args:
        metrics_df: DataFrame with cluster metrics for all k values
        sensitivity: Kneedle algorithm sensitivity parameter (S). Default 1.0.
                    Higher values (e.g., 2.0) make detection more conservative,
                    lower values (e.g., 0.5) make it more aggressive.

    Returns:
        Optimal k value, or None if no clear elbow point is found

    Notes:
        Uses the Kneedle algorithm to robustly detect the elbow point in the inertia curve.
    """
    optimal_k = bioregion_rs.select_optimal_k_elbow(
        metrics_df["num_clusters"].to_list(),
        metrics_df["inertia"].to_list(),
        sensitivity,
    )
    if optimal_k is not None:
        logger.info(f"Elbow method selected k={optimal_k}")
    return optimal_k


# Backwards compatibility alias
def select_optimal_k_multi_metric(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    min_silhouette_threshold: float | None = 0.25,
    selection_method: str = "elbow",
    elbow_sensitivity: float = 1.0,
) -> int | None:
    """
    Backwards compatibility wrapper for select_optimal_k_elbow.

    Always uses the elbow method. If elbow method fails to find a clear elbow,
    falls back to selecting k with the highest combined score.

    Args:
        metrics_df: DataFrame with cluster metrics for all k values
        min_silhouette_threshold: Ignored (kept for compatibility)
        selection_method: Ignored (kept for compatibility)
        elbow_sensitivity: Kneedle algorithm sensitivity parameter

    Returns:
        Optimal k value, or None if metrics_df is empty
    """
    optimal_k = select_optimal_k_elbow(metrics_df, sensitivity=elbow_sensitivity)

    if optimal_k is None and len(metrics_df) > 0:
        # Fallback to highest combined score
        logger.warning("Elbow method failed, falling back to highest combined score")
        best_row = metrics_df.sort("combined_score", descending=True).head(1)
        optimal_k = int(best_row["num_clusters"][0])

    return optimal_k


def _find_elbow_point(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    sensitivity: float = 1.0,
) -> int | None:
    """
    Find the elbow point in the inertia curve using the Kneedle algorithm.

    The elbow point is where the rate of decrease in inertia sharply changes,
    indicating that adding more clusters provides diminishing returns.

    Uses the Kneedle algorithm (Satopaa et al., 2011) which applies a difference
    curve approach to robustly detect knee points in noisy data.

    Args:
        metrics_df: DataFrame with cluster metrics including inertia values
        sensitivity: Kneedle algorithm sensitivity parameter (S). Default 1.0.
                    Higher values make detection more conservative (fewer points
                    detected as knees), lower values make it more aggressive.

    Returns:
        The k value at the elbow point, or None if it cannot be determined

    References:
        Satopaa, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011).
        Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior.
        31st International Conference on Distributed Computing Systems Workshops.
    """
    df = metrics_df.sort("num_clusters")

    if len(df) < 3:
        logger.warning("Need at least 3 k values to find elbow point")
        return None

    k_values = df["num_clusters"].to_list()
    inertia_values = df["inertia"].to_list()

    try:
        kneedle = KneeLocator(
            x=k_values,
            y=inertia_values,
            curve="convex",  # Inertia curves are convex (decreasing at decreasing rate)
            direction="decreasing",  # Inertia decreases as k increases
            S=sensitivity,
        )

        if kneedle.elbow is None:
            logger.warning("Kneedle algorithm could not find an elbow point")
            return None

        elbow_k = int(kneedle.elbow)
        logger.debug(
            f"Kneedle algorithm selected k={elbow_k} (sensitivity={sensitivity})"
        )
        return elbow_k

    except Exception as e:
        logger.error(f"Kneedle algorithm failed: {e}")
        return None


def get_elbow_analysis(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    sensitivity: float = 1.0,
) -> ElbowAnalysisResult:
    """
    Get detailed elbow analysis data for visualization using the Kneedle algorithm.

    Returns data useful for plotting the elbow curve and understanding
    the Kneedle algorithm's selection rationale.

    Args:
        metrics_df: DataFrame with cluster metrics including inertia values
        sensitivity: Kneedle algorithm sensitivity parameter (S). Default 1.0.

    Returns:
        Dictionary containing:
        - k_values: List of k values tested
        - inertia_values: Corresponding inertia values
        - elbow_k: The detected elbow point (from Kneedle algorithm)
        - distances: Normalized y-distance from each point to the baseline
                    (computed by Kneedle algorithm)
        - inertia_deltas: First derivative of inertia (rate of change)
        - inertia_delta2: Second derivative of inertia (acceleration)
    """
    df = metrics_df.sort("num_clusters")

    k_values = df["num_clusters"].to_list()
    inertia_values = df["inertia"].to_list()

    # Compute derivatives for additional analysis
    inertia_deltas = []
    inertia_delta2 = []

    for i in range(len(inertia_values)):
        if i == 0:
            inertia_deltas.append(0.0)
        else:
            inertia_deltas.append(inertia_values[i] - inertia_values[i - 1])

    for i in range(len(inertia_deltas)):
        if i == 0:
            inertia_delta2.append(0.0)
        else:
            inertia_delta2.append(inertia_deltas[i] - inertia_deltas[i - 1])

    # Use Kneedle algorithm to get elbow point and distance data
    elbow_k = _find_elbow_point(metrics_df, sensitivity=sensitivity)

    # Get normalized distance data from Kneedle algorithm for visualization
    distances = []
    try:
        kneedle = KneeLocator(
            x=k_values,
            y=inertia_values,
            curve="convex",
            direction="decreasing",
            S=sensitivity,
        )

        # The Kneedle algorithm provides normalized y-distances
        # These represent the distance from each point to the baseline
        if hasattr(kneedle, "y_difference"):
            # y_difference contains the normalized distances used for knee detection
            distances = list(kneedle.y_difference)
        else:
            # Fallback: compute normalized distances manually
            y_norm = (np.array(inertia_values) - min(inertia_values)) / (
                max(inertia_values) - min(inertia_values) + 1e-10
            )
            x_norm = (np.array(k_values) - min(k_values)) / (
                max(k_values) - min(k_values) + 1e-10
            )
            # Distance from normalized curve to diagonal
            distances = list(y_norm - x_norm)

    except Exception as e:
        logger.warning(f"Could not extract Kneedle distance data: {e}")
        distances = [0.0] * len(k_values)

    return {
        "k_values": k_values,
        "inertia_values": inertia_values,
        "elbow_k": elbow_k if elbow_k is not None else 0,
        "distances": distances,
        "inertia_deltas": inertia_deltas,
        "inertia_delta2": inertia_delta2,
    }


def get_metrics_summary(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
) -> pl.DataFrame:
    """
    Format metrics as a summary table for display.

    Returns DataFrame sorted by combined score with rank column added.
    """
    return (
        metrics_df.sort("combined_score", descending=True)
        .with_columns((pl.int_range(1, pl.len() + 1)).alias("rank"))
        .select(
            [
                "rank",
                "num_clusters",
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
                "inertia",
                "combined_score",
            ]
        )
    )


def get_metric_interpretations() -> dict[str, str]:
    """
    Return interpretation guidelines for each metric.

    Useful for documentation and UI display.
    """
    return {
        "silhouette_score": (
            "Silhouette Score [-1, 1]: Measures cluster cohesion and separation.\n"
            "  • 0.7-1.0: Strong structure\n"
            "  • 0.5-0.7: Reasonable structure\n"
            "  • 0.25-0.5: Weak structure\n"
            "  • < 0.25: No substantial structure"
        ),
        "calinski_harabasz_score": (
            "Calinski-Harabasz Index [0, ∞): Ratio of between-cluster to "
            "within-cluster variance.\n"
            "  • Higher values indicate better-defined clusters\n"
            "  • No absolute threshold; compare relative values across k"
        ),
        "davies_bouldin_score": (
            "Davies-Bouldin Index [0, ∞): Average similarity between clusters.\n"
            "  • Lower values indicate better clustering\n"
            "  • 0 = perfect clustering (rarely achieved)\n"
            "  • Values < 1 generally indicate good separation"
        ),
        "inertia": (
            "Inertia (WCSS) [0, ∞): Within-cluster sum of squared distances.\n"
            "  • Lower values indicate tighter clusters\n"
            "  • Used for elbow method: look for the 'elbow' point\n"
            "  • Always decreases as k increases"
        ),
        "combined_score": (
            "Combined Score [0, 1]: Weighted average of normalized metrics.\n"
            "  • Higher values indicate better overall clustering\n"
            "  • Balances all three metrics for robust selection"
        ),
    }
