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
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

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

    # Validate weights sum to 1
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        logger.warning(
            f"Metric weights sum to {weight_sum}, not 1.0. Normalizing weights."
        )
        weights = {k: v / weight_sum for k, v in weights.items()}

    # Get squareform distance matrix for feature-based metrics
    dm_square = distance_matrix.squareform()

    # Get unique k values from the geocode_cluster_df
    k_values = geocode_cluster_df["num_clusters"].unique().sort().to_list()

    logger.info(
        f"Computing cluster metrics for {len(k_values)} k values: {k_values[0]} to {k_values[-1]}"
    )

    results: list[dict[str, float | int]] = []

    for k in k_values:
        # Filter to just this k value
        k_df = geocode_cluster_df.filter(pl.col("num_clusters") == k)
        labels = k_df["cluster"].to_numpy()

        # Compute silhouette score using precomputed distances
        sil_score = float(
            silhouette_score(
                X=dm_square,
                labels=labels,
                metric="precomputed",
            )
        )

        # Compute Calinski-Harabasz using distance matrix as features
        # Each row represents a geocode's distances to all other geocodes
        ch_score = float(
            calinski_harabasz_score(
                X=dm_square,
                labels=labels,
            )
        )

        # Compute Davies-Bouldin using distance matrix as features
        db_score = float(
            davies_bouldin_score(
                X=dm_square,
                labels=labels,
            )
        )

        # Compute inertia (within-cluster sum of squares)
        inertia_val = _compute_inertia(dm_square, labels)

        results.append(
            {
                "num_clusters": k,
                "silhouette_score": sil_score,
                "calinski_harabasz_score": ch_score,
                "davies_bouldin_score": db_score,
                "inertia": inertia_val,
            }
        )

        logger.debug(
            f"k={k}: silhouette={sil_score:.4f}, "
            f"calinski_harabasz={ch_score:.2f}, "
            f"davies_bouldin={db_score:.4f}, "
            f"inertia={inertia_val:.2f}"
        )

    # Create DataFrame and compute normalized scores
    df = pl.DataFrame(results)

    # Normalize scores to [0, 1] range for combining
    df = _add_normalized_scores(df, weights)

    logger.info(
        f"Computed cluster metrics. Best combined score at k="
        f"{df.sort('combined_score', descending=True)['num_clusters'][0]}"
    )

    return GeocodeClusterMetricsSchema.validate(
        df.with_columns(pl.col("num_clusters").cast(pl.UInt32))
    )


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


def _add_normalized_scores(
    df: pl.DataFrame,
    weights: dict[str, float],
) -> pl.DataFrame:
    """
    Add normalized scores and combined weighted score to metrics DataFrame.

    Normalization approach:
    - Silhouette: Linear rescale from [-1, 1] to [0, 1]
    - Calinski-Harabasz: Min-max normalization (higher is better)
    - Davies-Bouldin: Inverted min-max normalization (lower is better)
    - Inertia: Inverted min-max normalization (lower is better)
    """
    # Silhouette normalization: [-1, 1] -> [0, 1]
    df = df.with_columns(
        ((pl.col("silhouette_score") + 1) / 2).alias("silhouette_normalized")
    )

    # Calinski-Harabasz normalization: min-max scaling
    ch_min = float(df["calinski_harabasz_score"].min())  # type: ignore[arg-type]
    ch_max = float(df["calinski_harabasz_score"].max())  # type: ignore[arg-type]
    ch_range = ch_max - ch_min if ch_max != ch_min else 1.0

    df = df.with_columns(
        ((pl.col("calinski_harabasz_score") - ch_min) / ch_range).alias(
            "calinski_harabasz_normalized"
        )
    )

    # Davies-Bouldin normalization: inverted min-max scaling (lower DB is better)
    db_min = float(df["davies_bouldin_score"].min())  # type: ignore[arg-type]
    db_max = float(df["davies_bouldin_score"].max())  # type: ignore[arg-type]
    db_range = db_max - db_min if db_max != db_min else 1.0

    df = df.with_columns(
        (1 - (pl.col("davies_bouldin_score") - db_min) / db_range).alias(
            "davies_bouldin_normalized"
        )
    )

    # Inertia normalization: inverted min-max scaling (lower inertia is better)
    inertia_min = float(df["inertia"].min())  # type: ignore[arg-type]
    inertia_max = float(df["inertia"].max())  # type: ignore[arg-type]
    inertia_range = inertia_max - inertia_min if inertia_max != inertia_min else 1.0

    df = df.with_columns(
        (1 - (pl.col("inertia") - inertia_min) / inertia_range).alias(
            "inertia_normalized"
        )
    )

    # Combined weighted score (inertia not included - used for elbow method)
    df = df.with_columns(
        (
            weights["silhouette"] * pl.col("silhouette_normalized")
            + weights["calinski_harabasz"] * pl.col("calinski_harabasz_normalized")
            + weights["davies_bouldin"] * pl.col("davies_bouldin_normalized")
        ).alias("combined_score")
    )

    return df


def select_optimal_k_multi_metric(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
    min_silhouette_threshold: float | None = 0.25,
    selection_method: str = "combined",
) -> int | None:
    """
    Select optimal k using multi-metric criteria.

    Args:
        metrics_df: DataFrame with cluster metrics for all k values
        min_silhouette_threshold: Minimum acceptable silhouette score.
                                  Set to None to disable filtering.
        selection_method: Method for selecting k:
            - "combined": Use combined weighted score (default)
            - "silhouette": Use silhouette score only (fallback to single metric)
            - "elbow": Use elbow method based on inertia curve.
              Finds the point of maximum curvature in the inertia vs k plot.

    Returns:
        Optimal k value, or None if no k meets the threshold criteria

    Notes:
        The "elbow" method finds the point of maximum curvature in the inertia
        curve. This is where adding more clusters stops providing significant
        reduction in within-cluster variance.
    """
    df = metrics_df

    # Apply silhouette threshold filter if specified (except for elbow method)
    if min_silhouette_threshold is not None and selection_method != "elbow":
        df = df.filter(pl.col("silhouette_score") >= min_silhouette_threshold)
        if len(df) == 0:
            logger.warning(
                f"No k values meet silhouette threshold of {min_silhouette_threshold}"
            )
            return None

    if selection_method == "combined":
        # Select k with highest combined score
        best_row = df.sort("combined_score", descending=True).head(1)
        return int(best_row["num_clusters"][0])

    elif selection_method == "silhouette":
        # Fallback to silhouette-only selection
        best_row = df.sort("silhouette_score", descending=True).head(1)
        return int(best_row["num_clusters"][0])

    elif selection_method == "elbow":
        # Use elbow method based on inertia curve
        optimal_k = _find_elbow_point(metrics_df)
        if optimal_k is not None:
            logger.info(f"Elbow method selected k={optimal_k}")
        return optimal_k

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


def _find_elbow_point(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
) -> int | None:
    """
    Find the elbow point in the inertia curve using the Kneedle algorithm.

    The elbow point is where the rate of decrease in inertia sharply changes,
    indicating that adding more clusters provides diminishing returns.

    Uses the perpendicular distance method: finds the point farthest from
    the line connecting the first and last points of the curve.

    Args:
        metrics_df: DataFrame with cluster metrics including inertia values

    Returns:
        The k value at the elbow point, or None if it cannot be determined
    """
    df = metrics_df.sort("num_clusters")

    if len(df) < 3:
        logger.warning("Need at least 3 k values to find elbow point")
        return None

    k_values = np.array(df["num_clusters"].to_list(), dtype=float)
    inertia_values = np.array(df["inertia"].to_list(), dtype=float)

    # Normalize both axes to [0, 1] for fair distance calculation
    k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    inertia_norm = (inertia_values - inertia_values.min()) / (
        inertia_values.max() - inertia_values.min() + 1e-10
    )

    # Line from first point to last point
    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])

    # Calculate perpendicular distance from each point to the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        logger.warning("Degenerate inertia curve (all values equal)")
        return None

    line_unit = line_vec / line_len

    distances = []
    for i in range(len(k_norm)):
        point = np.array([k_norm[i], inertia_norm[i]])
        point_vec = point - p1
        # Perpendicular distance = magnitude of cross product / line length
        cross = abs(point_vec[0] * line_unit[1] - point_vec[1] * line_unit[0])
        distances.append(cross)

    # Find the point with maximum distance (the elbow)
    elbow_idx = np.argmax(distances)
    elbow_k = int(k_values[elbow_idx])

    logger.debug(
        f"Elbow detection: max distance={distances[elbow_idx]:.4f} at k={elbow_k}"
    )

    return elbow_k


def get_elbow_analysis(
    metrics_df: dy.DataFrame[GeocodeClusterMetricsSchema],
) -> ElbowAnalysisResult:
    """
    Get detailed elbow analysis data for visualization.

    Returns data useful for plotting the elbow curve and understanding
    the selection rationale.

    Args:
        metrics_df: DataFrame with cluster metrics including inertia values

    Returns:
        Dictionary containing:
        - k_values: List of k values tested
        - inertia_values: Corresponding inertia values
        - elbow_k: The detected elbow point
        - distances: Perpendicular distances for each point
        - inertia_deltas: First derivative of inertia
        - inertia_delta2: Second derivative of inertia
    """
    df = metrics_df.sort("num_clusters")

    k_values = df["num_clusters"].to_list()
    inertia_values = df["inertia"].to_list()

    # Compute derivatives
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

    # Compute distances for visualization
    k_arr = np.array(k_values, dtype=float)
    inertia_arr = np.array(inertia_values, dtype=float)

    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() + 1e-10)
    inertia_norm = (inertia_arr - inertia_arr.min()) / (
        inertia_arr.max() - inertia_arr.min() + 1e-10
    )

    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    distances = []
    if line_len > 1e-10:
        line_unit = line_vec / line_len
        for i in range(len(k_norm)):
            point = np.array([k_norm[i], inertia_norm[i]])
            point_vec = point - p1
            cross = abs(point_vec[0] * line_unit[1] - point_vec[1] * line_unit[0])
            distances.append(float(cross))
    else:
        distances = [0.0] * len(k_values)

    elbow_k = _find_elbow_point(metrics_df)

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
