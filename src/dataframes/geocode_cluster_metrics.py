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

import numpy as np
import polars as pl
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from kneed import KneeLocator  # typed: ignore

import bioregion_rs
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

def build_geocode_cluster_metrics_df(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
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
        - Silhouette uses the precomputed distance matrix directly, which is
          standard for a metric defined on pairwise distances
        - Calinski-Harabasz and Davies-Bouldin use the UMAP embedding, the space
          the clustering actually happened in. They used to be handed the
          squareform distance matrix as a stand-in feature representation (each
          row = distances to all other points), which made the feature dimension
          equal to the number of geocodes, so the dispersion ratio moved with
          dataset size
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

    # Silhouette and inertia are defined on pairwise distances, so Rust computes
    # them from the condensed matrix. Calinski-Harabasz and Davies-Bouldin need a
    # feature matrix, and the feature matrix is now the raw count matrix rather
    # than a 32-column embedding -- marshalling that into Rust peaked at 5.3 GB
    # on a country-scale run, so they are computed here, on the numpy array,
    # where no copy is needed.
    df = bioregion_rs.build_geocode_cluster_metrics(
        distance_matrix.condensed().tolist(),
        geocode_cluster_df,
    )
    df = _add_feature_space_metrics(df, distance_matrix, geocode_cluster_df)
    df = _add_combined_score(df, weights)

    logger.info(
        f"Computed cluster metrics. Best combined score at k="
        f"{df.sort('combined_score', descending=True)['num_clusters'][0]}"
    )

    return df


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
    metrics_df: pl.DataFrame,
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


def _find_elbow_point(
    metrics_df: pl.DataFrame,
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
    metrics_df: pl.DataFrame,
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
    metrics_df: pl.DataFrame,
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


def _normalize_min_max(values: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """Scale to [0, 1], flipping first when a lower raw value is the better one.

    A constant column maps to all-ones: every k is equally good on that metric,
    so it should not tip the combined score either way.
    """
    if lower_is_better:
        values = -values
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def _add_feature_space_metrics(
    metrics_df: pl.DataFrame,
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
) -> pl.DataFrame:
    """Calinski-Harabasz and Davies-Bouldin, computed on the feature matrix.

    Both need a feature space rather than pairwise distances. Since the pipeline
    no longer builds an embedding, the feature space is the count matrix itself
    -- the same rows the distances are computed from.
    """
    features = distance_matrix.features()
    ch: list[float] = []
    db: list[float] = []
    for k in metrics_df["num_clusters"].to_list():
        labels = (
            geocode_cluster_df.filter(pl.col("num_clusters") == k)
            .sort("geocode")["cluster"]
            .to_numpy()
        )
        # Both are undefined for a single group, and sklearn raises rather than
        # returning a sentinel.
        if len(np.unique(labels)) < 2:
            ch.append(0.0)
            db.append(0.0)
            continue
        ch.append(float(calinski_harabasz_score(features, labels)))
        db.append(float(davies_bouldin_score(features, labels)))
    return metrics_df.with_columns(
        calinski_harabasz_score=pl.Series(ch, dtype=pl.Float64),
        davies_bouldin_score=pl.Series(db, dtype=pl.Float64),
    )


def _add_combined_score(
    metrics_df: pl.DataFrame, weights: dict[str, float]
) -> pl.DataFrame:
    """Normalise each metric across k, then weight them into one score.

    Moved here from Rust along with the two metrics it combines, so that the
    weighting lives next to the weights rather than a language boundary away.
    """
    total = sum(weights[k] for k in ("silhouette", "calinski_harabasz", "davies_bouldin"))
    sil = metrics_df["silhouette_score"].to_numpy()
    ch = metrics_df["calinski_harabasz_score"].to_numpy()
    db = metrics_df["davies_bouldin_score"].to_numpy()
    ine = metrics_df["inertia"].to_numpy()

    # Silhouette is already on a fixed [-1, 1], so it is mapped rather than
    # min-max scaled; the others have no fixed range.
    sil_norm = (sil + 1.0) / 2.0
    ch_norm = _normalize_min_max(ch, lower_is_better=False)
    db_norm = _normalize_min_max(db, lower_is_better=True)
    ine_norm = _normalize_min_max(ine, lower_is_better=True)
    combined = (
        weights["silhouette"] * sil_norm
        + weights["calinski_harabasz"] * ch_norm
        + weights["davies_bouldin"] * db_norm
    ) / total
    return metrics_df.with_columns(
        silhouette_normalized=pl.Series(sil_norm, dtype=pl.Float64),
        calinski_harabasz_normalized=pl.Series(ch_norm, dtype=pl.Float64),
        davies_bouldin_normalized=pl.Series(db_norm, dtype=pl.Float64),
        inertia_normalized=pl.Series(ine_norm, dtype=pl.Float64),
        combined_score=pl.Series(combined, dtype=pl.Float64),
    )
