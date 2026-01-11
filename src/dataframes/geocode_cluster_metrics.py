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

Using multiple metrics provides more robust k selection than any single metric alone.
"""

import logging

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
    # Normalized scores for combining metrics (all scaled to [0, 1])
    silhouette_normalized = dy.Float64(nullable=False)
    calinski_harabasz_normalized = dy.Float64(nullable=False)
    davies_bouldin_normalized = dy.Float64(nullable=False)
    # Combined score using weighted average of normalized scores
    combined_score = dy.Float64(nullable=False)


def build_geocode_cluster_metrics_df(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterMultiKSchema],
    weights: dict[str, float] | None = None,
) -> dy.DataFrame[GeocodeClusterMetricsSchema]:
    """
    Build multi-metric cluster validation scores for all clustering results.

    Computes silhouette score, Calinski-Harabasz index, and Davies-Bouldin index
    for each k value in the clustering results. Also computes normalized versions
    of each metric and a combined weighted score for easier comparison.

    Args:
        distance_matrix: Precomputed distance matrix between geocodes
        geocode_cluster_df: DataFrame with clustering results for all k values
        weights: Optional dict with metric weights for combined score.
                 Keys: "silhouette", "calinski_harabasz", "davies_bouldin"
                 Default: {"silhouette": 0.4, "calinski_harabasz": 0.3, "davies_bouldin": 0.3}

    Returns:
        DataFrame with validation metrics for all k values tested

    Notes:
        - Silhouette uses precomputed distance matrix directly
        - Calinski-Harabasz and Davies-Bouldin use the squareform distance matrix
          as a feature representation (each row = distances to all other points)
        - Davies-Bouldin is inverted for normalization (since lower is better)
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

        results.append(
            {
                "num_clusters": k,
                "silhouette_score": sil_score,
                "calinski_harabasz_score": ch_score,
                "davies_bouldin_score": db_score,
            }
        )

        logger.debug(
            f"k={k}: silhouette={sil_score:.4f}, "
            f"calinski_harabasz={ch_score:.2f}, "
            f"davies_bouldin={db_score:.4f}"
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

    # Combined weighted score
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

    Returns:
        Optimal k value, or None if no k meets the threshold criteria
    """
    df = metrics_df

    # Apply silhouette threshold filter if specified
    if min_silhouette_threshold is not None:
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

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


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
        "combined_score": (
            "Combined Score [0, 1]: Weighted average of normalized metrics.\n"
            "  • Higher values indicate better overall clustering\n"
            "  • Balances all three metrics for robust selection"
        ),
    }
