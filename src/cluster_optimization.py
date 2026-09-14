"""
Cluster optimization module for automatically determining the optimal number of clusters.

k is chosen by the combined score, which aggregates the silhouette,
Calinski-Harabasz and Davies-Bouldin indices. The elbow method (Kneedle on
inertia) is retained only as a fallback: it was the primary selector, and
routinely disagreed with every other metric.
"""

import logging
from typing import Tuple

import bioregion_rs
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
        f"  Silhouette: {silhouette:.4f}\n"
        f"  Calinski-Harabasz: {selected_metrics['calinski_harabasz_score'][0]:.2f}\n"
        f"  Davies-Bouldin: {selected_metrics['davies_bouldin_score'][0]:.4f}\n"
        f"  Inertia: {selected_metrics['inertia'][0]:.2f}"
    )

    _log_abundance_silhouette(distance_matrix, geocode_cluster_df, optimal_k, silhouette)

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


def _log_abundance_silhouette(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
    optimal_k: int,
    reported_silhouette: float,
) -> None:
    """Log the chosen partition's silhouette in the composition space too.

    The silhouette reported above is measured on the UMAP embedding, which is
    the space the clustering happened in but not the space the data lives in.
    Measured instead on Bray-Curtis distances over the composition vectors --
    presence bits or counts, whichever the run used -- the
    same partition scores very differently, and always lower. Across six
    configurations the reported figure ran 3.4x to 24x the abundance-space one,
    and twice the latter was negative -- geocodes closer on average to a
    neighbouring cluster than to their own -- while the reported value was 0.0966
    and 0.4400. The second of those sits above MIN_SILHOUETTE_THRESHOLD, so no
    warning fired for a partition with no support at all.

    Logged rather than acted on. It is a second reading, offered so the first one
    is not taken at face value.
    """
    abundance_condensed = distance_matrix.abundance_condensed()
    if abundance_condensed is None:
        return

    scores = bioregion_rs.build_geocode_silhouette_score(
        abundance_condensed.tolist(), geocode_cluster_df
    )
    at_k = scores.filter(pl.col("num_clusters") == optimal_k)
    if at_k.height == 0:
        return
    mean_score = at_k["silhouette_score"].mean()
    if mean_score is None:
        return
    abundance_silhouette = float(mean_score)  # type: ignore[arg-type]

    # A ratio only means something when both readings share a sign; across a
    # sign change it is noise dressed as a number.
    if abundance_silhouette > 1e-9:
        comparison = f"reported is {reported_silhouette / abundance_silhouette:.1f}x this"
    else:
        comparison = f"reported is {reported_silhouette:.4f}, of the opposite sign"
    logger.info(
        f"  Silhouette in the composition space (pre-UMAP): "
        f"{abundance_silhouette:.4f} ({comparison}). "
        f"The reported figure measures the embedding; this one measures the data."
    )
    if abundance_silhouette < 0:
        logger.warning(
            f"Silhouette on the abundance metric is negative "
            f"({abundance_silhouette:.4f}) for k={optimal_k}: on average a geocode "
            f"sits closer to a neighbouring cluster than to its own. This "
            f"partition is not supported by the composition data, whatever the "
            f"embedding-space score says."
        )
