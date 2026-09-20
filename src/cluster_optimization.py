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


def score_all_k(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
    weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Score every cut of the tree. Chooses nothing.

    Args:
        distance_matrix: Precomputed distances between geocodes.
        geocode_cluster_df: The multi-k clustering, one row per (geocode, k).
        weights: Combined-score weights. See defaults.METRIC_WEIGHTS.

    Returns:
        One row per k, carrying silhouette, Calinski-Harabasz, Davies-Bouldin,
        inertia and the combined score.
    """
    return build_geocode_cluster_metrics_df(
        distance_matrix, geocode_cluster_df, weights=weights
    )


def select_k(
    metrics_df: pl.DataFrame,
    pinned_k: int | None = None,
    elbow_sensitivity: float = 1.0,
) -> int:
    """Pick a k from the scores. Reports nothing about whether it is any good.

    Separate from `score_all_k` because scoring every cut and preferring one of
    them are different claims, and welding them together made the preference
    look load-bearing. It is not: see defaults.DEFAULT_DISPLAY_LEVEL for what
    actually decides the published cut.

    Args:
        metrics_df: Output of `score_all_k`.
        pinned_k: A k asked for via --num-clusters, which wins outright.
        elbow_sensitivity: Kneedle sensitivity for the fallback path.

    Returns:
        The selected k.
    """
    selection_method = "combined score"
    best_row = metrics_df.sort("combined_score", descending=True).head(1)
    optimal_k = int(best_row["num_clusters"][0])

    # Pinning is a stated preference, not a discovery, and is logged as one.
    # No metric here favours more regions: measured on Colombia at presence and
    # k from 2 to 12, the silhouette the selector sees falls monotonically from
    # 0.4300 to 0.1159, Calinski-Harabasz falls and Davies-Bouldin rises. Every
    # criterion prefers the smallest k in range, so a larger one can only be
    # asked for.
    if pinned_k is not None:
        available = metrics_df["num_clusters"].to_list()
        if pinned_k not in available:
            raise ValueError(
                f"--num-clusters={pinned_k} is outside the range tested "
                f"({min(available)}..{max(available)}). Widen --min-clusters/"
                f"--max-clusters, or pick a k inside it."
            )
        if pinned_k != optimal_k:
            logger.info(
                f"k pinned to {pinned_k}; the combined score preferred "
                f"{optimal_k}. This is a choice about how many regions to draw, "
                f"not a claim that {pinned_k} fits the data better."
            )
        logger.info(f"select_k: k={pinned_k} (pinned via --num-clusters)")
        return pinned_k

    if metrics_df["combined_score"].n_unique() <= 1:
        selection_method = "elbow method"
        logger.warning(
            "Combined score is constant across k; falling back to the elbow method."
        )
        elbow_k = select_optimal_k_elbow(metrics_df, sensitivity=elbow_sensitivity)
        if elbow_k is not None:
            optimal_k = elbow_k

    logger.info(f"select_k: k={optimal_k} (via {selection_method})")
    return optimal_k


def report_partition(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
    metrics_df: pl.DataFrame,
    k: int,
) -> None:
    """Log how well supported one cut is, and warn when it is not.

    Takes the cut explicitly rather than assuming the selector's. This used to
    live inside the selector and describe whatever the selector chose, which is
    not what the run publishes -- so the warning that says "this partition
    should not be read as a set of well-separated bioregions" could describe a
    partition no output contains, and stay silent about the one every output is
    built from.

    Args:
        k: The cut to describe. Callers want the published level.
    """
    selected_metrics = metrics_df.filter(pl.col("num_clusters") == k)
    if selected_metrics.height == 0:
        logger.warning(f"report_partition: k={k} was never scored; nothing to report")
        return

    silhouette = selected_metrics["silhouette_score"][0]
    logger.info(
        f"Partition at k={k}:\n"
        f"  Silhouette: {silhouette:.4f}\n"
        f"  Calinski-Harabasz: {selected_metrics['calinski_harabasz_score'][0]:.2f}\n"
        f"  Davies-Bouldin: {selected_metrics['davies_bouldin_score'][0]:.4f}\n"
        f"  Inertia: {selected_metrics['inertia'][0]:.2f}"
    )

    _log_abundance_silhouette(distance_matrix, geocode_cluster_df, k, silhouette)

    # A silhouette this low means the partition is not supported by the data,
    # whatever k was chosen. Say so rather than presenting it as a finding.
    if silhouette < MIN_SILHOUETTE_THRESHOLD:
        logger.warning(
            f"Silhouette for k={k} is {silhouette:.4f}, below "
            f"{MIN_SILHOUETTE_THRESHOLD}: the data show no substantial cluster "
            f"structure and this partition should not be read as a set of "
            f"well-separated bioregions."
        )


def optimize_num_clusters(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
    elbow_sensitivity: float = 1.0,
    weights: dict[str, float] | None = None,
    pinned_k: int | None = None,
) -> Tuple[int, pl.DataFrame]:
    """Score every k, select one, and report on it.

    The three steps composed, kept for callers that want all of it in one go.
    The notebook runs them separately, so that it can report on the cut it
    publishes rather than the one selected here.

    Returns:
        (selected k, the metrics for every k).
    """
    metrics_df = score_all_k(distance_matrix, geocode_cluster_df, weights=weights)
    optimal_k = select_k(
        metrics_df, pinned_k=pinned_k, elbow_sensitivity=elbow_sensitivity
    )
    report_partition(distance_matrix, geocode_cluster_df, metrics_df, optimal_k)
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
