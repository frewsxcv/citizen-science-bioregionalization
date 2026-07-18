import polars as pl
import logging


import bioregion_rs

logger = logging.getLogger(__name__)

def build_cluster_significant_differences_df(
    all_stats: pl.DataFrame,
    cluster_neighbors: pl.LazyFrame,
) -> pl.DataFrame:
    """Build a ClusterSignificantDifferencesSchema DataFrame.

    Identifies taxa that significantly differ between clusters and their neighbors
    using Fisher's exact test.

    Args:
        all_stats: DataFrame of taxa statistics per cluster
        cluster_neighbors: LazyFrame of cluster neighbor relationships

    Returns:
        A validated DataFrame conforming to ClusterSignificantDifferencesSchema
    """
    logger.info("build_cluster_significant_differences_df: Starting")

    cluster_neighbors_df = cluster_neighbors.collect(engine="streaming")

    df = bioregion_rs.build_cluster_significant_differences(
        all_stats, cluster_neighbors_df
    )

    logger.info(
        f"build_cluster_significant_differences_df: Output has {df.height} rows"
    )

    return df
