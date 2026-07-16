import logging

import dataframely as dy

import bioregion_rs
from src.dataframes.cluster_neighbors import ClusterNeighborsSchema
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema

logger = logging.getLogger(__name__)


class ClusterSignificantDifferencesSchema(dy.Schema):
    """
    A dataframe that contains the significant differences between clusters.
    """

    P_VALUE_THRESHOLD = 0.05
    MIN_COUNT_THRESHOLD = 5

    cluster = dy.UInt32(nullable=False)
    taxonId = dy.UInt32(nullable=False)
    p_value = dy.Float64(nullable=False)
    log2_fold_change = dy.Float64(nullable=False)
    cluster_count = dy.UInt32(nullable=False)
    neighbor_count = dy.UInt32(nullable=False)
    high_log2_high_count_score = dy.Float64(nullable=False)
    low_log2_high_count_score = dy.Float64(nullable=False)


def build_cluster_significant_differences_df(
    all_stats: dy.DataFrame[ClusterTaxaStatisticsSchema],
    cluster_neighbors: dy.LazyFrame[ClusterNeighborsSchema],
) -> dy.DataFrame[ClusterSignificantDifferencesSchema]:
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

    return ClusterSignificantDifferencesSchema.validate(df)
