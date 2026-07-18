import polars as pl
import logging


import bioregion_rs

logger = logging.getLogger(__name__)

def build_cluster_neighbors_df(
    geocode_neighbors_df: pl.DataFrame,
    geocode_cluster_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build cluster neighbor relationships from geocode neighbor data.

    Determines which clusters are neighbors based on the geocode neighbor
    relationships within each cluster.

    Args:
        geocode_neighbors_df: DataFrame containing geocode neighbor information
        geocode_cluster_df: DataFrame mapping geocodes to clusters

    Returns:
        A validated DataFrame conforming to ClusterNeighborsSchema
    """
    logger.info("build_cluster_neighbors_df: Starting")

    df = bioregion_rs.build_cluster_neighbors(
        geocode_neighbors_df.select(
            "geocode", "direct_neighbors", "direct_and_indirect_neighbors"
        ),
        geocode_cluster_df.select("geocode", "cluster"),
    )
    return df
