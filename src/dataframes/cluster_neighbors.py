import logging

import dataframely as dy

import bioregion_rs
from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.dataframes.geocode_neighbors import GeocodeNeighborsSchema

logger = logging.getLogger(__name__)


class ClusterNeighborsSchema(dy.Schema):
    cluster = dy.UInt32(nullable=False)
    direct_neighbors = dy.List(dy.UInt32(), nullable=False)
    direct_and_indirect_neighbors = dy.List(dy.UInt32(), nullable=False)


def build_cluster_neighbors_df(
    geocode_neighbors_df: dy.DataFrame[GeocodeNeighborsSchema],
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
) -> dy.DataFrame[ClusterNeighborsSchema]:
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
    return ClusterNeighborsSchema.validate(df)
