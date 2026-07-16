import logging

import dataframely as dy
import networkx as nx

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


def to_graph(
    cluster_neighbors_lf: dy.LazyFrame[ClusterNeighborsSchema],
) -> nx.Graph:
    """Convert the DataFrame back to a NetworkX graph."""
    G: nx.Graph[str] = nx.Graph()

    cluster_neighbors_df = cluster_neighbors_lf.collect(engine="streaming")

    # Add all clusters as nodes
    G.add_nodes_from(cluster_neighbors_df["cluster"])

    # Add edges based on direct neighbors
    for row in cluster_neighbors_df.iter_rows(named=True):
        cluster = row["cluster"]
        for neighbor in row["direct_neighbors"]:
            G.add_edge(cluster, neighbor, type="direct")

        # Add indirect neighbors that aren't already direct neighbors
        for neighbor in row["direct_and_indirect_neighbors"]:
            if neighbor not in row["direct_neighbors"]:
                G.add_edge(cluster, neighbor, type="indirect")

    return G
