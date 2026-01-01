import dataframely as dy
import networkx as nx
import polars as pl

from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import GeocodeClusterSchema, cluster_for_geocode


class ClusterNeighborsSchema(dy.Schema):
    cluster = dy.UInt32(nullable=False)
    direct_neighbors = dy.List(dy.UInt32(), nullable=False)
    direct_and_indirect_neighbors = dy.List(dy.UInt32(), nullable=False)


def build_cluster_neighbors_df(
    geocode_df: dy.LazyFrame[GeocodeNoEdgesSchema],
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
) -> dy.DataFrame[ClusterNeighborsSchema]:
    """Build cluster neighbor relationships from geocode data.

    Determines which clusters are neighbors based on the geocode neighbor
    relationships within each cluster.

    Args:
        geocode_df: LazyFrame containing geocode and neighbor information
        geocode_cluster_df: DataFrame mapping geocodes to clusters

    Returns:
        A validated DataFrame conforming to ClusterNeighborsSchema
    """
    # Collect the LazyFrame once at the start
    geocode_collected = geocode_df.collect()

    # Get unique clusters
    unique_clusters = geocode_cluster_df["cluster"].unique()

    # Initialize a dictionary to store the direct and indirect neighbors
    direct_neighbors_map: dict[int, set[int]] = {
        cluster: set() for cluster in unique_clusters
    }
    all_neighbors_map: dict[int, set[int]] = {
        cluster: set() for cluster in unique_clusters
    }

    # For each geocode, find neighbors in different clusters
    for (
        geocode,
        direct_neighbors,
        direct_and_indirect_neighbors,
    ) in geocode_collected.select(
        "geocode", "direct_neighbors", "direct_and_indirect_neighbors"
    ).iter_rows(named=False):
        # Get the cluster of the current geocode
        current_cluster = cluster_for_geocode(geocode_cluster_df, geocode)

        # For each direct neighbor, check if it's in a different cluster
        for neighbor in direct_neighbors:
            neighbor_cluster = cluster_for_geocode(geocode_cluster_df, neighbor)

            # If clusters are different, add to direct neighbors
            if current_cluster != neighbor_cluster:
                direct_neighbors_map[current_cluster].add(neighbor_cluster)
                all_neighbors_map[current_cluster].add(neighbor_cluster)

        # For each indirect neighbor, check if it's in a different cluster
        for neighbor in direct_and_indirect_neighbors:
            neighbor_cluster = cluster_for_geocode(geocode_cluster_df, neighbor)

            # If clusters are different, add to all neighbors
            if current_cluster != neighbor_cluster:
                all_neighbors_map[current_cluster].add(neighbor_cluster)

    df = pl.DataFrame(
        [
            {
                "cluster": cluster,
                "direct_neighbors": list(direct_neighbors_map[cluster]),
                "direct_and_indirect_neighbors": list(all_neighbors_map[cluster]),
            }
            for cluster in unique_clusters
        ]
    ).with_columns(
        pl.col("cluster").cast(pl.UInt32),
        pl.col("direct_neighbors").cast(pl.List(pl.UInt32)),
        pl.col("direct_and_indirect_neighbors").cast(pl.List(pl.UInt32)),
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
