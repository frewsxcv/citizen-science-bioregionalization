import dataframely as dy
import networkx as nx
import polars as pl

from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import GeocodeClusterSchema, cluster_for_geocode


class ClusterNeighborsSchema(dy.Schema):
    cluster = dy.UInt32(nullable=False)
    direct_neighbors = dy.List(dy.UInt32(), nullable=False)
    direct_and_indirect_neighbors = dy.List(dy.UInt32(), nullable=False)

    @classmethod
    def build(
        cls,
        geocode_dataframe: dy.LazyFrame[GeocodeNoEdgesSchema],
        geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
    ) -> dy.DataFrame["ClusterNeighborsSchema"]:
        # Collect the LazyFrame once at the start (handle both LazyFrame and DataFrame)
        geocode_df = (
            geocode_dataframe.collect()
            if isinstance(geocode_dataframe, pl.LazyFrame)
            else geocode_dataframe
        )

        # Get unique clusters
        unique_clusters = geocode_cluster_dataframe["cluster"].unique()

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
        ) in geocode_df.select(
            "geocode", "direct_neighbors", "direct_and_indirect_neighbors"
        ).iter_rows(named=False):
            # Get the cluster of the current geocode
            current_cluster = cluster_for_geocode(geocode_cluster_dataframe, geocode)

            # For each direct neighbor, check if it's in a different cluster
            for neighbor in direct_neighbors:
                neighbor_cluster = cluster_for_geocode(
                    geocode_cluster_dataframe, neighbor
                )

                # If clusters are different, add to direct neighbors
                if current_cluster != neighbor_cluster:
                    direct_neighbors_map[current_cluster].add(neighbor_cluster)
                    all_neighbors_map[current_cluster].add(neighbor_cluster)

            # For each indirect neighbor, check if it's in a different cluster
            for neighbor in direct_and_indirect_neighbors:
                neighbor_cluster = cluster_for_geocode(
                    geocode_cluster_dataframe, neighbor
                )

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
        return cls.validate(df)


def to_graph(
    cluster_neighbors_lazyframe: dy.LazyFrame[ClusterNeighborsSchema],
) -> nx.Graph:
    """Convert the DataFrame back to a NetworkX graph."""
    G: nx.Graph[str] = nx.Graph()

    cluster_neighbors_dataframe = cluster_neighbors_lazyframe.collect(
        engine="streaming"
    )

    # Add all clusters as nodes
    G.add_nodes_from(cluster_neighbors_dataframe["cluster"])

    # Add edges based on direct neighbors
    for row in cluster_neighbors_dataframe.iter_rows(named=True):
        cluster = row["cluster"]
        for neighbor in row["direct_neighbors"]:
            G.add_edge(cluster, neighbor, type="direct")

        # Add indirect neighbors that aren't already direct neighbors
        for neighbor in row["direct_and_indirect_neighbors"]:
            if neighbor not in row["direct_neighbors"]:
                G.add_edge(cluster, neighbor, type="indirect")

    return G
