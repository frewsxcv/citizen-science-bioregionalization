import networkx as nx
import polars as pl
from typing import Self

from src.data_container import DataContainer, assert_dataframe_schema
from src.dataframes.geocode import GeocodeDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame


class ClusterNeighborsDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "cluster": pl.UInt32(),
        "direct_neighbors": pl.List(pl.UInt32),
        "direct_and_indirect_neighbors": pl.List(pl.UInt32),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls,
        geocode_dataframe: GeocodeDataFrame,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
    ) -> Self:
        # Get unique clusters
        unique_clusters = geocode_cluster_dataframe.df["cluster"].unique()

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
        ) in geocode_dataframe.df.select(
            "geocode", "direct_neighbors", "direct_and_indirect_neighbors"
        ).iter_rows(
            named=False
        ):
            # Get the cluster of the current geocode
            current_cluster = geocode_cluster_dataframe.cluster_for_geocode(geocode)

            # For each direct neighbor, check if it's in a different cluster
            for neighbor in direct_neighbors:
                neighbor_cluster = geocode_cluster_dataframe.cluster_for_geocode(
                    neighbor
                )

                # If clusters are different, add to direct neighbors
                if current_cluster != neighbor_cluster:
                    direct_neighbors_map[current_cluster].add(neighbor_cluster)
                    all_neighbors_map[current_cluster].add(neighbor_cluster)

            # For each indirect neighbor, check if it's in a different cluster
            for neighbor in direct_and_indirect_neighbors:
                neighbor_cluster = geocode_cluster_dataframe.cluster_for_geocode(
                    neighbor
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
            ],
            schema=cls.SCHEMA,
        )
        return cls(df)

    def graph(self) -> nx.Graph:
        """Convert the DataFrame back to a NetworkX graph."""
        G: nx.Graph[str] = nx.Graph()

        # Add all clusters as nodes
        G.add_nodes_from(self.df["cluster"])

        # Add edges based on direct neighbors
        for row in self.df.iter_rows(named=True):
            cluster = row["cluster"]
            for neighbor in row["direct_neighbors"]:
                G.add_edge(cluster, neighbor, type="direct")

            # Add indirect neighbors that aren't already direct neighbors
            for neighbor in row["direct_and_indirect_neighbors"]:
                if neighbor not in row["direct_neighbors"]:
                    G.add_edge(cluster, neighbor, type="indirect")

        return G
