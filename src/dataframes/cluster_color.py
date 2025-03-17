from typing import Dict, List
import polars as pl
from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.cluster_boundary import ClusterBoundaryDataFrame
from src.types import ClusterId
from src.data_container import DataContainer, assert_dataframe_schema
import seaborn as sns
import networkx as nx


class ClusterColorDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "cluster": pl.UInt32(),
        "color": pl.Utf8(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    def get_color_for_cluster(self, cluster: ClusterId) -> str:
        return self.df.filter(pl.col("cluster") == cluster)["color"].to_list()[0]

    @classmethod
    def build(
        cls,
        cluster_neighbors_dataframe: ClusterNeighborsDataFrame,
        cluster_boundary_dataframe: ClusterBoundaryDataFrame,
        ocean_threshold: float = 0.90,
    ) -> "ClusterColorDataFrame":
        # Import here to avoid circular imports
        from src.geojson import find_ocean_clusters

        G = cluster_neighbors_dataframe.graph()

        # Find ocean clusters
        ocean_clusters = set(find_ocean_clusters(
            cluster_boundary_dataframe, threshold=ocean_threshold
        ))

        # Find land clusters
        land_clusters = set(G.nodes()) - ocean_clusters

        # Use NetworkX to color the entire graph - this ensures adjacent nodes have different colors
        color_indices = nx.coloring.greedy_color(G)
        ocean_color_indices = {cluster: color_indices[cluster] for cluster in ocean_clusters}
        land_color_indices = {cluster: color_indices[cluster] for cluster in land_clusters}

        # Determine how many unique colors needed for each group
        num_ocean_colors = len(set(ocean_color_indices.values()))
        num_land_colors = len(set(land_color_indices.values()))

        # Generate color palettes with the exact sizes needed
        ocean_palette = sns.color_palette("Blues", num_ocean_colors).as_hex()
        land_palette = sns.color_palette("YlOrRd", num_land_colors).as_hex()

        # Create mapping from color indices to colors
        ocean_palette_map = dict(zip(sorted(set(ocean_color_indices.values())), ocean_palette))
        land_palette_map = dict(zip(sorted(set(land_color_indices.values())), land_palette))

        # Map color indices to actual colors
        rows = []
        for cluster, color_index in color_indices.items():
            if cluster in ocean_clusters:
                color = ocean_palette_map[color_index]
            else:
                color = land_palette_map[color_index]

            rows.append({"cluster": cluster, "color": color})

        return cls(pl.DataFrame(rows, schema=cls.SCHEMA))

    def to_dict(self) -> Dict[ClusterId, str]:
        return {x: self.get_color_for_cluster(x) for x in self.df["cluster"]}
