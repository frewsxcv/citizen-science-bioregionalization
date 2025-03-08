from typing import Dict, List
import polars as pl
from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.types import ClusterId
from src.data_container import DataContainer
import seaborn as sns
import networkx as nx


class ClusterColorDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "cluster": pl.UInt32,
        "color": pl.Utf8,
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert df.schema == self.SCHEMA
        self.df = df

    def get_color_for_cluster(self, cluster: ClusterId) -> str:
        return self.df.filter(pl.col("cluster") == cluster)["color"].to_list()[0]

    @classmethod
    def build(
        cls,
        cluster_neighbors_dataframe: ClusterNeighborsDataFrame,
    ) -> "ClusterColorDataFrame":
        G = cluster_neighbors_dataframe.graph()

        colors = nx.coloring.greedy_color(G)

        color_palette = sns.color_palette("hsv", max(colors.values()) + 1).as_hex()
        rows = [
            {"cluster": cluster, "color": color_palette[color_index]}
            for cluster, color_index in colors.items()
        ]

        df = pl.DataFrame(rows).with_columns([
            pl.col("cluster").cast(pl.UInt32),
            pl.col("color").cast(pl.Utf8),
        ])

        return cls(df)

    def to_dict(self) -> Dict[ClusterId, str]:
        return {x: self.get_color_for_cluster(x) for x in self.df["cluster"]}


def rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255))
