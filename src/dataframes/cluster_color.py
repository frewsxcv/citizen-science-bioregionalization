from typing import Dict, List, Self
import polars as pl
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.types import ClusterId
from src.data_container import DataContainer
import seaborn as sns


class ClusterColorDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "cluster": pl.UInt32,
        "color": pl.String,
    }

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def get_color_for_cluster(self, cluster: ClusterId) -> str:
        return self.df.filter(pl.col("cluster") == cluster)["color"].to_list()[0]

    @classmethod
    def build(cls, geocode_cluster_dataframe: GeocodeClusterDataFrame) -> Self:
        clusters = geocode_cluster_dataframe.cluster_ids()
        palette = sns.color_palette("Spectral", n_colors=len(clusters))
        colors = [rgb_to_hex(*palette[i]) for i in range(len(clusters))]
        return cls(
            df=pl.DataFrame(
                data={
                    "cluster": clusters,
                    "color": colors,
                },
                schema=cls.SCHEMA,
            )
        )

    def to_dict(self) -> Dict[ClusterId, str]:
        return {x: self.get_color_for_cluster(x) for x in self.df["cluster"]}


def rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255))
