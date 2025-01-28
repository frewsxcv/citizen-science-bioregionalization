from typing import List, Self
import polars as pl
from src.types import ClusterId
import seaborn as sns

class ClusterColorDataFrame:
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
    def from_clusters(cls, clusters: List[ClusterId]) -> Self:
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

def rgb_to_hex(r: float, g: float, b: float) -> str:
  return "#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255))
