from typing import List, Self
import polars as pl
from src.cluster_color_builder import ClusterColorBuilder
from src.types import ClusterId


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
        colors = [
            # TODO: Migrate this to the better color builder method
            ClusterColorBuilder.random()
            for _ in clusters
        ]
        return cls(
            df=pl.DataFrame(
                data={
                    "cluster": clusters,
                    "color": colors,
                },
                schema=cls.SCHEMA,
            )
        )
