from typing import List
import polars as pl
from src.cluster_color_builder import ClusterColorBuilder
from src.types import ClusterId


class ClusterColorDataFrame:
    df: pl.DataFrame

    SCHEMA = {
        "cluster": pl.UInt32,
        "color": pl.String,
    }

    def __init__(self, clusters: List[ClusterId]):
        colors = [
            # TODO: Migrate this to the better color builder method
            ClusterColorBuilder.random()
            for _ in clusters
        ]
        self.df = pl.DataFrame(
            data={
                "cluster": clusters,
                "color": colors,
            },
            schema=self.SCHEMA,
        )
