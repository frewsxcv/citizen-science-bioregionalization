from typing import Iterator, List, NewType, Tuple
import polars as pl
from src.cluster_color_builder import ClusterColorBuilder
from src.cluster_stats import Stats
from src.darwin_core_aggregations import DarwinCoreAggregations
from src.types import Geohash, ClusterId

class GeohashClusterDataFrame(pl.DataFrame):
    SCHEMA = {
        "geohash": pl.String,
        "cluster": pl.UInt32,
    }

    def __init__(
        self,
        ordered_seen_geohash: List[Geohash],
        clusters: List[ClusterId],
    ) -> None:
        assert len(ordered_seen_geohash) == len(clusters)
        super().__init__(
            data={
                "geohash": ordered_seen_geohash,
                "cluster": clusters,
            },
            schema=self.SCHEMA,
        )

    def iter_clusters_and_geohashes(
        self,
    ) -> Iterator[Tuple[ClusterId, List[Geohash]]]:
        for row in (self.group_by("cluster").all().sort("cluster")).iter_rows(
            named=True
        ):
            yield row["cluster"], row["geohash"]


    def geohashes_for_cluster(self, cluster: ClusterId) -> List[Geohash]:
        return self.filter(pl.col("cluster") == cluster)["geohash"].to_list()


    def determine_color_for_cluster(
        self,
        darwin_core_aggregations: DarwinCoreAggregations,
        cluster: ClusterId,
    ) -> str:
        stats = Stats.build(
            darwin_core_aggregations,
            geohash_filter=self.geohashes_for_cluster(cluster),
        )
        return ClusterColorBuilder.determine_color_for_cluster(stats)


    def num_clusters(self) -> int:
        num = self["cluster"].max()
        assert isinstance(num, int)
        return num
