from typing import Iterator, List, NewType, Tuple
import polars as pl
from src.cluster import ClusterId
from src.cluster_color_builder import ClusterColorBuilder
from src.cluster_stats import Stats
from src.darwin_core_aggregations import DarwinCoreAggregations
from src.geohash import Geohash

ClusterIndex = NewType("ClusterIndex", pl.DataFrame)
"""
Schema:
- `geohash`: `str`
- `cluster`: `int`
"""


def build(
    ordered_seen_geohash: List[Geohash], clusters: List[ClusterId]
) -> ClusterIndex:
    dataframe = pl.DataFrame(
        data={
            "geohash": ordered_seen_geohash,
            "cluster": clusters,
        },
        schema={"geohash": pl.String, "cluster": pl.UInt32},
    )
    return ClusterIndex(dataframe)


def geohashes_for_cluster(clusters: ClusterIndex, cluster: ClusterId) -> List[Geohash]:
    return clusters.filter(pl.col("cluster") == cluster)["geohash"].to_list()


def determine_color_for_cluster(
    clusters: ClusterIndex,
    cluster: ClusterId,
    darwin_core_aggregations: DarwinCoreAggregations,
) -> str:
    stats = Stats.build(
        darwin_core_aggregations,
        geohash_filter=geohashes_for_cluster(clusters, cluster),
    )
    return ClusterColorBuilder.determine_color_for_cluster(stats)


def iter_clusters_and_geohashes(
    clusters: ClusterIndex,
) -> Iterator[Tuple[ClusterId, List[Geohash]]]:
    for row in (clusters.group_by("cluster").all().sort("cluster")).iter_rows(
        named=True
    ):
        yield row["cluster"], row["geohash"]


def num_clusters(self) -> int:
    num = self.dataframe["cluster"].max()
    assert isinstance(num, int)
    return num
