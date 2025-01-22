from typing import Iterator, List, NewType, Tuple
import polars as pl
from src.cluster_color_builder import ClusterColorBuilder
from src.cluster_stats import Stats
from src.dataframes.geohash_taxa_counts import GeohashTaxaCountsDataFrame
from src.types import Geohash, ClusterId

class GeohashClusterDataFrame:
    df: pl.DataFrame

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
        self.df = pl.DataFrame(
            data={
                "geohash": ordered_seen_geohash,
                "cluster": clusters,
            },
            schema=self.SCHEMA,
        )

    def cluster_ids(self) -> List[ClusterId]:
        return self.df["cluster"].unique().to_list()

    def iter_clusters_and_geohashes(
        self,
    ) -> Iterator[Tuple[ClusterId, List[Geohash]]]:
        for row in (self.df.group_by("cluster").all().sort("cluster")).iter_rows(
            named=True
        ):
            yield row["cluster"], row["geohash"]


    def geohashes_for_cluster(self, cluster: ClusterId) -> List[Geohash]:
        return self.df.filter(pl.col("cluster") == cluster)["geohash"].to_list()


    def determine_color_for_cluster(
        self,
        geohash_taxa_counts_dataframe: GeohashTaxaCountsDataFrame,
        cluster: ClusterId,
    ) -> str:
        stats = Stats.build(
            geohash_taxa_counts_dataframe,
            geohash_filter=self.geohashes_for_cluster(cluster),
        )
        return ClusterColorBuilder.determine_color_for_cluster(stats)


    def num_clusters(self) -> int:
        num = self.df["cluster"].max()
        assert isinstance(num, int)
        return num
