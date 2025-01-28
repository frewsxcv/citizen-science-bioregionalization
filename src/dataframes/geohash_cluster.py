from typing import Iterator, List, Self, Tuple
import polars as pl
from src.cluster_stats import Stats
from src.dataframes.geohash_taxa_counts import GeohashTaxaCountsDataFrame
from src.types import Geohash, ClusterId


class GeohashClusterDataFrame:
    df: pl.DataFrame

    SCHEMA = {
        "geohash": pl.String,
        "cluster": pl.UInt32,
    }

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    @classmethod
    def from_lists(
        cls,
        geohashes: List[Geohash],
        clusters: List[ClusterId],
    ) -> Self:
        assert len(geohashes) == len(clusters)
        return cls(
            df=pl.DataFrame(
                data={
                    "geohash": geohashes,
                    "cluster": clusters,
                },
                schema=cls.SCHEMA,
            )
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

    def cluster_for_geohash(self, geohash: Geohash) -> ClusterId:
        return self.df.filter(pl.col("geohash") == geohash)["cluster"].to_list()[0]

    def geohashes_for_cluster(self, cluster: ClusterId) -> List[Geohash]:
        return self.df.filter(pl.col("cluster") == cluster)["geohash"].to_list()

    def num_clusters(self) -> int:
        num = self.df["cluster"].max()
        assert isinstance(num, int)
        return num
