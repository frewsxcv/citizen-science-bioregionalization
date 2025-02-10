from typing import Iterator, List, Self, Tuple
import logging
import polars as pl
from src.dataframes.geohash_species_counts import GeohashSpeciesCountsDataFrame
from src.types import Geohash, ClusterId
from scipy.cluster.hierarchy import linkage, fcluster
from src.distance_matrix import DistanceMatrix

logger = logging.getLogger(__name__)


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

    @classmethod
    def build(
        cls,
        geohash_taxa_counts_dataframe: GeohashSpeciesCountsDataFrame,
        distance_matrix: DistanceMatrix,
        num_clusters: int,
    ) -> Self:
        ordered_seen_geohash = geohash_taxa_counts_dataframe.ordered_geohashes()
        Z = linkage(distance_matrix.condensed(), "ward")
        clusters = list(map(int, fcluster(Z, t=num_clusters, criterion="maxclust")))
        return cls.from_lists(ordered_seen_geohash, clusters)

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
