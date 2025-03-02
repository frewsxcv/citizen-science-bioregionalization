from typing import List, Optional, Self

import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.darwin_core import kingdom_enum
from src.types import ClusterId
from src.data_container import DataContainer


class ClusterTaxaStatisticsDataFrame(DataContainer):
    df: pl.DataFrame
    SCHEMA = {
        "cluster": pl.UInt32(),  # `null` if stats for all clusters
        "kingdom": kingdom_enum,
        "taxonRank": pl.String(),
        "scientificName": pl.String(),
        "count": pl.UInt32(),
        "average": pl.Float64(),  # average of taxa with `name` within `rank` and `cluster`
    }

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def iter_cluster_ids(self) -> list[ClusterId]:
        return self.df["cluster"].unique().to_list()

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_dataframe: GeocodeTaxaCountsDataFrame,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
        taxonomy_dataframe: TaxonomyDataFrame,
    ) -> Self:
        df = pl.DataFrame(schema=cls.SCHEMA)

        # Schema:
        #   - geocode: String
        #   - kingdom: Enum
        #   - taxonRank: String
        #   - scientificName: String
        #   - count: UInt32
        #   - phylum: String
        #   - class: String
        #   - order: String
        #   - family: String
        #   - genus: String
        #   - species: String
        joined = geocode_taxa_counts_dataframe.df.join(
            taxonomy_dataframe.df, on=["kingdom", "scientificName", "taxonRank"]
        )

        # Total count of all observations
        total_count = joined["count"].sum()

        # Calculate stats for all clusters
        df.vstack(
            joined.group_by(["kingdom", "scientificName", "taxonRank"])
            .agg(
                pl.col("count").sum().alias("count"),
                (pl.col("count").sum() / total_count).alias("average"),
            )
            .pipe(add_cluster_column, value=None)
            .select(cls.SCHEMA.keys()),  # Reorder columns
            in_place=True,
        )

        for (
            cluster,
            geocodees,
        ) in geocode_cluster_dataframe.iter_clusters_and_geocodees():
            total_count_in_cluster = joined.filter(pl.col("geocode").is_in(geocodees))[
                "count"
            ].sum()

            df.vstack(
                joined.filter(pl.col("geocode").is_in(geocodees))
                .group_by(["kingdom", "taxonRank", "scientificName"])
                .agg(
                    pl.col("count").sum().alias("count"),
                    (pl.col("count").sum() / total_count_in_cluster).alias("average"),
                )
                .pipe(add_cluster_column, value=cluster)
                .select(cls.SCHEMA.keys()),  # Reorder columns
                in_place=True,
            )

        return cls(df=df)

    def _get_count_by_rank_and_name(self, rank: str, name: str) -> int:
        counts = self.df.filter(
            (pl.col("rank") == rank) & (pl.col("name") == name)
        ).get_column("count")
        assert len(counts) <= 1
        sum = counts.sum()
        assert isinstance(sum, int)
        return sum

    def order_count(self, order: str) -> int:
        return self._get_count_by_rank_and_name("order", order)

    def class_count(self, class_name: str) -> int:
        return self._get_count_by_rank_and_name("class", class_name)

    def waterfowl_count(self) -> int:
        return (
            self.order_count("Anseriformes")
            + self.order_count("Charadriiformes")
            + self.order_count("Gaviiformes")
        )

    def aves_count(self) -> int:
        return self.class_count("Aves")


def add_cluster_column(df: pl.DataFrame, value: Optional[int]) -> pl.DataFrame:
    return df.with_columns(pl.lit(value).cast(pl.UInt32()).alias("cluster"))
