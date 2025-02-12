from typing import List, Optional, Self

import polars as pl

from src.dataframes.geohash_cluster import GeohashClusterDataFrame
from src.dataframes.geohash_species_counts import GeohashSpeciesCountsDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.geohash import Geohash
from src.darwin_core import TaxonRank, kingdom_enum


class Stats:
    df: pl.DataFrame
    SCHEMA = {
        "cluster": pl.UInt32(),  # `null` if stats for all clusters
        "kingdom": kingdom_enum,
        "rank": pl.Enum(TaxonRank),
        "name": pl.String(),
        "count": pl.UInt32(),
        "average": pl.Float64(),  # average of taxa with `name` at `rank` within `cluster`
    }

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    @classmethod
    def build(
        cls,
        geohash_taxa_counts_dataframe: GeohashSpeciesCountsDataFrame,
        geohash_cluster_dataframe: GeohashClusterDataFrame,
        taxonomy_dataframe: TaxonomyDataFrame,
    ) -> Self:
        df = pl.DataFrame(schema=Stats.SCHEMA)

        # Schema:
        #   - geohash: String
        #   - kingdom: Enum
        #   - species: String
        #   - count: UInt32
        #   - phylum: String
        #   - class: String
        #   - order: String
        #   - family: String
        #   - genus: String
        joined = geohash_taxa_counts_dataframe.filtered().join(
            taxonomy_dataframe.df, on=["kingdom", "species"]
        )

        for rank in TaxonRank:
            # Calculate stats for all clusters
            df.vstack(
                joined.group_by(["kingdom", rank.value])
                .agg(
                    pl.col("count").sum(),
                    (pl.col("count").sum() / joined["count"].sum()).alias("average"),
                )
                .with_columns(
                    pl.lit(None).alias("cluster"),
                    pl.lit(rank.value).cast(pl.Enum(TaxonRank)).alias("rank"),
                )
                .rename({rank.value: "name"})
                .select(Stats.SCHEMA.keys()),
                in_place=True,
            )

            # Calculate stats for each cluster
            # df.vstack(
            #     joined.group_by(["kingdom", rank.value, "cluster"])
            #     .agg(
            #         pl.col("count").sum(),
            #         (pl.col("count").sum() / joined["count"].sum()).alias("average"),
            #     )
            # )

        import pdb
        pdb.set_trace()

        return cls(df=taxa)

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
