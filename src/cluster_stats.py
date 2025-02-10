from typing import List, NamedTuple, Optional, Self

import polars as pl

from src.dataframes.geohash_species_counts import GeohashSpeciesCountsDataFrame
from src.geohash import Geohash


class Stats(NamedTuple):
    taxon: pl.LazyFrame
    """
    Schema:
    - `name`: `str` species name
    - `count`: `int`
    - `average`: `float`
    """

    order_counts: pl.LazyFrame
    """
    Schema:
    - `order`: `str`
    - `count`: `int`
    """

    class_counts: pl.LazyFrame
    """
    Schema:
    - `class`: `str`
    - `count`: `int`
    """

    @classmethod
    def build(
        cls,
        geohash_taxa_counts_dataframe: GeohashSpeciesCountsDataFrame,
        geohash_filter: Optional[List[Geohash]] = None,
    ) -> Self:
        geohash_filter_clause = (
            pl.col("geohash").is_in(geohash_filter)
            if geohash_filter
            else pl.lit(True)
        )

        # Schema:
        # - `taxonKey`: `int`
        # - `count`: `int`
        taxon_counts: pl.LazyFrame = (
            geohash_taxa_counts_dataframe.filtered().lazy()
            .filter(geohash_filter_clause)
            .select(["kingdom", "species", "count"])
            .group_by("kingdom", "species")
            .agg(pl.col("count").sum())
        )

        # Total observation count all filtered geohashes
        total_count: int = taxon_counts.select("count").sum().collect()["count"].item()

        # Schema:
        # - `kingdom`: `str`
        # - `species`: `str`
        # - `count`: `int`
        # - `average`: `float`
        taxon: pl.LazyFrame = taxon_counts.with_columns(
            (pl.col("count") / total_count).alias("average")
        )

        order_counts = (
            geohash_taxa_counts_dataframe.df.lazy()
            .filter(
                geohash_filter_clause,
                pl.col("rank") == "order",
            )
            .group_by("name")
            .agg(pl.col("count").sum())
        )

        class_counts = (
            geohash_taxa_counts_dataframe.df.lazy()
            .filter(geohash_filter_clause, pl.col("rank") == "class")
            .group_by("name")
            .agg(pl.col("count").sum())
        )

        return cls(
            taxon=taxon,
            order_counts=order_counts,
            class_counts=class_counts,
        )

    def order_count(self, order: str) -> int:
        counts = (
            self.order_counts.filter(pl.col("name") == order)
            .collect()
            .get_column("count")
        )
        assert len(counts) <= 1
        sum = counts.sum()
        assert isinstance(sum, int)
        return sum

    def class_count(self, class_name: str) -> int:
        counts = (
            self.class_counts.filter(pl.col("name") == class_name)
            .collect()
            .get_column("count")
        )
        assert len(counts) <= 1
        sum = counts.sum()
        assert isinstance(sum, int)
        return sum

    def waterfowl_count(self) -> int:
        return (
            self.order_count("Anseriformes")
            + self.order_count("Charadriiformes")
            + self.order_count("Gaviiformes")
        )

    def aves_count(self) -> int:
        return self.class_count("Aves")
