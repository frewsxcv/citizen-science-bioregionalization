from typing import List, NamedTuple, Optional, Self

import polars as pl

from src.darwin_core_aggregations import DarwinCoreAggregations
from src.geohash import Geohash


class Stats(NamedTuple):
    geohashes: List[Geohash]

    taxon: pl.LazyFrame
    """
    Schema:
    - `taxonKey`: `int`
    - `count`: `int`
    - `average`: `float`
    """

    order_counts: pl.LazyFrame
    """
    Schema:
    - `order`: `str`
    - `count`: `int`
    """

    @classmethod
    def build(
        cls,
        darwin_core_aggregations: DarwinCoreAggregations,
        geohash_filter: Optional[List[Geohash]] = None,
    ) -> Self:
        geohashes = (
            darwin_core_aggregations.ordered_geohashes()
            if geohash_filter is None
            else [
                g
                for g in darwin_core_aggregations.ordered_geohashes()
                if g in geohash_filter
            ]
        )

        # Schema:
        # - `taxonKey`: `int`
        # - `count`: `int`
        taxon_counts: pl.LazyFrame = (
            darwin_core_aggregations.taxon_counts.lazy()
            .filter(pl.col("geohash").is_in(geohashes))
            .select(["taxonKey", "count"])
            .group_by("taxonKey")
            .agg(pl.col("count").sum())
        )

        # Total observation count all filtered geohashes
        total_count: int = taxon_counts.select("count").sum().collect()["count"].item()

        # Schema:
        # - `taxonKey`: `int`
        # - `count`: `int`
        # - `average`: `float`
        taxon: pl.LazyFrame = taxon_counts.with_columns(
            (pl.col("count") / total_count).alias("average")
        )

        order_counts = (
            darwin_core_aggregations.taxon_counts_2.lazy()
            .filter(
                pl.col("geohash").is_in(geohashes),
                pl.col("rank") == "order",
            )
            .group_by("name")
            .agg(pl.col("count").sum())
        )

        return cls(
            geohashes=geohashes,
            taxon=taxon,
            order_counts=order_counts,
        )

    def order_count(self, order: str) -> int:
        return (
            self.order_counts.filter(pl.col("name") == order)
            .collect()
            .get_column("count")
            .item()
        )
