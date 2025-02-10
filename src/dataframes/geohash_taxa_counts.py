import logging
import polars as pl
from typing import List
from src.darwin_core import kingdom_enum, TaxonRank
from src.geohash import Geohash, build_geohash_series_lazy
from contexttimer import Timer

from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame


logger = logging.getLogger(__name__)


class GeohashTaxaCountsDataFrame:
    df: pl.DataFrame

    SCHEMA = {
        "geohash": pl.String(),
        "kingdom": kingdom_enum,
        # "species": pl.String(),
        "rank": pl.Enum(TaxonRank),
        "name": pl.String(),
        "count": pl.UInt32(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame,
        geohash_precision: int,
    ):
        # Will this work for eBird?
        # geohash_to_taxon_id_to_user_to_count: DefaultDict[
        #     Geohash, DefaultDict[TaxonId, Counter[str]]
        # ] = defaultdict(lambda: defaultdict(Counter))

        with Timer(output=logger.info, prefix="Reading rows"):
            # Temporary: Filter out orders that are not Diptera
            # read_dataframe = read_dataframe.filter(pl.col("order") == "Diptera")

            # `aggregated` is a dataframe that looks like this:
            #
            # +------------+----------+---------+--------------+-------+
            # | geohash    | kingdom  | rank    | name         | count |
            # +------------+----------+---------+--------------+-------+
            # | u4pruydqqvj| Animalia | species | Panthera leo | 42    |
            # +------------+----------+---------+--------------+-------+
            aggregated = (
                darwin_core_csv_lazy_frame.lf.pipe(
                    build_geohash_series_lazy,
                    lat_col=pl.col("decimalLatitude"),
                    lon_col=pl.col("decimalLongitude"),
                    precision=geohash_precision,
                )
                .filter(
                    pl.col("species").is_not_null()
                )  # TODO: DONT DO THIS. THIS LOSES DATA
                .group_by(["geohash", "kingdom", "species"])
                .agg(pl.len().alias("count"))
                .rename({"species": "name"})
                .with_columns(
                    pl.lit("species", dtype=pl.Enum(TaxonRank)).alias("rank"),
                )
                .select(["geohash", "kingdom", "rank", "name", "count"])
                .sort(by="geohash")
                .collect()
            )

            # for row in dataframe_with_geohash.collect(streaming=True).iter_rows(
            #     named=True
            # ):
            #     geohash_to_taxon_id_to_user_to_count[row["geohash"]][
            #         row["taxonKey"]
            #     ][row["recordedBy"]] += 1
            #     # If the observer has seen the taxon more than 5 times, skip it
            #     if (
            #         geohash_to_taxon_id_to_user_to_count[row["geohash"]][
            #             row["taxonKey"]
            #         ][row["recordedBy"]]
            #         > 5
            #     ):
            #         continue

            return cls(aggregated)

    def filtered(self) -> pl.DataFrame:
        return (
            self.df.lazy()
            .filter(pl.col("rank") == "species", pl.col("name").is_not_null())
            .select("geohash", "kingdom", "name", "count")
            .rename({"name": "species"})
            .sort(by="geohash")
            .collect()
        )

    # @functools.cache
    def ordered_geohashes(self) -> List[Geohash]:
        return (
            self.filtered()
            .select("geohash")
            .unique()
            .sort(by="geohash")
            .get_column("geohash")
            .to_list()
        )

    # # @functools.cache
    # def ordered_taxon_keys(self) -> List[int]:
    #     return (
    #         self.taxon_counts.select("taxonKey")
    #         .unique()
    #         .sort(by="taxonKey")
    #         .get_column("taxonKey")
    #         .to_list()
    #     )
