from enum import Enum
import logging
import polars as pl
from typing import List, NamedTuple, Self
from src.darwin_core import read_rows, kingdom_enum
from src.geohash import Geohash, build_geohash_series_lazy
from contexttimer import Timer


logger = logging.getLogger(__name__)


class TaxonRank(Enum):
    """
    Each of these values is also the name of a Darwin Core column
    """
    phylum = "phylum"
    class_ = "class"
    order = "order"
    family = "family"
    genus = "genus"
    species = "species"


class GeohashTaxaCountsDataFrame:
    df: pl.DataFrame

    SCHEMA = {
        "geohash": pl.String(),
        "kingdom": kingdom_enum,
        "rank": pl.Enum(TaxonRank),
        "name": pl.String(),
        "count": pl.UInt32(),
    }

    def __init__(self, input_file: str, geohash_precision: int):
        unfiltered_taxon_counts = pl.DataFrame(
            schema=self.SCHEMA,
        )

        # Will this work for eBird?
        # geohash_to_taxon_id_to_user_to_count: DefaultDict[
        #     Geohash, DefaultDict[TaxonId, Counter[str]]
        # ] = defaultdict(lambda: defaultdict(Counter))

        with Timer(output=logger.info, prefix="Reading rows"):
            for read_dataframe in read_rows(
                input_file,
                columns=[
                    "decimalLatitude",
                    "decimalLongitude",
                    "recordedBy",
                    "kingdom",
                    *map(lambda rank: rank.value, TaxonRank),
                ],
            ):
                # Temporary: Filter out orders that are not Diptera
                read_dataframe = read_dataframe.filter(pl.col("order") == "Diptera")

                for variant in TaxonRank:
                    # `aggregated` is a dataframe that looks like this:
                    #
                    # +------------+----------+---------+--------------+-------+
                    # | geohash    | kingdom  | rank    | name         | count |
                    # +------------+----------+---------+--------------+-------+
                    # | u4pruydqqvj| Animalia | species | Panthera leo | 42    |
                    # +------------+----------+---------+--------------+-------+
                    aggregated = (
                        read_dataframe.lazy()
                        .pipe(
                            build_geohash_series_lazy,
                            lat_col=pl.col("decimalLatitude"),
                            lon_col=pl.col("decimalLongitude"),
                            precision=geohash_precision,
                        )
                        .filter(
                            pl.col(variant.value).is_not_null()
                        )  # TODO: DONT DO THIS. THIS LOSES DATA
                        .group_by(["geohash", "kingdom", variant.value])
                        .agg(
                            pl.len().alias("count"),
                        )
                        .rename({variant.value: "name"})
                        .with_columns(
                            pl.lit(variant, dtype=pl.Enum(TaxonRank)).alias("rank"),
                        )
                        .select(["geohash", "kingdom", "rank", "name", "count"])
                        .collect()
                    )
                    unfiltered_taxon_counts.vstack(
                        aggregated,
                        in_place=True,
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

        self.df = (
            unfiltered_taxon_counts.lazy()
            .group_by(["geohash", "kingdom", "rank", "name"])
            .agg(pl.col("count").sum())
            .sort(by="geohash")
            .collect()
        )

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
            self.filtered().select("geohash")
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
