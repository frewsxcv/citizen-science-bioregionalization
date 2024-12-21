from enum import Enum
import logging
import polars as pl
from typing import List, NamedTuple, Self
from src.darwin_core import read_rows, kingdom_enum
from src.geohash import Geohash, build_geohash_series
from contexttimer import Timer


logger = logging.getLogger(__name__)


class DarwinCoreAggregations(NamedTuple):
    taxon_counts: pl.DataFrame
    """
    Schema:
    - `geohash`: `str`
    - `taxonKey`: `int`
    - `count`: `int`
    """

    taxon_counts_2: pl.DataFrame
    """
    Schema:
    - `geohash`: `str`
    - `kingdom`: `enum`
    - `rank`: `enum`
    - `name`: `str`
    - `count`: `int`
    """

    taxon_index: pl.DataFrame
    """
    Schema:
    - `taxonKey`: `int`
    - `verbatimScientificName`: `str`
    """

    @classmethod
    def build(cls, input_file: str, geohash_precision: int) -> Self:
        taxon_counts = pl.DataFrame(
            schema={
                "geohash": pl.String,
                "taxonKey": pl.UInt64,
                "count": pl.UInt32,
            }
        )

        class TaxonRank(Enum):
            phylum = "phylum"
            class_ = "class"
            order = "order"
            family = "family"
            genus = "genus"
            species = "species"

        # Each of these values is also the name of a Darwin Core column
        taxon_rank_enum = pl.Enum(TaxonRank)

        taxon_counts_2 = pl.DataFrame(
            schema={
                "geohash": pl.String,
                "kingdom": kingdom_enum,
                "rank": taxon_rank_enum,
                "name": pl.String,
                "count": pl.UInt32,
            }
        )

        # Will this work for eBird?
        # geohash_to_taxon_id_to_user_to_count: DefaultDict[
        #     Geohash, DefaultDict[TaxonId, Counter[str]]
        # ] = defaultdict(lambda: defaultdict(Counter))

        taxon_index = pl.DataFrame(
            schema={
                "taxonKey": pl.UInt64,
                "verbatimScientificName": pl.String,
            }
        )

        with Timer(output=logger.info, prefix="Reading rows"):
            for read_dataframe in read_rows(
                input_file,
                columns=[
                    "decimalLatitude",
                    "decimalLongitude",
                    "taxonKey",
                    "verbatimScientificName",
                    "recordedBy",
                    "kingdom",
                    *map(lambda rank: rank.value, TaxonRank),
                ],
            ):
                dataframe_with_geohash = read_dataframe.pipe(
                    build_geohash_series,
                    lat_col=pl.col("decimalLatitude"),
                    lon_col=pl.col("decimalLongitude"),
                    precision=geohash_precision,
                )

                taxon_counts.vstack(
                    dataframe_with_geohash.group_by(["geohash", "taxonKey"]).agg(
                        pl.len().alias("count")
                    ),
                    in_place=True,
                )

                for variant in TaxonRank:
                    aggregated = (
                        dataframe_with_geohash.lazy()
                        .group_by(["geohash", "kingdom", variant.value])
                        .agg(
                            pl.len().alias("count"),
                        )
                        .rename({variant.value: "name"})
                        .with_columns(
                            pl.lit(variant, dtype=taxon_rank_enum).alias("rank"),
                        )
                        .select(
                            ["geohash", "kingdom", "rank", "name", "count"]
                        )
                        .collect()
                    )
                    taxon_counts_2.vstack(
                        aggregated,
                        in_place=True,
                    )

                taxon_index = taxon_index.vstack(
                    dataframe_with_geohash.select(
                        ["taxonKey", "verbatimScientificName"]
                    )
                ).unique()

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

        taxon_counts = (
            taxon_counts.group_by(["geohash", "taxonKey"])
            .agg(pl.col("count").sum())
            .sort(by="geohash")
        )

        taxon_counts_2 = taxon_counts_2.group_by(
            ["geohash", "kingdom", "rank", "name"]
        ).agg(pl.col("count").sum())

        return cls(
            taxon_counts=taxon_counts,
            taxon_counts_2=taxon_counts_2,
            taxon_index=taxon_index,
        )

    def scientific_name_for_taxon_key(self, taxon_key: int) -> str:
        column = self.taxon_index.filter(pl.col("taxonKey") == taxon_key).get_column(
            "verbatimScientificName"
        )
        if len(column) > 1:
            # TODO: what should we do here? e.g. "Sciurus carolinensis leucotis" and "Sciurus carolinensis"
            # raise ValueError(f"Multiple scientific names for taxon key {taxon_key}")
            logger.error(f"Multiple scientific names for taxon key {taxon_key}")
            return column.limit(1).item()
        return column.item()

    # @functools.cache
    def ordered_geohashes(self) -> List[Geohash]:
        return (
            self.taxon_counts.select("geohash")
            .unique()
            .sort(by="geohash")
            .get_column("geohash")
            .to_list()
        )

    # @functools.cache
    def ordered_taxon_keys(self) -> List[int]:
        return (
            self.taxon_counts.select("taxonKey")
            .unique()
            .sort(by="taxonKey")
            .get_column("taxonKey")
            .to_list()
        )
