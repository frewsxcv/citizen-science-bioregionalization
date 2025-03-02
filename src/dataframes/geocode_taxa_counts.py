import logging
import polars as pl
from typing import List
from src.darwin_core import kingdom_enum
from contexttimer import Timer
import polars_h3

from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
from src.data_container import DataContainer
from src.dataframes.geocode import GeocodeDataFrame

logger = logging.getLogger(__name__)


class GeocodeTaxaCountsDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "geocode": pl.String(),
        "kingdom": kingdom_enum,
        "taxonRank": pl.String(),
        "scientificName": pl.String(),
        "count": pl.UInt32(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert df.schema == self.SCHEMA
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame,
        geocode_precision: int,
    ):
        # Will this work for eBird?
        # geocode_to_taxon_id_to_user_to_count: DefaultDict[
        #     Geocode, DefaultDict[TaxonId, Counter[str]]
        # ] = defaultdict(lambda: defaultdict(Counter))

        with Timer(output=logger.info, prefix="Reading rows"):
            # Temporary: Filter out orders that are not Diptera
            # read_dataframe = read_dataframe.filter(pl.col("order") == "Diptera")

            # `aggregated` is a dataframe that looks like this:
            #
            # +------------+----------+---------+--------------+-------+
            # | geocode    | kingdom  | rank    | name         | count |
            # +------------+----------+---------+--------------+-------+
            # | u4pruydqqvj| Animalia | species | Panthera leo | 42    |
            # +------------+----------+---------+--------------+-------+
            aggregated = (
                darwin_core_csv_lazy_frame.lf.with_columns(
                    polars_h3.latlng_to_cell(
                        "decimalLatitude",
                        "decimalLongitude",
                        resolution=geocode_precision,
                        return_dtype=pl.Utf8
                    ).alias("geocode"),
                )
                .group_by(["geocode", "kingdom", "scientificName", "taxonRank"])
                .agg(pl.len().alias("count"))
                .select(["geocode", "kingdom", "taxonRank", "scientificName", "count"])
                .sort(by="geocode")
                .collect()
            )

            # for row in dataframe_with_geocode.collect(streaming=True).iter_rows(
            #     named=True
            # ):
            #     geocode_to_taxon_id_to_user_to_count[row["geocode"]][
            #         row["taxonKey"]
            #     ][row["recordedBy"]] += 1
            #     # If the observer has seen the taxon more than 5 times, skip it
            #     if (
            #         geocode_to_taxon_id_to_user_to_count[row["geocode"]][
            #             row["taxonKey"]
            #         ][row["recordedBy"]]
            #         > 5
            #     ):
            #         continue

            return cls(aggregated)
