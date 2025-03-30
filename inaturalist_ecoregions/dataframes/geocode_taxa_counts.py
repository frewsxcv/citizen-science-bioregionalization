import logging
import polars as pl
from typing import List
from inaturalist_ecoregions.darwin_core import kingdom_enum
from inaturalist_ecoregions.logging import Timer
import polars_h3

from inaturalist_ecoregions.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
from inaturalist_ecoregions.data_container import DataContainer, assert_dataframe_schema
from inaturalist_ecoregions.dataframes.geocode import GeocodeDataFrame
from inaturalist_ecoregions.dataframes.taxonomy import TaxonomyDataFrame

logger = logging.getLogger(__name__)


class GeocodeTaxaCountsDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "geocode": pl.String(),
        "taxonId": pl.UInt32(),
        "count": pl.UInt32(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame,
        geocode_precision: int,
        taxonomy_dataframe: TaxonomyDataFrame,
    ):
        with Timer(output=logger.info, prefix="Reading rows"):
            # First, create the raw aggregation with the old schema
            raw_aggregated = (
                darwin_core_csv_lazy_frame.lf.with_columns(
                    polars_h3.latlng_to_cell(
                        "decimalLatitude",
                        "decimalLongitude",
                        resolution=geocode_precision,
                        return_dtype=pl.Utf8,
                    ).alias("geocode"),
                )
                .group_by(["geocode", "kingdom", "scientificName", "taxonRank"])
                .agg(pl.len().alias("count"))
                .select(["geocode", "kingdom", "taxonRank", "scientificName", "count"])
                .sort(by="geocode")
                .collect()
            )

            # Join with taxonomy dataframe to get taxonId
            joined = raw_aggregated.join(
                taxonomy_dataframe.df.select(
                    ["taxonId", "kingdom", "scientificName", "taxonRank"]
                ),
                on=["kingdom", "scientificName", "taxonRank"],
                how="left",
            )

            # Select only the columns we need for our new schema
            aggregated = joined.select(["geocode", "taxonId", "count"])

            # Handle any missing taxonId values (this shouldn't happen if taxonomy is comprehensive)
            if aggregated.filter(pl.col("taxonId").is_null()).height > 0:
                logger.warning(
                    f"Found {aggregated.filter(pl.col('taxonId').is_null()).height} records with no matching taxonomy entry"
                )
                # Drop records with no matching taxonomy as they can't be handled in the new schema
                aggregated = aggregated.filter(pl.col("taxonId").is_not_null())

            return cls(aggregated)
