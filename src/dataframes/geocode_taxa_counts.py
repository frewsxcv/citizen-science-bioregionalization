import logging
import polars as pl
from typing import List
from src.logging import Timer

from src.data_container import DataContainer, assert_dataframe_schema
from src.dataframes.geocode import GeocodeDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.geocode import geocode_lazy_frame
from polars_darwin_core import DarwinCoreLazyFrame

logger = logging.getLogger(__name__)


class GeocodeTaxaCountsDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "geocode": pl.UInt64(),
        "taxonId": pl.UInt32(),
        "count": pl.UInt32(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreLazyFrame,
        geocode_precision: int,
        taxonomy_dataframe: TaxonomyDataFrame,
    ):
        with Timer(output=logger.info, prefix="Reading rows"):
            # First, create the raw aggregation with the old schema
            raw_aggregated = (
                darwin_core_csv_lazy_frame._inner
                .pipe(geocode_lazy_frame, geocode_precision=geocode_precision)
                .filter(pl.col("geocode").is_not_null())
                .group_by(["geocode", "kingdom", "scientificName", "taxonRank"])
                .agg(pl.len().alias("count"))
                .select(["geocode", "kingdom", "taxonRank", "scientificName", "count"])
                .cast({"taxonRank": pl.Categorical()})
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
