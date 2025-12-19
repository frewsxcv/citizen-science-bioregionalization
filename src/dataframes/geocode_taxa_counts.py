import logging

import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.taxonomy import TaxonomySchema
from src.geocode import with_geocode_lazy_frame
from src.logging import Timer

logger = logging.getLogger(__name__)


class GeocodeTaxaCountsSchema(dy.Schema):
    geocode = dy.UInt64(nullable=False)
    taxonId = dy.UInt32(nullable=False)
    count = dy.UInt32(nullable=False)

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: pl.LazyFrame,
        geocode_precision: int,
        taxonomy_dataframe: dy.DataFrame[TaxonomySchema],
        geocode_lazyframe: dy.LazyFrame[GeocodeNoEdgesSchema],
    ) -> dy.DataFrame["GeocodeTaxaCountsSchema"]:
        with Timer(output=logger.info, prefix="Reading rows"):
            geocodes = (
                geocode_lazyframe.select("geocode")
                .collect(engine="streaming")
                .to_series()
                .to_list()
            )

            # First, create the raw aggregation with the old schema
            raw_aggregated = (
                darwin_core_csv_lazy_frame.pipe(
                    with_geocode_lazy_frame, geocode_precision=geocode_precision
                )
                .filter(
                    # Ensure geocode exists and is not an edge
                    pl.col("geocode").is_in(geocodes)
                )
                .group_by(["geocode", "kingdom", "scientificName", "taxonRank"])
                .agg(pl.len().alias("count"))
                .select(["geocode", "kingdom", "taxonRank", "scientificName", "count"])
                .cast(
                    {"kingdom": pl.Enum(KINGDOM_VALUES), "taxonRank": pl.Categorical()}
                )
                .sort(by="geocode")
                .collect(engine="streaming")
            )

            # Join with taxonomy dataframe to get taxonId
            joined = raw_aggregated.join(
                taxonomy_dataframe.select(
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

            return cls.validate(
                aggregated.with_columns(
                    pl.col("taxonId").cast(pl.UInt32), pl.col("count").cast(pl.UInt32)
                )
            )
