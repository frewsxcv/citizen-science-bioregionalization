import logging

import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.taxonomy import TaxonomySchema
from src.geocode import with_geocode_lazy_frame

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
        taxonomy_lazyframe: dy.LazyFrame[TaxonomySchema],
        geocode_lazyframe: dy.LazyFrame[GeocodeNoEdgesSchema],
    ) -> dy.DataFrame["GeocodeTaxaCountsSchema"]:
        geocodes = (
            geocode_lazyframe.select("geocode")
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

        aggregated = (
            darwin_core_csv_lazy_frame.select(
                "decimalLatitude",
                "decimalLongitude",
                "scientificName",
                pl.col("taxonKey").alias("gbifTaxonId"),
            )
            .cast({"gbifTaxonId": pl.UInt32()})
            .pipe(with_geocode_lazy_frame, geocode_precision=geocode_precision)
            .select(
                "geocode",
                "scientificName",
                "gbifTaxonId",
            )
            .filter(
                # Ensure geocode exists and is not an edge
                pl.col("geocode").is_in(geocodes)
            )
            .join(
                taxonomy_lazyframe.select(  # TODO: don't call lazy() here
                    ["taxonId", "scientificName", "gbifTaxonId"]
                ),
                on=["scientificName", "gbifTaxonId"],
                how="left",
            )
            .select(
                "geocode",
                "taxonId",
            )
            .group_by(["geocode", "taxonId"])
            .agg(pl.len().alias("count"))
            .sort(by="geocode")
            # .show_graph(plan_stage="physical", engine="streaming")
            .collect(engine="streaming")
            # .collect_batches()
        )

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
