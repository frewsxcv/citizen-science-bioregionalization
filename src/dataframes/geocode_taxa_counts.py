import logging

import dataframely as dy
import polars as pl

from src.dataframes.darwin_core import DarwinCoreSchema
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.taxonomy import TaxonomySchema
from src.geocode import filter_by_bounding_box, with_geocode_lf
from src.types import Bbox

logger = logging.getLogger(__name__)


class GeocodeTaxaCountsSchema(dy.Schema):
    geocode = dy.UInt64(nullable=False)
    taxonId = dy.UInt32(nullable=False)
    count = dy.UInt32(nullable=False)


def build_geocode_taxa_counts_df(
    darwin_core_lf: dy.LazyFrame[DarwinCoreSchema],
    geocode_precision: int,
    taxonomy_lf: dy.LazyFrame[TaxonomySchema],
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
    bounding_box: Bbox,
) -> dy.DataFrame[GeocodeTaxaCountsSchema]:
    """Build a GeocodeTaxaCountsSchema DataFrame from Darwin Core data.

    Aggregates occurrence counts per geocode and taxon.

    Args:
        darwin_core_lf: LazyFrame of Darwin Core occurrence records
        geocode_precision: H3 resolution (0-15)
        taxonomy_lf: LazyFrame of taxonomy information
        geocode_lf: LazyFrame of geocodes (without edges)
        bounding_box: Geographic bounding box to filter records

    Returns:
        A validated DataFrame conforming to GeocodeTaxaCountsSchema
    """
    geocodes = (
        geocode_lf.select("geocode").collect(engine="streaming").to_series().to_list()
    )

    aggregated = (
        darwin_core_lf.select(
            "decimalLatitude",
            "decimalLongitude",
            "scientificName",
            pl.col("taxonKey").cast(pl.String()).alias("gbifTaxonId"),
            "individualCount",
        )
        .pipe(filter_by_bounding_box, bounding_box=bounding_box)
        .pipe(with_geocode_lf, geocode_precision=geocode_precision)
        .select(
            "geocode",
            "scientificName",
            "gbifTaxonId",
            "individualCount",
        )
        .filter(
            # Ensure geocode exists and is not an edge
            pl.col("geocode").is_in(geocodes)
        )
        .join(
            taxonomy_lf.select("taxonId", "scientificName", "gbifTaxonId"),
            on=["scientificName", "gbifTaxonId"],
            how="left",
        )
        .select(
            "geocode",
            "taxonId",
            "individualCount",
        )
        .group_by("geocode", "taxonId")
        .agg(pl.col("individualCount").fill_null(1).sum().alias("count"))
        .collect(engine="streaming")
        .sort(by="geocode")
    )

    # Handle any missing taxonId values (this shouldn't happen if taxonomy is comprehensive)
    if aggregated.filter(pl.col("taxonId").is_null()).height > 0:
        logger.warning(
            f"Found {aggregated.filter(pl.col('taxonId').is_null()).height} records with no matching taxonomy entry"
        )
        # Drop records with no matching taxonomy as they can't be handled in the new schema
        aggregated = aggregated.filter(pl.col("taxonId").is_not_null())

    return GeocodeTaxaCountsSchema.validate(
        aggregated.with_columns(
            pl.col("taxonId").cast(pl.UInt32), pl.col("count").cast(pl.UInt32)
        )
    )
