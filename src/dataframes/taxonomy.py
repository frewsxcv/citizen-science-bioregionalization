import logging

import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES, TAXON_RANK_VALUES
from src.dataframes.darwin_core import DarwinCoreSchema
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.geocode import filter_by_bounding_box, with_geocode_lf
from src.types import Bbox

logger = logging.getLogger(__name__)


class TaxonomySchema(dy.Schema):
    """
    A dataframe of taxonomy information. Note that this may include taxa for geocodes that were filtered out.
    """

    taxonId = dy.UInt32(nullable=False)  # Unique identifier for each taxon
    kingdom = dy.Enum(KINGDOM_VALUES, nullable=True)
    taxonRank = dy.Enum(TAXON_RANK_VALUES, nullable=False)
    scientificName = dy.String(nullable=True)
    gbifTaxonId = dy.Int64(nullable=False)


def build_taxonomy_lf(
    darwin_core_lf: dy.LazyFrame[DarwinCoreSchema],
    geocode_precision: int,
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
    bounding_box: Bbox,
) -> dy.LazyFrame[TaxonomySchema]:
    """Build a validated TaxonomySchema LazyFrame from Darwin Core data.

    Args:
        darwin_core_lf: LazyFrame of Darwin Core occurrence records
        geocode_precision: H3 resolution (0-15). Higher = smaller hexagons.
        geocode_lf: LazyFrame of geocodes (used to filter taxa to valid geocodes)
        bounding_box: Geographic bounding box to filter records

    Returns:
        A validated LazyFrame conforming to TaxonomySchema
    """
    logger.info("build_taxonomy_lf: Starting")

    # Use semi-join to filter without materializing geocodes to Python list
    geocode_filter_lf = geocode_lf.select("geocode")

    lf = (
        darwin_core_lf.pipe(filter_by_bounding_box, bounding_box=bounding_box)
        .pipe(with_geocode_lf, geocode_precision=geocode_precision)
        # Semi-join: keeps rows where geocode exists, without loading list to memory
        .join(geocode_filter_lf, on="geocode", how="semi")
        .select(
            "kingdom",
            "taxonRank",
            "scientificName",
            # pl.col("acceptedTaxonKey").alias("gbifTaxonId"),
            pl.col("taxonKey").alias("gbifTaxonId"),
        )
        .unique(
            subset=[
                "scientificName",  # Need to confirm this. Will there be different scientific names for the same GBIF taxon ID?
                "gbifTaxonId",
            ],
        )
        # Add a unique taxonId for each row
        .with_row_index("taxonId")
        .cast({"taxonId": pl.UInt32})
    )

    return TaxonomySchema.validate(lf, eager=False)
