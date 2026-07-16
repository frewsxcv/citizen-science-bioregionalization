import logging

import dataframely as dy

import bioregion_rs
from src.dataframes.darwin_core import DarwinCoreSchema
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.types import Bbox

logger = logging.getLogger(__name__)


class TaxonomySchema(dy.Schema):
    """
    A dataframe of taxonomy information. Note that this may include taxa for geocodes that were filtered out.
    """

    taxonId = dy.UInt32(nullable=False)  # Unique identifier for each taxon
    scientificName = dy.String(nullable=True)
    gbifTaxonId = dy.UInt32(nullable=False)


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

    darwin_core_df = darwin_core_lf.select(
        "decimalLatitude",
        "decimalLongitude",
        "scientificName",
        "taxonKey",
    ).collect()
    geocode_df = geocode_lf.select("geocode").collect()

    df = bioregion_rs.build_taxonomy(
        darwin_core_df,
        geocode_precision,
        geocode_df,
        bounding_box.min_lat,
        bounding_box.max_lat,
        bounding_box.min_lng,
        bounding_box.max_lng,
    )
    return TaxonomySchema.validate(df.lazy(), eager=False)
