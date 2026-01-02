"""Darwin Core occurrence data schema.

This module defines the schema for validated Darwin Core occurrence data,
ensuring that data loaded from Darwin Core archives or Parquet files
meets the requirements for downstream analysis.
"""

from pathlib import Path
from typing import Union

import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES, TAXON_RANK_VALUES
from src.darwin_core_utils import build_darwin_core_raw_lf
from src.geocode import filter_by_bounding_box
from src.types import Bbox


class DarwinCoreSchema(dy.Schema):
    """Schema for Darwin Core occurrence records.

    This schema defines the core fields needed for bioregionalization analysis
    from Darwin Core occurrence data. It validates that required geographic
    and taxonomic fields are present and properly typed.
    """

    # Geographic fields (required for spatial analysis)
    decimalLatitude = dy.Float64(nullable=False)
    decimalLongitude = dy.Float64(nullable=False)

    # Taxonomic hierarchy
    kingdom = dy.Enum(KINGDOM_VALUES, nullable=True)

    # Taxonomic metadata
    taxonRank = dy.Enum(TAXON_RANK_VALUES, nullable=False)
    scientificName = dy.String(nullable=True)
    taxonKey = dy.String(nullable=True)

    individualCount = dy.Int32(nullable=True)

    @dy.rule()
    def valid_latitude(cls) -> pl.Expr:
        """Validate that latitude is within valid range [-90, 90]."""
        return pl.col("decimalLatitude").is_between(-90, 90)

    @dy.rule()
    def valid_longitude(cls) -> pl.Expr:
        """Validate that longitude is within valid range [-180, 180]."""
        return pl.col("decimalLongitude").is_between(-180, 180)


def build_darwin_core_lf(
    source_path: Union[str, Path],
    bounding_box: Bbox,
    limit: Union[int, None] = None,
    taxon_filter: str = "",
) -> dy.LazyFrame[DarwinCoreSchema]:
    """Build a validated Darwin Core lazyframe from a source file.

    Args:
        source_path: Path to either a Darwin Core archive directory or Parquet file/directory.
        bounding_box: Geographic bounding box to filter records.
        limit: Optional maximum number of records to return.
        taxon_filter: Optional taxon name to filter by (e.g., 'Aves').

    Returns:
        A validated LazyFrame conforming to DarwinCoreSchema.
    """
    lf = build_darwin_core_raw_lf(source_path=str(source_path))

    # Apply geographic bounding box filter
    lf = lf.pipe(filter_by_bounding_box, bounding_box=bounding_box)

    # Apply limit if specified
    if limit is not None:
        lf = lf.limit(limit)

    # Select only the columns we need
    lf = lf.select(
        "decimalLatitude",
        "decimalLongitude",
        "kingdom",
        "taxonRank",
        "scientificName",
        "taxonKey",
        "individualCount",
    )

    return DarwinCoreSchema.validate(lf, eager=False)
