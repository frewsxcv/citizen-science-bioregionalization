"""Darwin Core occurrence data schema.

This module defines the schema for validated Darwin Core occurrence data,
ensuring that data loaded from Darwin Core archives or Parquet files
meets the requirements for downstream analysis.
"""

import logging
from pathlib import Path
from typing import Union

import polars as pl

logger = logging.getLogger(__name__)

from src.darwin_core_utils import build_darwin_core_raw_lf
from src.geocode import filter_by_bounding_box
from src.types import Bbox

def build_darwin_core_lf(
    source_path: Union[str, Path],
    bounding_box: Bbox,
    limit: Union[int, None] = None,
    taxon_filter: str = "",
) -> pl.LazyFrame:
    """Build a validated Darwin Core lazyframe from a source file.

    Args:
        source_path: Path to either a Darwin Core archive directory or Parquet file/directory.
        bounding_box: Geographic bounding box to filter records.
        limit: Optional maximum number of records to return.
        taxon_filter: Optional taxon name to filter by (e.g., 'Aves').

    Returns:
        A validated LazyFrame conforming to DarwinCoreSchema.
    """
    logger.info(
        f"build_darwin_core_lf: Loading from {source_path}, "
        f"bounding_box={bounding_box}, limit={limit}, taxon_filter={taxon_filter!r}"
    )
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
        "scientificName",
        "taxonKey",
        "individualCount",
    )

    return lf
