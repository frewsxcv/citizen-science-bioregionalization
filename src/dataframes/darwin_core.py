"""Darwin Core occurrence data schema.

This module defines the schema for validated Darwin Core occurrence data,
ensuring that data loaded from Darwin Core archives or Parquet files
meets the requirements for downstream analysis.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import polars as pl

logger = logging.getLogger(__name__)

from src.darwin_core_utils import build_darwin_core_raw_lf, build_taxon_filter
from src.geocode import filter_by_bounding_box
from src.types import Bbox, TaxonScope

def build_darwin_core_lf(
    source_path: Union[str, Path],
    bounding_box: Bbox,
    limit: Union[int, None] = None,
    scope: Optional[TaxonScope] = None,
) -> pl.LazyFrame:
    """Build a validated Darwin Core lazyframe from a source file.

    Args:
        source_path: Path to either a Darwin Core archive directory or Parquet file/directory.
        bounding_box: Geographic bounding box to filter records.
        limit: Optional maximum number of records to return.
        scope: Optional taxonomic scope to restrict to (e.g. order:Coleoptera).
            None means no taxonomic filtering.

    Returns:
        A validated LazyFrame conforming to DarwinCoreSchema.

    Raises:
        ValueError: if a scope is requested but the source lacks that rank's key
            column.
    """
    logger.info(
        f"build_darwin_core_lf: Loading from {source_path}, "
        f"bounding_box={bounding_box}, limit={limit}, scope={scope}"
    )
    lf = build_darwin_core_raw_lf(source_path=str(source_path))

    # Apply geographic bounding box filter
    lf = lf.pipe(filter_by_bounding_box, bounding_box=bounding_box)

    # Apply the taxonomic scope. This must happen before the limit below:
    # limiting first would take the first N records overall and *then* narrow to
    # the clade, which for any selective scope yields near-zero rows.
    if scope is not None:
        available = lf.collect_schema().names()
        if scope.column not in available:
            raise ValueError(
                f"Cannot scope to {scope}: source {source_path} has no "
                f"{scope.column!r} column. Available columns: {sorted(available)}"
            )
        lf = lf.filter(build_taxon_filter(scope))

    # Apply limit if specified
    if limit is not None:
        lf = lf.limit(limit)

    # Select only the columns we need. The rank key column is deliberately not
    # carried forward: it has served its purpose as a filter, and keeping the
    # downstream schema fixed means no consumer needs to know about scoping.
    lf = lf.select(
        "decimalLatitude",
        "decimalLongitude",
        "scientificName",
        "taxonKey",
        "individualCount",
    )

    return lf
