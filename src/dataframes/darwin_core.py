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

#: Column added to carry the sampling hash. Prefixed so it cannot collide with
#: a Darwin Core term.
_ROW_INDEX = "_sample_row_index"


def sample_records_lf(
    lf: pl.LazyFrame, target: int, seed: int = 0
) -> pl.LazyFrame:
    """Reduce `lf` to approximately `target` rows, chosen uniformly.

    `limit` would be cheaper, and is wrong here. It takes the first N rows in
    scan order, and a GBIF snapshot's file order is grouped by source dataset,
    so the rows it keeps are a sample of whichever datasets sort earliest rather
    than of the region. On the published bounding box a 300M cap discards 54.4%
    of the records that way.

    Each row is instead kept with probability `target / total`, decided by a
    seeded hash of its row index. That is a stateless per-row test, so it
    streams and holds nothing; the one extra cost is a pass to count the rows,
    measured at 1.5s and well below the pipeline's other stages.

    The count is exact but the result is binomial around `target`, not equal to
    it. Callers wanting a precise row count want `limit`, and should accept what
    it selects.

    Args:
        lf: The records to sample.
        target: Approximate number of rows to keep.
        seed: Seed for the sampling hash. Two runs at one seed agree exactly.

    Returns:
        `lf` unchanged if it already holds `target` rows or fewer, else a
        uniform sample of approximately that many.
    """
    total = lf.select(pl.len()).collect(engine="streaming").item()
    if total <= target:
        logger.info(
            f"sample_records_lf: {total} rows is already at or under the "
            f"{target}-row target; keeping all of them"
        )
        return lf

    keep_probability = target / total
    logger.info(
        f"sample_records_lf: sampling {target} of {total} rows "
        f"(p={keep_probability:.6f}, seed={seed})"
    )
    return (
        lf.with_row_index(_ROW_INDEX)
        # hash() is uniform over u64, so dividing by 2**64 gives a value in
        # [0, 1) that is fixed for a given row and independent between rows.
        .filter(
            (pl.col(_ROW_INDEX).hash(seed=seed) / pl.lit(2.0**64)) < keep_probability
        )
        .drop(_ROW_INDEX)
    )

def build_darwin_core_lf(
    source_path: Union[str, Path],
    bounding_box: Bbox,
    limit: Union[int, None] = None,
    scope: Optional[TaxonScope] = None,
    sample_records: Union[int, None] = None,
    seed: int = 0,
) -> pl.LazyFrame:
    """Build a validated Darwin Core lazyframe from a source file.

    Args:
        source_path: Path to either a Darwin Core archive directory or Parquet file/directory.
        bounding_box: Geographic bounding box to filter records.
        limit: Optional maximum number of records to return, taken in scan
            order. Ignored when `sample_records` is given.
        sample_records: Optional approximate number of records to keep, chosen
            uniformly rather than by scan order. See `sample_records_lf`.
        seed: Seed for the sampling hash.
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

    # A uniform sample and a scan-order limit are alternatives, not a pipeline:
    # applying both would sample from an already biased head.
    if sample_records is not None:
        lf = sample_records_lf(lf, sample_records, seed=seed)
    elif limit is not None:
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
