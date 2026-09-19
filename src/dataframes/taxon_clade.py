"""Which clade each taxon belongs to.

The findings section asks two questions that need a clade per taxon: what share
of the records and of the taxa each clade contributes, and whether two clades
clustered separately draw the same map. Both key off `taxonId`, so this reduces
the rank columns carried on the occurrence frame down to one row per taxon.

The reduction happens against the spilled Darwin Core parquet rather than the
source, so it is a local pass -- see the note on the sampling floor in
`notebook.py` for why that distinction is load-bearing.
"""

import logging
from typing import Optional

import polars as pl

from src.dataframes.darwin_core import CLADE_COLUMNS

logger = logging.getLogger(__name__)


def build_taxon_clade_lf(
    darwin_core_lf: pl.LazyFrame,
    taxonomy_lf: pl.LazyFrame,
) -> Optional[pl.LazyFrame]:
    """Map each `taxonId` to its kingdom and class.

    Args:
        darwin_core_lf: Occurrence records, carrying the columns in
            `CLADE_COLUMNS`.
        taxonomy_lf: The taxonomy, which supplies `taxonId`.

    Returns:
        A LazyFrame of `taxonId`, `kingdom`, `class`, one row per taxon, or
        `None` if the source carried no rank columns -- a run against such a
        source simply reports no clade findings.
    """
    available = darwin_core_lf.collect_schema().names()
    if any(c not in available for c in CLADE_COLUMNS):
        logger.info(
            "build_taxon_clade_lf: Darwin Core frame has no clade columns; "
            "skipping the clade findings"
        )
        return None

    logger.info("build_taxon_clade_lf: Starting")

    # `min()` rather than `first()`. A taxon should have one kingdom and one
    # class, but the backbone is not perfectly consistent and a group_by's
    # iteration order is not fixed, so `first()` would be a reproducibility
    # hazard of exactly the kind this pipeline has been bitten by: the same
    # input assigning a different clade between runs. `min()` is total and
    # order-independent, and where a taxon really is inconsistent it at least
    # picks the same way every time.
    per_taxon = (
        darwin_core_lf.select(
            "scientificName",
            pl.col("taxonKey").alias("gbifTaxonId"),
            *CLADE_COLUMNS,
        )
        .group_by("scientificName", "gbifTaxonId")
        .agg(*[pl.col(c).min().alias(c) for c in CLADE_COLUMNS])
    )

    # Joined onto the taxonomy rather than the other way round, so the result
    # carries exactly the taxa the rest of the run knows about -- and in the
    # taxonomy's order, which is sorted, so nothing here depends on the
    # group_by above having produced any particular one.
    return (
        taxonomy_lf.select("taxonId", "scientificName", "gbifTaxonId")
        .join(per_taxon, on=["scientificName", "gbifTaxonId"], how="left")
        .select("taxonId", *CLADE_COLUMNS)
        .sort("taxonId")
    )
