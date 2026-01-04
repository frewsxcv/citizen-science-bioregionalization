import logging

import dataframely as dy
import polars as pl

from src.dataframes.darwin_core import DarwinCoreSchema
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.taxonomy import TaxonomySchema
from src.geocode import filter_by_bounding_box, with_geocode_lf
from src.logging import log_action
from src.types import Bbox

logger = logging.getLogger(__name__)


class GeocodeTaxaCountsSchema(dy.Schema):
    geocode = dy.UInt64(nullable=False)
    taxonId = dy.UInt32(nullable=False)
    count = dy.UInt32(nullable=False)


def build_geocode_taxa_counts_lf(
    darwin_core_lf: dy.LazyFrame[DarwinCoreSchema],
    geocode_precision: int,
    taxonomy_lf: dy.LazyFrame[TaxonomySchema],
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
    bounding_box: Bbox,
) -> dy.LazyFrame[GeocodeTaxaCountsSchema]:
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
        .join(
            # Ensure geocode exists and is not an edge
            geocode_lf.select("geocode"),
            on="geocode",
            how="semi",
        )
        .join(
            taxonomy_lf.select("taxonId", "scientificName", "gbifTaxonId"),
            on=["scientificName", "gbifTaxonId"],
        )
        .select(
            "geocode",
            "taxonId",
            "individualCount",
        )
        .group_by("geocode", "taxonId")
        .agg(pl.col("individualCount").fill_null(1).sum().alias("count"))
        .sort(by="geocode")
    )

    return GeocodeTaxaCountsSchema.validate(
        aggregated.with_columns(
            pl.col("taxonId").cast(pl.UInt32), pl.col("count").cast(pl.UInt32)
        ),
        eager=False,
    )


def filter_top_taxa_lf(
    geocode_taxa_counts_lf: dy.LazyFrame[GeocodeTaxaCountsSchema],
    max_taxa: int | None = None,
    min_geocode_presence: float | None = None,
) -> dy.LazyFrame[GeocodeTaxaCountsSchema]:
    """
    Filter to most informative taxa before pivoting.

    This reduces the dimensionality of the taxa matrix by keeping only
    the most relevant taxa, which speeds up pivoting and downstream
    distance calculations significantly for large datasets.

    Args:
        geocode_taxa_counts_lf: LazyFrame of geocode-taxa counts

        max_taxa: Keep only top N taxa by total occurrence count.
            If None, no limit is applied. Recommended: 5,000-10,000 for
            large datasets.

        min_geocode_presence: Keep only taxa present in at least this
            fraction of geocodes. If None, no minimum presence filter
            is applied.

            This removes rare taxa that only appear in a small fraction
            of hexagons. Taxa appearing in very few locations add noise
            to distance calculations but don't help distinguish bioregions.

            For example, with 100 geocodes and min_geocode_presence=0.05:
            - Oak tree (present in 80 geocodes, 80%) -> kept
            - Robin (present in 45 geocodes, 45%) -> kept
            - Deer (present in 5 geocodes, 5%) -> kept
            - Rare orchid (present in 2 geocodes, 2%) -> filtered out
            - Single sighting moth (present in 1 geocode, 1%) -> filtered out

            Recommended values:
            - 0.01 (1%): Very permissive, keeps most taxa
            - 0.02-0.05 (2-5%): Good balance of speed and signal
            - 0.10 (10%): Aggressive, focuses on widespread taxa
            - 0.20+ (20%+): Very aggressive, may lose regional specialists

            For bioregionalization, 0.02-0.05 is usually a good starting
            point—it removes the long tail of rare sightings while keeping
            regionally important species.

    Returns:
        A filtered LazyFrame conforming to GeocodeTaxaCountsSchema

    Note:
        When both filters are specified, min_geocode_presence is applied
        first, then max_taxa. This ensures the "top N" selection is made
        from taxa that are already sufficiently widespread.
    """
    logger.info(
        f"filter_top_taxa_lf called with max_taxa={max_taxa}, min_geocode_presence={min_geocode_presence}"
    )
    lf = geocode_taxa_counts_lf

    if min_geocode_presence is not None:
        # Count unique geocodes
        total_geocodes = log_action(
            "Counting unique geocodes for taxa filtering",
            lambda: lf.select("geocode").unique().collect(engine="streaming").height,
        )
        min_geocodes = int(total_geocodes * min_geocode_presence)
        logger.info(
            f"Filtering taxa to those present in at least {min_geocodes} geocodes "
            f"({min_geocode_presence:.1%} of {total_geocodes})"
        )

        # Keep taxa present in enough geocodes
        taxa_to_keep = (
            lf.group_by("taxonId")
            .agg(pl.col("geocode").n_unique().alias("geocode_count"))
            .filter(pl.col("geocode_count") >= min_geocodes)
            .select("taxonId")
        )
        lf = lf.join(taxa_to_keep, on="taxonId", how="semi")

    if max_taxa is not None:
        logger.info(f"Filtering to top {max_taxa} taxa by total count")

        # Keep top N taxa by total count
        top_taxa = (
            lf.group_by("taxonId")
            .agg(pl.col("count").sum().alias("total_count"))
            .sort("total_count", descending=True)
            .head(max_taxa)
            .select("taxonId")
        )
        lf = lf.join(top_taxa, on="taxonId", how="semi")

    return GeocodeTaxaCountsSchema.validate(lf, eager=False)
