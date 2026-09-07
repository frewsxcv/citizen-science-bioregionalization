import polars as pl
import polars_h3

from src.types import Bbox


def with_geocode_lf(lf: pl.LazyFrame, geocode_precision: int) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimalLatitude and decimalLongitude columns."""
    return lf.with_columns(
        polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ).alias("geocode"),
    )


def filter_by_bounding_box(
    lf: pl.LazyFrame,
    bounding_box: Bbox,
    lat_col: str = "decimalLatitude",
    lng_col: str = "decimalLongitude",
) -> pl.LazyFrame:
    """Filter a LazyFrame to rows within a geographic bounding box.

    Args:
        lf: The LazyFrame to filter.
        bounding_box: Geographic bounding box to filter records.
        lat_col: Name of the latitude column.
        lng_col: Name of the longitude column.

    Returns:
        A LazyFrame filtered to rows with valid coordinates within the bounding box.
    """
    return lf.filter(
        pl.col(lat_col).is_not_null()
        & pl.col(lng_col).is_not_null()
        & pl.col(lat_col).is_between(bounding_box.min_lat, bounding_box.max_lat)
        & pl.col(lng_col).is_between(bounding_box.min_lng, bounding_box.max_lng)
    )


def select_geocode_lf(lf: pl.LazyFrame, geocode_precision: int) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimalLatitude and decimalLongitude columns."""
    return lf.select(
        geocode=polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ),
    )


def filter_sparse_geocodes_lf(
    lf: pl.LazyFrame,
    geocode_precision: int,
    min_records: int,
) -> pl.LazyFrame:
    """Drop occurrences in hexagons holding fewer than `min_records` records.

    A hexagon observed once yields a composition vector of a single taxon, which
    is maximally distant from everything else and says more about where people
    looked than about what lives there. Citizen-science effort is concentrated
    enough that these dominate the long tail: in a country-scale run at
    precision 5, a quarter of hexagons held fewer than 8 records against a
    median of 133.

    Applied to the occurrence records rather than to the geocodes, so that the
    geocode set and the taxa counts are both derived from the same rows and
    agree by construction.

    Args:
        lf: Occurrence records with decimalLatitude/decimalLongitude.
        geocode_precision: H3 resolution; must match the run's precision.
        min_records: Minimum records a hexagon must hold to be kept.

    Returns:
        The input restricted to sufficiently sampled hexagons.
    """
    with_geocode = lf.with_columns(
        polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ).alias("_geocode")
    )
    well_sampled = (
        with_geocode.group_by("_geocode")
        .len()
        .filter(pl.col("len") >= min_records)
        .select("_geocode")
    )
    return with_geocode.join(well_sampled, on="_geocode", how="semi").drop("_geocode")
