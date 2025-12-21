import polars as pl
import polars_h3

from src.types import Bbox


def with_geocode_lazy_frame(
    lazy_frame: pl.LazyFrame, geocode_precision: int
) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimalLatitude and decimalLongitude columns."""
    return lazy_frame.with_columns(
        polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ).alias("geocode"),
    )


def filter_by_bounding_box(
    lazy_frame: pl.LazyFrame,
    bounding_box: Bbox,
    lat_col: str = "decimalLatitude",
    lng_col: str = "decimalLongitude",
) -> pl.LazyFrame:
    return lazy_frame.filter(
        (pl.col(lat_col) >= bounding_box.min_lat)
        & (pl.col(lat_col) <= bounding_box.max_lat)
        & (pl.col(lng_col) >= bounding_box.min_lng)
        & (pl.col(lng_col) <= bounding_box.max_lng)
    )


def select_geocode_lazy_frame(
    lazy_frame: pl.LazyFrame, geocode_precision: int
) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimalLatitude and decimalLongitude columns."""
    return lazy_frame.select(
        geocode=polars_h3.latlng_to_cell(
            "decimalLatitude",
            "decimalLongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ),
    )
