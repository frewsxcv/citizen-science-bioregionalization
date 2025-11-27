import polars as pl
import polars_h3


def with_geocode_lazy_frame(
    lazy_frame: pl.LazyFrame, geocode_precision: int
) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimallatitude and decimallongitude columns."""
    return lazy_frame.with_columns(
        polars_h3.latlng_to_cell(
            "decimallatitude",
            "decimallongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ).alias("geocode"),
    )


def select_geocode_lazy_frame(
    lazy_frame: pl.LazyFrame, geocode_precision: int
) -> pl.LazyFrame:
    """Geocodes a lazy frame with decimallatitude and decimallongitude columns."""
    return lazy_frame.select(
        geocode=polars_h3.latlng_to_cell(
            "decimallatitude",
            "decimallongitude",
            resolution=geocode_precision,
            return_dtype=pl.UInt64,
        ),
    )
