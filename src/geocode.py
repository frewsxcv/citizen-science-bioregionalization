import polars as pl
import polars_h3  # type: ignore


def geocode_lazy_frame(
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