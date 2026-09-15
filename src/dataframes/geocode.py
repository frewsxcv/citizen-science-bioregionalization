import polars as pl
import polars_h3

import bioregion_rs
from src.types import Bbox

def build_geocode_lf(
    darwin_core_lf: pl.LazyFrame,
    geocode_precision: int,
    bounding_box: Bbox,
) -> pl.LazyFrame:
    """Build a GeocodeSchema LazyFrame from Darwin Core occurrence data.

    Args:
        darwin_core_lf: LazyFrame of Darwin Core occurrence records
        geocode_precision: H3 resolution (0-15). Higher = smaller hexagons.
        bounding_box: Geographic bounding box to determine edge hexagons

    Returns:
        A validated LazyFrame conforming to GeocodeSchema
    """
    # Reduced to distinct cells here rather than in Rust, which used to take the
    # coordinates and geocode them itself. That required collecting every record
    # in the run to make the call: 16.4 GB on the published bounding box
    # uncapped, against a 16 GB runner, and the largest single allocation in the
    # pipeline. Geocoding and deduplicating stream, and the frame that crosses
    # into Rust is then one row per hexagon.
    geocode_df = (
        darwin_core_lf.select(
            polars_h3.latlng_to_cell(
                "decimalLatitude",
                "decimalLongitude",
                resolution=geocode_precision,
                return_dtype=pl.UInt64,
            ).alias("geocode")
        )
        .drop_nulls()
        .unique()
        .sort("geocode")
        .collect(engine="streaming")
    )
    df = bioregion_rs.build_geocode(
        geocode_df,
        bounding_box.min_lat,
        bounding_box.max_lat,
        bounding_box.min_lng,
        bounding_box.max_lng,
    )
    return df.lazy()


def build_geocode_df(
    darwin_core_lf: pl.LazyFrame,
    geocode_precision: int,
    bounding_box: Bbox,
) -> pl.DataFrame:
    """Build a GeocodeSchema DataFrame from Darwin Core occurrence data.

    This is a convenience wrapper around build_geocode_lf() that collects
    the result into an eager DataFrame.

    Args:
        darwin_core_lf: LazyFrame of Darwin Core occurrence records
        geocode_precision: H3 resolution (0-15). Higher = smaller hexagons.
        bounding_box: Geographic bounding box to determine edge hexagons

    Returns:
        A validated DataFrame conforming to GeocodeSchema
    """
    lf = build_geocode_lf(darwin_core_lf, geocode_precision, bounding_box)
    return lf.collect()


def build_geocode_no_edges_lf(
    geocode_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Build a GeocodeNoEdgesSchema by filtering out edge hexagons from a GeocodeSchema.

    This function is fully lazy - no collection occurs until downstream code
    explicitly collects the result.

    Args:
        geocode_lf: A validated GeocodeSchema lazy dataframe

    Returns:
        A validated GeocodeNoEdgesSchema lazyframe with edge hexagons removed
    """
    lf = geocode_lf.filter(~pl.col("is_edge")).sort(by="geocode")
    return lf


def index_of_geocode(
    geocode: int,
    geocode_df: pl.DataFrame | pl.DataFrame,
) -> int:
    """Find the index of a geocode in the dataframe.

    Args:
        geocode: The geocode to find
        geocode_df: DataFrame to search in

    Returns:
        The 0-based index of the geocode

    Raises:
        ValueError: If the geocode is not found
    """
    index = geocode_df["geocode"].index_of(geocode)
    if index is None:
        raise ValueError(f"Geocode {geocode} not found in GeocodeDataFrame")
    return index
