import dataframely as dy
import polars as pl
import polars_h3
import polars_st as pl_st

from src.dataframes.darwin_core import DarwinCoreSchema
from src.geocode import select_geocode_lf
from src.types import Bbox


class GeocodeSchema(dy.Schema):
    """Schema for geocode spatial information.

    Contains H3 hexagon identifiers with their center points, boundaries,
    and edge status relative to the bounding box.
    """

    geocode = dy.UInt64(nullable=False)
    center = dy.Any()  # Binary (geometry)
    boundary = dy.Any()  # Binary (geometry)
    # Whether this hexagon is on the edge (intersects bounding box boundary)
    is_edge = dy.Bool(nullable=False)


class GeocodeNoEdgesSchema(GeocodeSchema):
    """Schema that validates there are no edge hexagons in the dataset.

    This schema inherits all columns and validations from GeocodeSchema
    and adds an additional rule to ensure that no hexagons intersect
    the bounding box boundary (i.e., all is_edge values must be False).
    """

    @dy.rule()
    def no_edges(cls) -> pl.Expr:
        """Validate that no hexagons are on the edge of the bounding box."""
        return ~pl.col("is_edge")


def _bbox_boundary_coords(bounding_box: Bbox) -> list[list[float]]:
    """Create bounding box boundary coordinates as a closed linestring.

    Args:
        bounding_box: Geographic bounding box

    Returns:
        List of [lng, lat] coordinates forming a closed ring
    """
    return [
        [bounding_box.min_lng, bounding_box.min_lat],
        [bounding_box.max_lng, bounding_box.min_lat],
        [bounding_box.max_lng, bounding_box.max_lat],
        [bounding_box.min_lng, bounding_box.max_lat],
        [bounding_box.min_lng, bounding_box.min_lat],  # close the ring
    ]


def build_geocode_lf(
    darwin_core_lf: dy.LazyFrame[DarwinCoreSchema],
    geocode_precision: int,
    bounding_box: Bbox,
) -> dy.LazyFrame[GeocodeSchema]:
    """Build a GeocodeSchema LazyFrame from Darwin Core occurrence data.

    This function is fully lazy - no collection occurs until downstream code
    explicitly collects the result.

    Args:
        darwin_core_lf: LazyFrame of Darwin Core occurrence records
        geocode_precision: H3 resolution (0-15). Higher = smaller hexagons.
        bounding_box: Geographic bounding box to determine edge hexagons

    Returns:
        A validated LazyFrame conforming to GeocodeSchema
    """
    # Create bounding box boundary as a linestring for intersection testing
    bbox_coords = _bbox_boundary_coords(bounding_box)

    lf = (
        darwin_core_lf.pipe(select_geocode_lf, geocode_precision=geocode_precision)
        .filter(pl.col("geocode").is_not_null())
        .unique()
        .sort(by="geocode")
        # Create center point from geocode
        .with_columns(
            center_xy=polars_h3.cell_to_latlng(pl.col("geocode")).list.reverse()
        )
        .with_columns(
            center=pl_st.point("center_xy"),
        )
        # Get boundary coordinates from H3 (returns list[list[f64]] in lat,lng order)
        .with_columns(_boundary_coords=polars_h3.cell_to_boundary("geocode"))
        .with_columns(
            _boundary_coords_lnglat=pl.col("_boundary_coords").list.eval(
                pl.element().list.reverse()
            )
        )
        # Close the polygon ring by appending the first point
        .with_columns(
            _boundary_coords_closed=pl.col("_boundary_coords_lnglat").list.concat(
                pl.col("_boundary_coords_lnglat").list.slice(0, 1)
            )
        )
        # Add row index for wrapping operation
        .with_row_index("_row_idx")
        # Wrap in another list for polygon format: list[list[list[f64]]]
        .with_columns(
            _boundary_ring=pl.col("_boundary_coords_closed").implode().over("_row_idx")
        )
        # Create polygon geometry from coordinates
        .with_columns(boundary=pl_st.polygon("_boundary_ring"))
        # Create bbox boundary linestring and check intersection
        .with_columns(_bbox_boundary=pl_st.linestring(pl.lit(bbox_coords)))
        .with_columns(
            is_edge=pl_st.geom("boundary").st.intersects(pl_st.geom("_bbox_boundary"))
        )
        # Clean up intermediate columns
        .drop(
            "center_xy",
            "_boundary_coords",
            "_boundary_coords_lnglat",
            "_boundary_coords_closed",
            "_row_idx",
            "_boundary_ring",
            "_bbox_boundary",
        )
        # Select final columns in schema order
        .select(list(GeocodeSchema.columns().keys()))
    )

    return GeocodeSchema.validate(lf, eager=False)


def build_geocode_df(
    darwin_core_lf: dy.LazyFrame[DarwinCoreSchema],
    geocode_precision: int,
    bounding_box: Bbox,
) -> dy.DataFrame[GeocodeSchema]:
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
    return GeocodeSchema.validate(lf.collect())


def build_geocode_no_edges_lf(
    geocode_lf: dy.LazyFrame[GeocodeSchema],
) -> dy.LazyFrame[GeocodeNoEdgesSchema]:
    """Build a GeocodeNoEdgesSchema by filtering out edge hexagons from a GeocodeSchema.

    This function is fully lazy - no collection occurs until downstream code
    explicitly collects the result.

    Args:
        geocode_lf: A validated GeocodeSchema lazy dataframe

    Returns:
        A validated GeocodeNoEdgesSchema lazyframe with edge hexagons removed
    """
    lf = geocode_lf.filter(~pl.col("is_edge")).sort(by="geocode")
    return GeocodeNoEdgesSchema.validate(lf, eager=False)


def index_of_geocode(
    geocode: int,
    geocode_df: dy.DataFrame[GeocodeSchema] | dy.DataFrame[GeocodeNoEdgesSchema],
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
