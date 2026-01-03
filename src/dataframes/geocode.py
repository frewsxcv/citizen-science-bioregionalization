import logging

import dataframely as dy
import polars as pl
import polars_h3
import polars_st as pl_st
import shapely
from shapely.geometry import box

from src.dataframes.darwin_core import DarwinCoreSchema
from src.geocode import select_geocode_lf
from src.types import Bbox

logger = logging.getLogger(__name__)


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


def build_geocode_df(
    darwin_core_lf: dy.LazyFrame[DarwinCoreSchema],
    geocode_precision: int,
    bounding_box: Bbox,
) -> dy.DataFrame[GeocodeSchema]:
    """Build a GeocodeSchema DataFrame from Darwin Core occurrence data.

    Args:
        darwin_core_lf: LazyFrame of Darwin Core occurrence records
        geocode_precision: H3 resolution (0-15). Higher = smaller hexagons.
        bounding_box: Geographic bounding box to determine edge hexagons

    Returns:
        A validated DataFrame conforming to GeocodeSchema
    """
    df = (
        darwin_core_lf.pipe(select_geocode_lf, geocode_precision=geocode_precision)
        .filter(pl.col("geocode").is_not_null())
        .unique()
        .sort(by="geocode")
        .with_columns(
            center_xy=polars_h3.cell_to_latlng(pl.col("geocode")).list.reverse()
        )
        .with_columns(
            center=pl_st.point("center_xy"),
        )
        .collect(engine="streaming")
    )

    # Create boundaries for all hexagons
    boundaries: list[shapely.Polygon] = []

    for geocode, geometry in df.select(
        "geocode",
        polars_h3.cell_to_boundary("geocode").alias("geometry"),
    ).iter_rows():
        boundary = shapely.Polygon(latlng_list_to_lnglat_list(geometry))
        boundaries.append(boundary)

    # Use provided bounding box to determine edge hexagons
    logger.info(
        f"Using provided bounding box: lat=[{bounding_box.min_lat:.4f}, {bounding_box.max_lat:.4f}], "
        f"lng=[{bounding_box.min_lng:.4f}, {bounding_box.max_lng:.4f}]"
    )

    # Create bounding box boundary (the edges, not the filled box)
    bbox_boundary = box(
        bounding_box.min_lng,
        bounding_box.min_lat,
        bounding_box.max_lng,
        bounding_box.max_lat,
    ).boundary

    # Check which hexagons intersect the bounding box edges
    is_edge_list: list[bool] = []
    for boundary in boundaries:
        intersects_edge = boundary.intersects(bbox_boundary)
        if intersects_edge is None:
            raise ValueError(
                f"boundary.intersects() returned None for boundary: {boundary}"
            )
        is_edge_list.append(intersects_edge)

    df = df.with_columns(
        boundary=pl_st.from_shapely(pl.Series(boundaries)),
        is_edge=pl.Series(is_edge_list),
    )

    initial_count = len(df)
    edge_count = df.filter(pl.col("is_edge")).shape[0]
    logger.info(
        f"Identified {edge_count} edge hexagons ({edge_count / initial_count * 100:.1f}%)"
    )

    df = df.select(list(GeocodeSchema.columns().keys()))
    return GeocodeSchema.validate(df)


def build_geocode_no_edges_df(
    geocode_lf: dy.LazyFrame[GeocodeSchema],
) -> dy.DataFrame[GeocodeNoEdgesSchema]:
    """Build a GeocodeNoEdgesSchema by filtering out edge hexagons from a GeocodeSchema.

    Args:
        geocode_lf: A validated GeocodeSchema lazy dataframe

    Returns:
        A validated GeocodeNoEdgesSchema dataframe with edge hexagons removed
    """
    # Get the set of edge geocodes to remove by collecting just those rows
    edge_geocodes = set(
        geocode_lf.filter(pl.col("is_edge"))
        .select("geocode")
        .collect(engine="streaming")["geocode"]
        .to_list()
    )

    logger.info(f"Removing {len(edge_geocodes)} edge hexagons from dataset")

    # Filter out edge hexagons and collect
    df = geocode_lf.filter(~pl.col("is_edge")).collect(engine="streaming")

    df = df.sort(by="geocode")

    logger.info(f"GeocodeNoEdgesSchema contains {len(df)} hexagons (all non-edge)")

    return GeocodeNoEdgesSchema.validate(df)


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


def latlng_list_to_lnglat_list(
    latlng_list: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Convert a list of (lat, lng) tuples to (lng, lat) tuples.

    H3 returns coordinates in (lat, lng) order, but most GIS tools
    expect (lng, lat) order.
    """
    return [(lng, lat) for (lat, lng) in latlng_list]
