import logging
from typing import Union

import dataframely as dy
import networkx as nx
import polars as pl
import polars_h3
import polars_st as pl_st
import shapely.ops
from shapely import MultiPoint
from shapely.geometry import box

from src.geocode import select_geocode_lazy_frame
from src.types import Bbox, LatLng

logger = logging.getLogger(__name__)

MAX_NUM_NEIGHBORS = 6


class GeocodeSchema(dy.Schema):
    geocode = dy.UInt64(nullable=False)
    center = dy.Any()  # Binary
    boundary = dy.Any()  # Binary
    # Direct neighbors from H3 grid adjacency
    direct_neighbors = dy.List(dy.UInt64(), nullable=False)
    # Direct and indirect neighbors (includes both H3 adjacency and added connections)
    direct_and_indirect_neighbors = dy.List(dy.UInt64(), nullable=False)
    # Whether this hexagon is on the edge (intersects bounding box boundary)
    is_edge = dy.Bool(nullable=False)

    @classmethod
    def build(
        cls,
        darwin_core_lazy_frame: pl.LazyFrame,
        geocode_precision: int,
        bounding_box: Bbox,
    ) -> dy.DataFrame["GeocodeSchema"]:
        df = (
            darwin_core_lazy_frame.pipe(
                select_geocode_lazy_frame, geocode_precision=geocode_precision
            )
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

        # Calculate direct neighbors
        df = (
            df.with_columns(
                direct_neighbors_including_unknown=polars_h3.grid_ring(
                    pl.col("geocode"), 1
                ),
                known_geocodes=pl.lit(
                    df["geocode"].unique().to_list(), dtype=pl.List(pl.UInt64)
                ),
            )
            .with_columns(
                direct_neighbors=pl.col("direct_neighbors_including_unknown")
                .list.set_intersection(pl.col("known_geocodes"))
                .cast(pl.List(pl.UInt64)),
            )
            .with_columns(
                direct_and_indirect_neighbors=pl.col("direct_neighbors"),
            )
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

        df = _reduce_connected_components_to_one(df)

        df = df.select(list(GeocodeSchema.columns().keys()))
        return GeocodeSchema.validate(df)


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

    @classmethod
    def from_geocode_schema(
        cls,
        geocode_dataframe: dy.LazyFrame[GeocodeSchema],
    ) -> dy.DataFrame["GeocodeNoEdgesSchema"]:
        """Create a GeocodeNoEdgesSchema by filtering out edge hexagons from a GeocodeSchema.

        Args:
            geocode_dataframe: A validated GeocodeSchema lazy dataframe

        Returns:
            A validated GeocodeNoEdgesSchema dataframe with edge hexagons removed
            and neighbor lists updated to exclude the removed edges.
        """
        # Get the set of edge geocodes to remove by collecting just those rows
        edge_geocodes = set(
            geocode_dataframe.filter(pl.col("is_edge"))
            .select("geocode")
            .collect(engine="streaming")["geocode"]
            .to_list()
        )

        logger.info(f"Removing {len(edge_geocodes)} edge hexagons from dataset")

        # Filter out edge hexagons and collect to get valid geocodes
        df = geocode_dataframe.filter(~pl.col("is_edge")).collect(engine="streaming")

        # Update neighbor lists to remove references to edge geocodes
        # Get the set of remaining valid geocodes
        valid_geocodes = set(df["geocode"].to_list())
        valid_geocodes_lit = pl.lit(list(valid_geocodes), dtype=pl.List(pl.UInt64))

        df = df.with_columns(
            direct_neighbors=pl.col("direct_neighbors").list.set_intersection(
                valid_geocodes_lit
            ),
            direct_and_indirect_neighbors=pl.col(
                "direct_and_indirect_neighbors"
            ).list.set_intersection(valid_geocodes_lit),
        )

        # After removing edge geocodes, we may have disconnected components again
        # (e.g., if indirect connections went through edge geocodes)
        df = _reduce_connected_components_to_one(df)

        df = df.sort(by="geocode")

        logger.info(f"GeocodeNoEdgesSchema contains {len(df)} hexagons (all non-edge)")

        return cls.validate(df)


def graph(
    geocode_dataframe: Union[
        dy.DataFrame[GeocodeSchema], dy.DataFrame["GeocodeNoEdgesSchema"]
    ],
    include_indirect_neighbors: bool = False,
) -> nx.Graph:
    return _df_to_graph(geocode_dataframe, include_indirect_neighbors)


def _df_to_graph(
    df: pl.DataFrame, include_indirect_neighbors: bool = False
) -> nx.Graph:
    graph: nx.Graph[str] = nx.Graph()
    for geocode in df["geocode"]:
        graph.add_node(geocode)
    column = (
        "direct_and_indirect_neighbors"
        if include_indirect_neighbors
        else "direct_neighbors"
    )
    for geocode, neighbors in df.select(
        "geocode",
        column,
    ).iter_rows():
        for neighbor in neighbors:
            graph.add_edge(geocode, neighbor)
    return graph


def _add_indirect_neighbor_edge(
    df: pl.DataFrame, geocode1: int, geocode2: int
) -> pl.DataFrame:
    """Add a bidirectional indirect neighbor connection between two geocodes."""
    return df.with_columns(
        pl.when(pl.col("geocode") == geocode1)
        .then(pl.col("direct_and_indirect_neighbors").list.concat([geocode2]))
        .when(pl.col("geocode") == geocode2)
        .then(pl.col("direct_and_indirect_neighbors").list.concat([geocode1]))
        .otherwise(pl.col("direct_and_indirect_neighbors"))
        .cast(pl.List(pl.UInt64))
        .alias("direct_and_indirect_neighbors")
    )


def _reduce_connected_components_to_one(df: pl.DataFrame) -> pl.DataFrame:
    graph = _df_to_graph(df)

    while nx.number_connected_components(graph) > 1:
        num_components = nx.number_connected_components(graph)
        logger.info(
            f"More than one connected component (n={num_components}), connecting the first with the closest node not in that component"
        )

        first_component: set[int] = next(nx.connected_components(graph))

        # Build point-to-geocode mapping for efficient lookup
        all_nodes = list(
            df.select(
                pl_st.geom("center").st.to_shapely().alias("center"),
                "geocode",
                pl.col("direct_neighbors").list.len().alias("num_neighbors"),
            ).iter_rows()
        )
        point_to_geocode = {point.wkb: geocode for point, geocode, _ in all_nodes}

        first_component_points = [p for p, g, _ in all_nodes if g in first_component]
        other_component_points = [
            p
            for p, g, n in all_nodes
            if g not in first_component and n != MAX_NUM_NEIGHBORS
        ]

        p1, p2 = shapely.ops.nearest_points(
            MultiPoint(first_component_points),
            MultiPoint(other_component_points),
        )

        geocode1 = point_to_geocode.get(p1.wkb)
        geocode2 = point_to_geocode.get(p2.wkb)

        if geocode1 is None or geocode2 is None:
            raise ValueError("No closest pair found")

        logger.info(f"Adding edge between {geocode1} and {geocode2}")
        graph.add_edge(geocode1, geocode2)
        df = _add_indirect_neighbor_edge(df, geocode1, geocode2)

    return df


def index_of_geocode(
    geocode: int,
    geocode_dataframe: Union[
        dy.DataFrame[GeocodeSchema], dy.DataFrame["GeocodeNoEdgesSchema"]
    ],
) -> int:
    index = geocode_dataframe["geocode"].index_of(geocode)
    if index is None:
        raise ValueError(f"Geocode {geocode} not found in GeocodeDataFrame")
    return index


def latlng_list_to_lnglat_list(
    latlng_list: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [(lng, lat) for (lat, lng) in latlng_list]
