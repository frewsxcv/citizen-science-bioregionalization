import logging
from typing import Union

import dataframely as dy
import networkx as nx
import polars as pl
import polars_h3
import polars_st as pl_st
import shapely.ops
from polars_darwin_core import DarwinCoreLazyFrame
from shapely import MultiPoint
from shapely.geometry import box

from src.geocode import geocode_lazy_frame

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
        darwin_core_lazy_frame: DarwinCoreLazyFrame,
        geocode_precision: int,
    ) -> dy.DataFrame["GeocodeSchema"]:
        # First, get unique geocodes
        geocoded_lf = darwin_core_lazy_frame._inner.pipe(
            geocode_lazy_frame, geocode_precision=geocode_precision
        ).filter(pl.col("geocode").is_not_null())

        df = (
            geocoded_lf.select("geocode")
            .unique()
            .sort(by="geocode")
            .with_columns(
                center_xy=polars_h3.cell_to_latlng(pl.col("geocode")).list.reverse()
            )
            .with_columns(
                center=pl_st.point("center_xy"),
            )
            .collect()
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
                direct_neighbors=pl.col(
                    "direct_neighbors_including_unknown"
                ).list.set_intersection(pl.col("known_geocodes")),
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

        # Calculate the bounding box from the hexagon centers
        centers = df.select(pl_st.geom("center").st.to_shapely()).to_series().to_list()

        min_lng = min(center.x for center in centers)
        max_lng = max(center.x for center in centers)
        min_lat = min(center.y for center in centers)
        max_lat = max(center.y for center in centers)

        logger.info(
            f"Hexagon center extents: lat=[{min_lat:.4f}, {max_lat:.4f}], "
            f"lng=[{min_lng:.4f}, {max_lng:.4f}]"
        )

        # Create bounding box boundary (the edges, not the filled box)
        bbox_boundary = box(min_lng, min_lat, max_lng, max_lat).boundary

        # Check which hexagons intersect the bounding box edges
        is_edge_list: list[bool] = []
        for boundary in boundaries:
            intersects_edge = boundary.intersects(bbox_boundary)
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
        geocode_dataframe: dy.DataFrame[GeocodeSchema],
    ) -> dy.DataFrame["GeocodeNoEdgesSchema"]:
        """Create a GeocodeNoEdgesSchema by filtering out edge hexagons from a GeocodeSchema.

        Args:
            geocode_dataframe: A validated GeocodeSchema dataframe

        Returns:
            A validated GeocodeNoEdgesSchema dataframe with edge hexagons removed
            and neighbor lists updated to exclude the removed edges.
        """
        df = geocode_dataframe.clone()

        # Get the set of edge geocodes to remove
        edge_geocodes = set(df.filter(pl.col("is_edge"))["geocode"].to_list())

        logger.info(f"Removing {len(edge_geocodes)} edge hexagons from dataset")

        # Filter out edge hexagons
        df = df.filter(~pl.col("is_edge"))

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


def _reduce_connected_components_to_one(df: pl.DataFrame) -> pl.DataFrame:
    graph = _df_to_graph(df)
    number_of_connected_components = nx.number_connected_components(graph)
    while number_of_connected_components > 1:
        logger.info(
            f"More than one connected component (n={number_of_connected_components}), connecting the first with the closest node not in that component"
        )
        components = nx.connected_components(graph)
        first_component: set[int] = next(components)

        first_component_nodes = list(
            df.select(
                pl_st.geom("center").st.to_shapely(),
                "geocode",
            )
            .filter(pl.col("geocode").is_in(first_component))
            .iter_rows()
        )
        first_component_points = [center1 for center1, _ in first_component_nodes]

        other_component_nodes = list(
            df.filter(pl.col("direct_neighbors").list.len() != MAX_NUM_NEIGHBORS)
            .filter(pl.col("geocode").is_in(first_component).not_())
            .select(
                pl_st.geom("center").st.to_shapely(),
                "geocode",
            )
            .iter_rows()
        )
        other_component_points = [center2 for center2, _ in other_component_nodes]

        p1, p2 = shapely.ops.nearest_points(
            MultiPoint(first_component_points),
            MultiPoint(other_component_points),
        )

        geocode1 = None
        for i, node in enumerate(first_component_points):
            if node.equals_exact(p1, 1e-6):
                geocode1 = first_component_nodes[i][1]
                break

        geocode2 = None
        for i, node in enumerate(other_component_points):
            if node.equals_exact(p2, 1e-6):
                geocode2 = other_component_nodes[i][1]
                break

        if geocode1 is None or geocode2 is None:
            raise ValueError("No closest pair found")

        # Add edge between the closest nodes in both the graph and connectivity matrix
        logger.info(f"Adding edge between {geocode1} and {geocode2}")
        graph.add_edge(geocode1, geocode2)
        df = df.with_columns(
            pl.when(pl.col("geocode") == geocode1)
            .then(
                pl.concat_list(
                    [
                        pl.col("direct_and_indirect_neighbors"),
                        pl.lit([geocode2], dtype=pl.List(pl.UInt64)),
                    ]
                )
            )
            .when(pl.col("geocode") == geocode2)
            .then(
                pl.concat_list(
                    [
                        pl.col("direct_and_indirect_neighbors"),
                        pl.lit([geocode1], dtype=pl.List(pl.UInt64)),
                    ]
                )
            )
            .otherwise(pl.col("direct_and_indirect_neighbors"))
            .alias("direct_and_indirect_neighbors")
        )

        number_of_connected_components = nx.number_connected_components(graph)

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
    return [(lng, lat) for lat, lng in latlng_list]
