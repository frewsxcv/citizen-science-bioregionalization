import polars as pl
import polars_st as pl_st
import logging
from typing import Self
import dataframely as dy

from shapely import MultiPoint, Point
import shapely.ops
from src.geocode import geocode_lazy_frame
from polars_darwin_core import DarwinCoreLazyFrame
import polars_h3
import networkx as nx

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

    @classmethod
    def build(
        cls,
        darwin_core_lazy_frame: DarwinCoreLazyFrame,
        geocode_precision: int,
    ) -> dy.DataFrame["GeocodeSchema"]:
        df = (
            darwin_core_lazy_frame._inner.pipe(
                geocode_lazy_frame, geocode_precision=geocode_precision
            )
            .filter(pl.col("geocode").is_not_null())
            .select("geocode")
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

        df = _reduce_connected_components_to_one(df)

        boundaries: list[shapely.Polygon] = []

        for geocode, geometry in df.select(
            "geocode",
            polars_h3.cell_to_boundary("geocode").alias("geometry"),
        ).iter_rows():
            boundaries.append(shapely.Polygon(latlng_list_to_lnglat_list(geometry)))

        df = df.with_columns(boundary=pl_st.from_shapely(pl.Series(boundaries))).select(
            list(GeocodeSchema.columns().keys())
        )
        return GeocodeSchema.validate(df)


def graph(
    geocode_dataframe: dy.DataFrame[GeocodeSchema], include_indirect_neighbors: bool = False
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
    geocode: int, geocode_dataframe: dy.DataFrame[GeocodeSchema]
) -> int:
    index = geocode_dataframe["geocode"].index_of(geocode)
    if index is None:
        raise ValueError(f"Geocode {geocode} not found in GeocodeDataFrame")
    return index


def latlng_list_to_lnglat_list(
    latlng_list: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [(lng, lat) for lat, lng in latlng_list]
