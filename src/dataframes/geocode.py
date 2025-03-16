import polars as pl
import logging
from shapely import MultiPoint, Point
import shapely.ops
from src.data_container import DataContainer, assert_dataframe_schema
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
import polars_h3
import networkx as nx

logger = logging.getLogger(__name__)

MAX_NUM_NEIGHBORS = 6


class GeocodeDataFrame(DataContainer):
    """
    A dataframe of unique, in-order geocodes that are connected to another known geocode.
    """

    df: pl.DataFrame

    SCHEMA = {
        "geocode": pl.String(),
        "center": pl.Struct(
            {
                "lat": pl.Float64(),
                "lon": pl.Float64(),
            }
        ),
        # Direct neighbors from H3 grid adjacency
        "direct_neighbors": pl.List(pl.String()),
        # Direct and indirect neighbors (includes both H3 adjacency and added connections)
        "direct_and_indirect_neighbors": pl.List(pl.String()),
    }

    def __init__(self, df: pl.DataFrame):
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls,
        darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame,
        geocode_precision: int,
    ) -> 'GeocodeDataFrame':
        df = (
            darwin_core_csv_lazy_frame.lf.select("decimalLatitude", "decimalLongitude")
            .with_columns(
                polars_h3.latlng_to_cell(
                    "decimalLatitude",
                    "decimalLongitude",
                    resolution=geocode_precision,
                    return_dtype=pl.Utf8,
                ).alias("geocode"),
            )
            .select("geocode")
            .unique()
            .sort(by="geocode")
            .with_columns(
                polars_h3.cell_to_latlng(pl.col("geocode"))
                .list.to_struct(fields=["lat", "lon"])
                .alias("center")
            )
            .collect()
        )

        df = (
            df.with_columns(
                direct_neighbors_including_unknown=polars_h3.grid_ring(
                    pl.col("geocode"), 1
                ),
                known_geocodes=pl.lit(
                    df["geocode"].unique().to_list(), dtype=pl.List(pl.Utf8)
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

        return cls(_reduce_connected_components_to_one(df).select(cls.SCHEMA.keys()))

    def graph(self) -> nx.Graph:
        return _df_to_graph(self.df)


def _df_to_graph(
    df: pl.DataFrame, include_indirect_neighbors: bool = False
) -> nx.Graph:
    graph = nx.Graph()
    for geocode in df["geocode"]:
        graph.add_node(geocode)
    column = "direct_and_indirect_neighbors" if include_indirect_neighbors else "direct_neighbors"
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
        first_component: set[str] = next(components)

        first_component_nodes = list(
            df.select("center", "geocode")
            .filter(pl.col("geocode").is_in(first_component))
            .iter_rows()
        )
        first_component_points = MultiPoint(
            [
                Point(center1["lon"], center1["lat"])
                for center1, _ in first_component_nodes
            ]
        )

        other_component_nodes = list(
            df
            # Filter out nodes that are not on the edge of the grid
            .filter(pl.col("direct_neighbors").list.len() != MAX_NUM_NEIGHBORS)
            .filter(pl.col("geocode").is_in(first_component).not_())
            .select("center", "geocode")
            .iter_rows()
        )
        other_component_points = [
            Point(center2["lon"], center2["lat"])
            for center2, _ in other_component_nodes
        ]

        p1, p2 = shapely.ops.nearest_points(
            MultiPoint(first_component_points),
            MultiPoint(other_component_points),
        )

        geocode1 = None
        for i, node in enumerate(first_component_points.geoms):
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
                    [pl.col("direct_and_indirect_neighbors"), pl.lit(geocode2)]
                )
            )
            .when(pl.col("geocode") == geocode2)
            .then(
                pl.concat_list(
                    [pl.col("direct_and_indirect_neighbors"), pl.lit(geocode1)]
                )
            )
            .otherwise(pl.col("direct_and_indirect_neighbors"))
            .alias("direct_and_indirect_neighbors")
        )

        number_of_connected_components = nx.number_connected_components(graph)

    return df


def index_of_geocode_in_geocode_dataframe(
    geocode: str, geocode_dataframe: GeocodeDataFrame
) -> int:
    index = geocode_dataframe.df["geocode"].index_of(geocode)
    if index is None:
        raise ValueError(f"Geocode {geocode} not found in GeocodeDataFrame")
    return index
