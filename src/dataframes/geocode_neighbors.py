import logging
from typing import Union

import networkx as nx
import polars as pl
import polars_h3
import polars_st as pl_st
import shapely
import shapely.ops
from shapely import MultiPoint

import bioregion_rs

logger = logging.getLogger(__name__)

MAX_NUM_NEIGHBORS = 6

def build_geocode_neighbors_df(
    geocode_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build neighbor relationships for geocodes.

    Computes direct neighbors using H3 grid adjacency, then adds indirect
    neighbors as needed to ensure a single connected component.

    Args:
        geocode_df: DataFrame with 'geocode' and 'center' columns

    Returns:
        A validated DataFrame conforming to GeocodeNeighborsSchema
    """
    logger.info(
        f"build_geocode_neighbors_df: Building neighbors for {geocode_df.height} geocodes"
    )
    df = bioregion_rs.build_geocode_neighbors(geocode_df.select("geocode", "center"))
    return df


def build_geocode_neighbors_no_edges_df(
    geocode_neighbors_df: pl.DataFrame,
    geocode_no_edges_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build neighbor relationships for non-edge geocodes.

    Filters the neighbor lists to only include valid (non-edge) geocodes,
    then reconnects any disconnected components.

    Args:
        geocode_neighbors_df: DataFrame with neighbor relationships for all geocodes
        geocode_no_edges_df: DataFrame with geocodes that are not on edges
            (must have 'geocode' and 'center' columns)

    Returns:
        A validated DataFrame conforming to GeocodeNeighborsSchema
    """
    df = bioregion_rs.build_geocode_neighbors_no_edges(
        geocode_neighbors_df, geocode_no_edges_df.select("geocode", "center")
    )

    logger.info(f"GeocodeNeighborsSchema (no edges) contains {len(df)} geocodes")

    return df


def graph(
    geocode_neighbors_df: pl.DataFrame,
    include_indirect_neighbors: bool = False,
) -> nx.Graph:
    """Convert geocode neighbors to a NetworkX graph.

    Args:
        geocode_neighbors_df: DataFrame with neighbor relationships
        include_indirect_neighbors: If True, use direct_and_indirect_neighbors;
            otherwise use only direct_neighbors

    Returns:
        NetworkX graph with geocodes as nodes and neighbor relationships as edges
    """
    return _df_to_graph(geocode_neighbors_df, include_indirect_neighbors)


def _df_to_graph(
    df: pl.DataFrame, include_indirect_neighbors: bool = False
) -> nx.Graph:
    """Internal function to convert a DataFrame to a NetworkX graph."""
    g: nx.Graph[int] = nx.Graph()
    for geocode in df["geocode"]:
        g.add_node(geocode)
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
            g.add_edge(geocode, neighbor)
    return g


def _add_indirect_neighbor_edge(
    df: pl.DataFrame, geocode1: int, geocode2: int
) -> pl.DataFrame:
    """Add a bidirectional indirect neighbor connection between two geocodes."""
    return df.with_columns(
        pl.when(pl.col("geocode") == geocode1)
        .then(
            pl.col("direct_and_indirect_neighbors").list.concat(
                pl.lit([geocode2], dtype=pl.List(pl.UInt64))
            )
        )
        .when(pl.col("geocode") == geocode2)
        .then(
            pl.col("direct_and_indirect_neighbors").list.concat(
                pl.lit([geocode1], dtype=pl.List(pl.UInt64))
            )
        )
        .otherwise(pl.col("direct_and_indirect_neighbors"))
        .alias("direct_and_indirect_neighbors")
    )


def _reduce_connected_components_to_one(df: pl.DataFrame) -> pl.DataFrame:
    """Reduce multiple connected components to one by adding indirect edges.

    Uses spatial proximity to find the closest pair of nodes between
    disconnected components and adds an indirect edge between them.

    Args:
        df: DataFrame with 'geocode', 'center', 'direct_neighbors', and
            'direct_and_indirect_neighbors' columns

    Returns:
        DataFrame with indirect edges added to ensure single connected component
    """
    g = _df_to_graph(df)

    while nx.number_connected_components(g) > 1:
        num_components = nx.number_connected_components(g)
        logger.info(
            f"More than one connected component (n={num_components}), connecting the first with the closest node not in that component"
        )

        first_component: set[int] = next(nx.connected_components(g))

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
        g.add_edge(geocode1, geocode2)
        df = _add_indirect_neighbor_edge(df, geocode1, geocode2)

    return df
