import polars as pl
import numpy as np
from typing import Self

from shapely import MultiPoint
from src.dataframes.geohash import GeohashDataFrame
from src.data_container import DataContainer
import networkx as nx
from shapely.geometry import Point
import shapely.ops


class ConnectivityMatrix(DataContainer):
    _connectivity_matrix: np.ndarray

    def __init__(self, connectivity_matrix: np.ndarray):
        self._connectivity_matrix = connectivity_matrix

    @classmethod
    def build(cls, geohash_dataframe: GeohashDataFrame) -> Self:
        num_geohashes = len(geohash_dataframe.df)
        connectivity_matrix = np.zeros((num_geohashes, num_geohashes), dtype=int)

        for i, neighbors in enumerate(geohash_dataframe.df["neighbors"]):
            for neighbor in neighbors:
                j = index_of_geohash_in_geohash_dataframe(neighbor, geohash_dataframe)
                connectivity_matrix[i, j] = 1

        graph = nx.from_numpy_array(
            connectivity_matrix,
            nodelist=geohash_dataframe.df["geohash"].to_list(),
        )

        # As long as there are more than one connected component, connect the first with the closest node not in that component
        number_of_connected_components = nx.number_connected_components(graph)
        while number_of_connected_components > 1:
            print(
                f"More than one connected component (n={number_of_connected_components}), connecting the first with the closest node not in that component"
            )
            components = nx.connected_components(graph)
            first_component: set[str] = next(components)

            first_component_nodes = list(
                geohash_dataframe.df.select("center", "geohash")
                .filter(pl.col("geohash").is_in(first_component))
                .iter_rows()
            )
            first_component_points = MultiPoint(
                [
                    Point(center1["lon"], center1["lat"])
                    for center1, _ in first_component_nodes
                ]
            )

            other_component_nodes = list(
                geohash_dataframe.df
                # Filter out nodes that are not on the edge of the grid
                .filter(pl.col("neighbors").list.len() != 8)
                .filter(pl.col("geohash").is_in(first_component).not_())
                .select("center", "geohash")
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

            geohash1: None | str = None
            for i, node in enumerate(first_component_points.geoms):
                if node.equals_exact(p1, 1e-6):
                    geohash1 = first_component_nodes[i][1]
                    break

            geohash2: None | str = None
            for i, node in enumerate(other_component_points):
                if node.equals_exact(p2, 1e-6):
                    geohash2 = other_component_nodes[i][1]
                    break

            if geohash1 is None or geohash2 is None:
                raise ValueError("No closest pair found")

            # Add edge between the closest nodes in both the graph and connectivity matrix
            print(f"Adding edge between {geohash1} and {geohash2}")
            geohash1_idx = index_of_geohash_in_geohash_dataframe(
                geohash1, geohash_dataframe
            )
            geohash2_idx = index_of_geohash_in_geohash_dataframe(
                geohash2, geohash_dataframe
            )
            graph.add_edge(geohash1, geohash2)
            connectivity_matrix[geohash1_idx, geohash2_idx] = 1
            connectivity_matrix[geohash2_idx, geohash1_idx] = 1  # Ensure symmetry

            number_of_connected_components = nx.number_connected_components(graph)

        return cls(connectivity_matrix)


def index_of_geohash_in_geohash_dataframe(
    geohash: str, geohash_dataframe: GeohashDataFrame
) -> int:
    index = geohash_dataframe.df["geohash"].index_of(geohash)
    if index is None:
        raise ValueError(f"Geohash {geohash} not found in GeohashDataFrame")
    return index
