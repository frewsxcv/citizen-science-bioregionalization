import polars as pl
import numpy as np
from typing import Self

from shapely import MultiPoint
from src.dataframes.geocode import GeocodeDataFrame
from src.data_container import DataContainer
import networkx as nx
from shapely.geometry import Point
import shapely.ops


class ConnectivityMatrix(DataContainer):
    _connectivity_matrix: np.ndarray

    def __init__(self, connectivity_matrix: np.ndarray):
        self._connectivity_matrix = connectivity_matrix

    @classmethod
    def build(cls, geocode_dataframe: GeocodeDataFrame) -> Self:
        num_geocodees = len(geocode_dataframe.df)
        connectivity_matrix = np.zeros((num_geocodees, num_geocodees), dtype=int)

        for i, neighbors in enumerate(geocode_dataframe.df["neighbors"]):
            for neighbor in neighbors:
                j = index_of_geocode_in_geocode_dataframe(neighbor, geocode_dataframe)
                connectivity_matrix[i, j] = 1

        graph = nx.from_numpy_array(
            connectivity_matrix,
            nodelist=geocode_dataframe.df["geocode"].to_list(),
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
                geocode_dataframe.df.select("center", "geocode")
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
                geocode_dataframe.df
                # Filter out nodes that are not on the edge of the grid
                .filter(pl.col("neighbors").list.len() != 8)
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

            geocode1: None | str = None
            for i, node in enumerate(first_component_points.geoms):
                if node.equals_exact(p1, 1e-6):
                    geocode1 = first_component_nodes[i][1]
                    break

            geocode2: None | str = None
            for i, node in enumerate(other_component_points):
                if node.equals_exact(p2, 1e-6):
                    geocode2 = other_component_nodes[i][1]
                    break

            if geocode1 is None or geocode2 is None:
                raise ValueError("No closest pair found")

            # Add edge between the closest nodes in both the graph and connectivity matrix
            print(f"Adding edge between {geocode1} and {geocode2}")
            geocode1_idx = index_of_geocode_in_geocode_dataframe(
                geocode1, geocode_dataframe
            )
            geocode2_idx = index_of_geocode_in_geocode_dataframe(
                geocode2, geocode_dataframe
            )
            graph.add_edge(geocode1, geocode2)
            connectivity_matrix[geocode1_idx, geocode2_idx] = 1
            connectivity_matrix[geocode2_idx, geocode1_idx] = 1  # Ensure symmetry

            number_of_connected_components = nx.number_connected_components(graph)

        return cls(connectivity_matrix)


def index_of_geocode_in_geocode_dataframe(
    geocode: str, geocode_dataframe: GeocodeDataFrame
) -> int:
    index = geocode_dataframe.df["geocode"].index_of(geocode)
    if index is None:
        raise ValueError(f"Geocode {geocode} not found in GeocodeDataFrame")
    return index
