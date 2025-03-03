import numpy as np
from typing import Self

from src.dataframes.geocode import (
    GeocodeDataFrame,
    index_of_geocode_in_geocode_dataframe,
)
from src.data_container import DataContainer
import networkx as nx
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)


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

        assert_one_connected_component(connectivity_matrix)

        return cls(connectivity_matrix)


def assert_one_connected_component(connectivity_matrix: np.ndarray):
    assert (
        nx.number_connected_components(
            nx.from_numpy_array(
                connectivity_matrix,
            )
        )
        == 1
    )
