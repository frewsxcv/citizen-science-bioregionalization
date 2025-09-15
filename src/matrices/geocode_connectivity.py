import numpy as np

import dataframely as dy
from src.dataframes.geocode import (
    GeocodeSchema,
    index_of_geocode,
)
import networkx as nx
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)


class GeocodeConnectivityMatrix:
    _connectivity_matrix: np.ndarray

    def __init__(self, connectivity_matrix: np.ndarray):
        self._connectivity_matrix = connectivity_matrix

    @classmethod
    def build(
        cls, geocode_dataframe: dy.DataFrame[GeocodeSchema]
    ) -> "GeocodeConnectivityMatrix":
        num_geocodes = len(geocode_dataframe)
        connectivity_matrix = np.zeros((num_geocodes, num_geocodes), dtype=int)

        for i, neighbors in enumerate(
            geocode_dataframe["direct_and_indirect_neighbors"]
        ):
            for neighbor in neighbors:
                j = index_of_geocode(neighbor, geocode_dataframe)
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
