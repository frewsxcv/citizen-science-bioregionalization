import logging

import dataframely as dy
import networkx as nx
import numpy as np

from src.dataframes.geocode_neighbors import GeocodeNeighborsSchema

logger = logging.getLogger(__name__)


class GeocodeConnectivityMatrix:
    _connectivity_matrix: np.ndarray

    def __init__(self, connectivity_matrix: np.ndarray):
        self._connectivity_matrix = connectivity_matrix

    @classmethod
    def build(
        cls, geocode_neighbors_df: dy.DataFrame[GeocodeNeighborsSchema]
    ) -> "GeocodeConnectivityMatrix":
        """Build a connectivity matrix from geocode neighbor relationships.

        Args:
            geocode_neighbors_df: DataFrame containing geocode neighbor information

        Returns:
            GeocodeConnectivityMatrix with spatial adjacency constraints
        """
        num_geocodes = len(geocode_neighbors_df)
        connectivity_matrix = np.zeros((num_geocodes, num_geocodes), dtype=int)

        # Build geocode to index mapping
        geocode_to_index = {
            geocode: i for i, geocode in enumerate(geocode_neighbors_df["geocode"])
        }

        for i, neighbors in enumerate(
            geocode_neighbors_df["direct_and_indirect_neighbors"]
        ):
            for neighbor in neighbors:
                j = geocode_to_index.get(neighbor)
                if j is not None:
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
