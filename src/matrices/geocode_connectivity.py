import logging

import dataframely as dy
import numpy as np

import bioregion_rs
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
        connectivity_matrix = np.array(
            bioregion_rs.build_geocode_connectivity_matrix(geocode_neighbors_df)
        )
        return cls(connectivity_matrix)
