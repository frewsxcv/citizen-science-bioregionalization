import logging

import dataframely as dy
import networkx as nx
import numpy as np
import polars as pl

from src.dataframes.geocode import (
    GeocodeNoEdgesSchema,
    index_of_geocode,
)

logger = logging.getLogger(__name__)


class GeocodeConnectivityMatrix:
    _connectivity_matrix: np.ndarray

    def __init__(self, connectivity_matrix: np.ndarray):
        self._connectivity_matrix = connectivity_matrix

    @classmethod
    def build(
        cls, geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema]
    ) -> "GeocodeConnectivityMatrix":
        # Collect the LazyFrame once at the start
        geocode_df: dy.DataFrame[GeocodeNoEdgesSchema] = geocode_lf.collect()

        num_geocodes = len(geocode_df)
        connectivity_matrix = np.zeros((num_geocodes, num_geocodes), dtype=int)

        for i, neighbors in enumerate(geocode_df["direct_and_indirect_neighbors"]):
            for neighbor in neighbors:
                j = index_of_geocode(neighbor, geocode_df)
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
