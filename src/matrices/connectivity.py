import geohashr
import numpy as np
from typing import Self
from src.series.geohash import GeohashSeries
from src.data_container import DataContainer


class ConnectivityMatrix(DataContainer):
    _connectivity_matrix: np.ndarray

    def __init__(self, connectivity_matrix: np.ndarray):
        self._connectivity_matrix = connectivity_matrix

    @classmethod
    def build(
        cls, geohash_series: GeohashSeries
    ) -> Self:
        ordered_geohashes = geohash_series.series.to_list()

        # Step 1: Create a dictionary mapping each geohash to its neighbors
        geohash_neighbors = {
            gh: set(geohashr.neighbors(gh).values()) for gh in ordered_geohashes
        }

        # Step 2: Construct a connectivity matrix
        num_geohashes = len(ordered_geohashes)
        connectivity_matrix = np.zeros((num_geohashes, num_geohashes), dtype=int)

        for i, geoh1 in enumerate(ordered_geohashes):
            for j, geoh2 in enumerate(ordered_geohashes):
                if (
                    i != j and geoh2 in geohash_neighbors[geoh1]
                ):  # Check if geoh2 is a neighbor of geoh1
                    connectivity_matrix[i, j] = 1

        return cls(connectivity_matrix)
