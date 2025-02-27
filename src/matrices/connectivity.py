import geohashr
import numpy as np
from typing import Self
from src.dataframes.geohash import GeohashDataFrame
from src.data_container import DataContainer


class ConnectivityMatrix(DataContainer):
    _connectivity_matrix: np.ndarray

    def __init__(self, connectivity_matrix: np.ndarray):
        self._connectivity_matrix = connectivity_matrix

    @classmethod
    def build(
        cls, geohash_dataframe: GeohashDataFrame
    ) -> Self:
        num_geohashes = len(geohash_dataframe.df)
        connectivity_matrix = np.zeros((num_geohashes, num_geohashes), dtype=int)

        for i, neighbors in enumerate(geohash_dataframe.df["neighbors"]):
            for neighbor in neighbors:
                j = geohash_dataframe.df["geohash"].index_of(neighbor)
                if j is None:
                    raise ValueError(f"Neighbor {neighbor} not found in GeohashDataFrame")
                connectivity_matrix[i, j] = 1

        return cls(connectivity_matrix)
