import os
import numpy as np
import polars as pl
from typing import Tuple, List
from sklearn.preprocessing import RobustScaler  # type: ignore
from scipy.spatial.distance import pdist, squareform  # type: ignore
from sklearn.manifold import MDS  # type: ignore
from src.data_container import DataContainer
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.logging import log_action, logger


def pivot_taxon_counts_for_clusters(
    cluster_taxa_stats: ClusterTaxaStatisticsDataFrame,
) -> pl.DataFrame:
    """
    Create a matrix where each row is a cluster and each column is a taxon ID.
    The values represent the average occurrence of each taxon within the cluster.

    Example output:

    ```txt
    ┌─────────┬───────┬───────┬───────┬───┬───────┐
    │ cluster ┆ 12345 ┆ 23456 ┆ 34567 ┆ … ┆ 56789 │
    │ ---     ┆ ---   ┆ ---   ┆ ---   ┆   ┆ ---   │
    │ u32     ┆ f64   ┆ f64   ┆ f64   ┆   ┆ f64   │
    ╞═════════╪═══════╪═══════╪═══════╪═══╪═══════╡
    │ 1       ┆ 0.05  ┆ 0.02  ┆ 0.01  ┆ … ┆ 0.12  │
    │ 2       ┆ 0.03  ┆ 0.04  ┆ 0.07  ┆ … ┆ 0.01  │
    └─────────┴───────┴───────┴───────┴───┴───────┘
    ```
    """
    # Filter out the row with null cluster (represents overall statistics)
    df = cluster_taxa_stats.df.filter(pl.col("cluster").is_not_null())

    # Pivot the DataFrame so each row is a cluster and each column is a taxon
    return df.pivot(
        on="taxonId",
        index="cluster",
        values="average",
    )


def build_X(
    cluster_taxa_stats: ClusterTaxaStatisticsDataFrame,
) -> Tuple[pl.DataFrame, List[int]]:
    X = log_action(
        "Building cluster matrix",
        lambda: pivot_taxon_counts_for_clusters(cluster_taxa_stats),
    )

    assert X.height > 1, "More than one cluster is required to calculate distances"

    # fill null values with 0
    X = log_action("Filling null values", lambda: X.fill_null(0.0))

    # Keep a copy of cluster IDs before dropping the column
    cluster_ids = X["cluster"].to_list()

    X = log_action("Dropping cluster column", lambda: X.drop("cluster"))

    return log_action("Scaling values", lambda: X.pipe(scale_values)), cluster_ids


def scale_values(X: pl.DataFrame) -> pl.DataFrame:
    scaler = RobustScaler()
    return pl.from_numpy(scaler.fit_transform(X.to_numpy()))


class ClusterDistanceMatrix(DataContainer):
    """
    A distance matrix where each column and row is a cluster, and the cell at the intersection of a
    column and row is the similarity (or distance) between the two clusters based on their taxonomic
    composition. Internally it is stored as a condensed distance matrix, which is a one-dimensional
    array containing the upper triangular part of the distance matrix.
    """

    _condensed: np.ndarray
    _cluster_ids: List[int]

    def __init__(self, condensed: np.ndarray, cluster_ids: List[int]):
        self._condensed = condensed
        self._cluster_ids = cluster_ids

    @classmethod
    def build(
        cls,
        cluster_taxa_stats: ClusterTaxaStatisticsDataFrame,
    ) -> "ClusterDistanceMatrix":
        X, cluster_ids = build_X(cluster_taxa_stats)

        logger.info(
            f"Building cluster distance matrix: {X.shape[0]} clusters, {X.shape[1]} taxon IDs"
        )

        Y = log_action(
            f"Running pdist on cluster matrix",
            lambda: pdist(X, metric="braycurtis"),
        )

        # Replace any infinity values with 1.0 (maximum distance)
        Y = np.nan_to_num(Y, nan=1.0, posinf=1.0, neginf=1.0)

        return cls(Y, cluster_ids)

    def condensed(self) -> np.ndarray:
        return self._condensed

    def squareform(self) -> np.ndarray:
        return squareform(self._condensed)

    def cluster_ids(self) -> List[int]:
        """Return the cluster IDs in the order they appear in the distance matrix."""
        return self._cluster_ids

    def get_distance(self, cluster_id1: int, cluster_id2: int) -> float:
        """Get the distance between two clusters."""
        if cluster_id1 == cluster_id2:
            return 0.0

        # Find indices in cluster_ids
        idx1 = self._cluster_ids.index(cluster_id1)
        idx2 = self._cluster_ids.index(cluster_id2)

        # Get the distance from the square matrix
        square_matrix = self.squareform()
        distance = float(square_matrix[idx1, idx2])

        # Handle case where distance is infinity (clusters have no overlap in taxa)
        if np.isinf(distance) or np.isnan(distance):
            return 1.0

        return distance

    def get_most_similar_clusters(
        self, cluster_id: int, n: int = 3
    ) -> List[Tuple[int, float]]:
        """Get the n most similar clusters to the given cluster."""
        if cluster_id not in self._cluster_ids:
            raise ValueError(f"Cluster ID {cluster_id} not found in distance matrix")

        # Find index in cluster_ids
        idx = self._cluster_ids.index(cluster_id)

        # Get distances to all other clusters
        square_matrix = self.squareform()
        distances = square_matrix[idx]

        # Create a list of (cluster_id, distance) tuples, excluding the cluster itself
        cluster_distances = [
            (cid, float(distances[i]))
            for i, cid in enumerate(self._cluster_ids)
            if cid != cluster_id
        ]

        # Sort by distance (ascending) and take the top n
        return sorted(cluster_distances, key=lambda x: x[1])[:n]
