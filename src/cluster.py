import logging
import os

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
from scipy.cluster.hierarchy import linkage, fcluster
from src import dendrogram
from src.dataframes import geohash_species_counts
from src.dataframes.geohash_cluster import GeohashClusterDataFrame
from src.distance_matrix import DistanceMatrix
from src.logging import log_action

logger = logging.getLogger(__name__)


def run(
    geohash_taxa_counts_dataframe: geohash_species_counts.GeohashSpeciesCountsDataFrame,
    distance_matrix: DistanceMatrix,
    num_clusters: int,
    show_dendrogram_opt: bool,
) -> GeohashClusterDataFrame:
    ordered_seen_geohash = geohash_taxa_counts_dataframe.ordered_geohashes()
    Z = linkage(distance_matrix.condensed(), "ward")

    clusters = list(map(int, fcluster(Z, t=num_clusters, criterion="maxclust")))

    if show_dendrogram_opt:
        dendrogram.show(Z, ordered_seen_geohash)

    return GeohashClusterDataFrame.from_lists(ordered_seen_geohash, clusters)
