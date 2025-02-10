import logging

from scipy.cluster.hierarchy import linkage, fcluster
from src.dataframes import geohash_species_counts
from src.dataframes.geohash_cluster import GeohashClusterDataFrame
from src.distance_matrix import DistanceMatrix

logger = logging.getLogger(__name__)


def run(
    geohash_taxa_counts_dataframe: geohash_species_counts.GeohashSpeciesCountsDataFrame,
    distance_matrix: DistanceMatrix,
    num_clusters: int,
) -> GeohashClusterDataFrame:
    ordered_seen_geohash = geohash_taxa_counts_dataframe.ordered_geohashes()
    Z = linkage(distance_matrix.condensed(), "ward")
    clusters = list(map(int, fcluster(Z, t=num_clusters, criterion="maxclust")))
    return GeohashClusterDataFrame.from_lists(ordered_seen_geohash, clusters)
