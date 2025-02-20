import polars as pl
from src import geohash
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from src.darwin_core import TaxonRank
from src.dataframes.geohash_species_counts import GeohashSpeciesCountsDataFrame
from typing import List
from src.dataframes.geohash_cluster import GeohashClusterDataFrame
import logging

logger = logging.getLogger(__name__)


# TODO: Refactor this to use the new ClusterSignificantDifferencesDataFrame


def print_cluster_stats(
    cluster: int,
    geohashes: List[geohash.Geohash],
    all_stats: ClusterTaxaStatisticsDataFrame,
) -> None:
    # stats = Stats.build(darwin_core_aggregations, geohash_filter=geohashes)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geohashes)})")

    for kingdom, species, count, average in (
        all_stats.df.filter(
            pl.col("cluster") == cluster,
            pl.col("rank") == TaxonRank.species,
        )
        .sort(by="count", descending=True)
        .limit(20)
        .select(["kingdom", "name", "count", "average"])
        .iter_rows(named=False)
    ):
        all_average = (
            all_stats.df.filter(
                pl.col("kingdom") == kingdom,
                pl.col("name") == species,
                pl.col("cluster").is_null(),
                pl.col("rank") == TaxonRank.species,
            )
            .get_column("average")
            .item()
        )

        # If the difference between the average of the cluster and the average of all is greater than 20%, print it
        percent_diff = (average / all_average * 100) - 100
        if abs(percent_diff) > 10:
            # Print the percentage difference
            print(f"{species} ({kingdom}):")
            print(
                f"  - Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%"
            )
            print(f"  - Proportion: {average * 100:.2f}%")
            print(f"  - Count: {count}")


def print_all_cluster_stats(all_stats: ClusterTaxaStatisticsDataFrame) -> None:
    for kingdom, species, count, average in (
        all_stats.df.filter(
            pl.col("cluster").is_null(),
            pl.col("rank") == TaxonRank.species,
        )
        .sort(by="count", descending=True)
        .limit(5)
        .select(["kingdom", "name", "count", "average"])
        .iter_rows(named=False)
    ):
        print(f"{species} ({kingdom}):")
        print(f"  - Proportion: {average * 100:.2f}%")
        print(f"  - Count: {count}")


def print_results(
    all_stats: ClusterTaxaStatisticsDataFrame,
    geohash_cluster_dataframe: GeohashClusterDataFrame,
) -> None:
    # For each top count taxon, print the average per geohash
    print_all_cluster_stats(all_stats)

    logger.info(f"Number of clusters: {geohash_cluster_dataframe.num_clusters()}")

    for cluster, geohashes in geohash_cluster_dataframe.iter_clusters_and_geohashes():
        print_cluster_stats(
            cluster,
            geohashes,
            all_stats,
        )
