import polars as pl
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from typing import List
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
import logging

logger = logging.getLogger(__name__)


# TODO: Refactor this to use the new ClusterSignificantDifferencesDataFrame


def print_cluster_stats(
    cluster: int,
    geocodees: List[str],
    all_stats: ClusterTaxaStatisticsDataFrame,
) -> None:
    # stats = Stats.build(darwin_core_aggregations, geocode_filter=geocodees)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geocodees)})")

    for kingdom, taxonRank, scientificName, count, average in (
        all_stats.df.filter(
            pl.col("cluster") == cluster,
        )
        .sort(by="count", descending=True)
        .limit(20)
        .select(["kingdom", "taxonRank", "scientificName", "count", "average"])
        .iter_rows(named=False)
    ):
        all_average = (
            all_stats.df.filter(
                pl.col("kingdom") == kingdom,
                pl.col("scientificName") == scientificName,
                pl.col("taxonRank") == taxonRank,
                pl.col("cluster").is_null(),
            )
            .get_column("average")
            .item()
        )

        # If the difference between the average of the cluster and the average of all is greater than 20%, print it
        percent_diff = (average / all_average * 100) - 100
        if abs(percent_diff) > 10:
            # Print the percentage difference
            print(f"{scientificName} ({kingdom}) {taxonRank}:")
            print(
                f"  - Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%"
            )
            print(f"  - Proportion: {average * 100:.2f}%")
            print(f"  - Count: {count}")


def print_all_cluster_stats(all_stats: ClusterTaxaStatisticsDataFrame) -> None:
    for kingdom, taxonRank, scientificName, count, average in (
        all_stats.df.filter(
            pl.col("cluster").is_null(),
        )
        .sort(by="count", descending=True)
        .limit(5)
        .select(["kingdom", "taxonRank", "scientificName", "count", "average"])
        .iter_rows(named=False)
    ):
        print(f"{scientificName} ({kingdom}) {taxonRank}:")
        print(f"  - Proportion: {average * 100:.2f}%")
        print(f"  - Count: {count}")


def print_results(
    all_stats: ClusterTaxaStatisticsDataFrame,
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
) -> None:
    # For each top count taxon, print the average per geocode
    print_all_cluster_stats(all_stats)

    logger.info(f"Number of clusters: {geocode_cluster_dataframe.num_clusters()}")

    for cluster, geocodees in geocode_cluster_dataframe.iter_clusters_and_geocodees():
        print_cluster_stats(
            cluster,
            geocodees,
            all_stats,
        )
