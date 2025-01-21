import polars as pl
from src import cluster_index, geohash
from src.cluster_stats import Stats
from src.darwin_core_aggregations import DarwinCoreAggregations
from typing import List
from src.types import Geohash
import logging

logger = logging.getLogger(__name__)


def print_cluster_stats(
    cluster: cluster_index.ClusterId,
    geohashes: List[geohash.Geohash],
    darwin_core_aggregations: DarwinCoreAggregations,
    all_stats: Stats,
) -> None:
    stats = Stats.build(darwin_core_aggregations, geohash_filter=geohashes)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geohashes)})")

    for kingdom, species, count in (
        stats.taxon.sort(by="count", descending=True)
        .limit(5)
        .select(["kingdom", "species", "count"])
        .collect()
        .iter_rows(named=False)
    ):
        average = (
            stats.taxon.filter(
                pl.col("kingdom") == kingdom, pl.col("species") == species
            )
            .collect()
            .get_column("average")
            .item()
        )
        all_average = (
            all_stats.taxon.filter(
                pl.col("kingdom") == kingdom, pl.col("species") == species
            )
            .collect()
            .get_column("average")
            .item()
        )

        # If the difference between the average of the cluster and the average of all is greater than 20%, print it
        percent_diff = (average / all_average * 100) - 100
        if abs(percent_diff) > 20:
            # Print the percentage difference
            print(f"{species} ({kingdom}):")
            print(
                f"  - Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%"
            )
            print(f"  - Proportion: {average * 100:.2f}%")
            print(f"  - Count: {count}")


def print_all_cluster_stats(
    darwin_core_aggregations: DarwinCoreAggregations, all_stats: Stats
) -> None:
    for kingdom, species, count in (
        all_stats.taxon.sort(by="count", descending=True)
        .limit(5)
        .select(["kingdom", "species", "count"])
        .collect()
        .iter_rows(named=False)
    ):
        average = (
            all_stats.taxon.filter(
                pl.col("kingdom") == kingdom, pl.col("species") == species
            )
            .collect()
            .get_column("average")
            .item()
        )
        print(f"{species} ({kingdom}):")
        print(f"  - Proportion: {average * 100:.2f}%")
        print(f"  - Count: {count}")


def print_results(
    darwin_core_aggregations: DarwinCoreAggregations,
    all_stats: Stats,
    clusters: cluster_index.ClusterIndex,
) -> None:
    # For each top count taxon, print the average per geohash
    print_all_cluster_stats(darwin_core_aggregations, all_stats)

    logger.info(f"Number of clusters: {cluster_index.num_clusters(clusters)}")

    for cluster, geohashes in cluster_index.iter_clusters_and_geohashes(clusters):
        print_cluster_stats(cluster, geohashes, darwin_core_aggregations, all_stats)
