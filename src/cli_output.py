import polars as pl
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from typing import List
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.dataframes.permanova_results import PermanovaResultsDataFrame
from src.dataframes.cluster_significant_differences import ClusterSignificantDifferencesDataFrame # Added import
import logging

logger = logging.getLogger(__name__)


def print_cluster_stats(
    cluster: int,
    geocodes: List[str],
    significant_differences_df: ClusterSignificantDifferencesDataFrame,
    all_stats_df: ClusterTaxaStatisticsDataFrame,
    taxonomy_df: TaxonomyDataFrame,
) -> None:
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geocodes)})")

    # Filter significant differences for the current cluster
    cluster_diffs = significant_differences_df.df.filter(pl.col("cluster") == cluster)

    # Join with stats and taxonomy to get details
    cluster_details = cluster_diffs.join(
        all_stats_df.df.filter(pl.col("cluster") == cluster), on=["cluster", "taxonId"], how="inner"
    ).join(
        taxonomy_df.df.select(["taxonId", "kingdom", "taxonRank", "scientificName"]), on="taxonId", how="left"
    ).sort("percentage_difference", descending=True) # Sort by difference magnitude

    if cluster_details.height == 0:
        print("  No significant differences found.")
        return

    # Iterate and print
    for row in cluster_details.iter_rows(named=True):
        percent_diff = row["percentage_difference"]
        scientificName = row["scientificName"]
        kingdom = row["kingdom"]
        taxonRank = row["taxonRank"]
        average = row["average"]
        count = row["count"]

        print(f"{scientificName} ({kingdom}) {taxonRank}:")
        print(
            f"  - Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%"
        )
        print(f"  - Proportion: {average * 100:.2f}%")
        print(f"  - Count: {count}")


def print_all_cluster_stats(all_stats_df: ClusterTaxaStatisticsDataFrame, taxonomy_df: TaxonomyDataFrame) -> None:
    # Filter for overall stats (cluster is null) and join with taxonomy
    overall_stats_with_taxonomy = (
        all_stats_df.df.filter(pl.col("cluster").is_null())
        .join(
            taxonomy_df.df.select(["taxonId", "kingdom", "taxonRank", "scientificName"]),
            on="taxonId",
            how="left",
        )
        .sort(by="count", descending=True)
        .limit(5)
        .select(["kingdom", "taxonRank", "scientificName", "count", "average"])
    )

    # Iterate and print
    for kingdom, taxonRank, scientificName, count, average in overall_stats_with_taxonomy.iter_rows(named=False):
        print(f"{scientificName} ({kingdom}) {taxonRank}:")
        print(f"  - Proportion: {average * 100:.2f}%")
        print(f"  - Count: {count}")


def print_results(
    all_stats: ClusterTaxaStatisticsDataFrame,
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    taxonomy_dataframe: TaxonomyDataFrame,
    permanova_results_dataframe: PermanovaResultsDataFrame,
) -> None:
    # Build significant differences dataframe
    significant_differences_df = ClusterSignificantDifferencesDataFrame.build(all_stats)

    # Print overall stats
    print("\n--- Overall Top Taxa ---")
    # Pass the original dataframes, the function now handles the join
    print_all_cluster_stats(all_stats, taxonomy_dataframe)
    print("-" * 24)

    # Print PERMANOVA results using the new dataframe's method
    print("\n" + permanova_results_dataframe.format_results() + "\n")

    # Print stats per cluster
    logger.info(f"Number of clusters: {geocode_cluster_dataframe.num_clusters()}")
    print("\n--- Cluster Specific Stats ---")
    for cluster, geocodes in geocode_cluster_dataframe.iter_clusters_and_geocodes():
        print_cluster_stats(
            cluster=cluster,
            geocodes=geocodes,
            significant_differences_df=significant_differences_df,
            all_stats_df=all_stats,
            taxonomy_df=taxonomy_dataframe,
        )
