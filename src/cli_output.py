import polars as pl
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from typing import List
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.matrices.geocode_distance import GeocodeDistanceMatrix # Added import
from src.dataframes.geocode import GeocodeDataFrame # Added import
from src.dataframes.taxonomy import TaxonomyDataFrame # Added import
from src.stats.permanova import run_permanova_test, format_permanova_results # Updated import path
import logging

logger = logging.getLogger(__name__)


# TODO: Refactor this to use the new ClusterSignificantDifferencesDataFrame


def print_cluster_stats(
    cluster: int,
    geocodes: List[str],
    stats_with_taxonomy: pl.DataFrame, # Changed type hint
) -> None:
    # stats = Stats.build(darwin_core_aggregations, geocode_filter=geocodes)
    print("-" * 10)
    print(f"cluster {cluster} (count: {len(geocodes)})")

    # Use the joined dataframe directly
    for kingdom, taxonRank, scientificName, count, average in (
        stats_with_taxonomy.filter(
            pl.col("cluster") == cluster,
        )
        .sort(by="count", descending=True)
        .limit(20)
        .select(["kingdom", "taxonRank", "scientificName", "count", "average"])
        .iter_rows(named=False)
    ):
        # Handle potential division by zero if all_average is 0
        try:
            # Filter the joined dataframe for the overall average
            all_average = (
                stats_with_taxonomy.filter(
                    pl.col("kingdom") == kingdom,
                    pl.col("scientificName") == scientificName,
                    pl.col("taxonRank") == taxonRank,
                    pl.col("cluster").is_null(),
                )
                .get_column("average")
                .item()
            )
            if all_average == 0:
                 percent_diff = float('inf') if average > 0 else 0.0 # Assign infinity or 0 if base is 0
            else:
                percent_diff = (average / all_average * 100) - 100

            # If the difference between the average of the cluster and the average of all is greater than 10%, print it
            if abs(percent_diff) > 10:
                # Print the percentage difference
                print(f"{scientificName} ({kingdom}) {taxonRank}:")
                print(
                    f"  - Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%"
                )
                print(f"  - Proportion: {average * 100:.2f}%")
                print(f"  - Count: {count}")
        except Exception as e:
             logger.warning(f"Could not calculate diff for {scientificName}: {e}")


def print_all_cluster_stats(stats_with_taxonomy: pl.DataFrame) -> None: # Changed type hint
    # Use the joined dataframe directly
    for kingdom, taxonRank, scientificName, count, average in (
        stats_with_taxonomy.filter(
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
    geocode_distance_matrix: GeocodeDistanceMatrix,
    geocode_dataframe: GeocodeDataFrame,
    taxonomy_dataframe: TaxonomyDataFrame, # Added argument
) -> None:
    # Join stats with taxonomy info
    stats_with_taxonomy = all_stats.df.join(
        taxonomy_dataframe.df.select(["taxonId", "kingdom", "taxonRank", "scientificName"]),
        on="taxonId",
        how="left" # Use left join to keep all stats rows
    )

    # Print overall stats using the joined dataframe
    print("\n--- Overall Top Taxa ---")
    print_all_cluster_stats(stats_with_taxonomy) # Pass joined dataframe
    print("-" * 24)

    # Print PERMANOVA results
    try:
        permanova_results = run_permanova_test(
            geocode_distance_matrix=geocode_distance_matrix,
            geocode_cluster_dataframe=geocode_cluster_dataframe,
            geocode_dataframe=geocode_dataframe,
        )
        print("\n" + format_permanova_results(permanova_results) + "\n")
    except Exception as e:
        logger.error(f"Failed to run or format PERMANOVA test: {e}")

    # Print stats per cluster using the joined dataframe
    logger.info(f"Number of clusters: {geocode_cluster_dataframe.num_clusters()}")
    print("\n--- Cluster Specific Stats ---")
    for cluster, geocodes in geocode_cluster_dataframe.iter_clusters_and_geocodes():
        print_cluster_stats(
            cluster,
            geocodes,
            stats_with_taxonomy, # Pass joined dataframe
        )
