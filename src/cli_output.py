import polars as pl
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.permanova_results import PermanovaResultsDataFrame
import logging

logger = logging.getLogger(__name__)


def print_results(
    # Removed unused arguments: all_stats, taxonomy_dataframe
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    permanova_results_dataframe: PermanovaResultsDataFrame,
) -> None:
    print("\n" + permanova_results_dataframe.format_results() + "\n")

    # Log number of clusters found
    logger.info(f"Number of clusters found: {geocode_cluster_dataframe.num_clusters()}")
    print("-" * 25) # Separator
